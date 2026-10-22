"""Reproducible circuit-diagram pipeline entry point.

Evaluation is intentionally separate under src/evaluation. This script runs the
production flow: perception, canonical JSON, confidence-gated connectivity
recovery, topology/white-canvas reconstruction, and optional multimodal RAG VQA.
"""

from dotenv import load_dotenv

load_dotenv()


import argparse
import json
import sys
from pathlib import Path

import cv2
import pandas as pd

from image import component_detection, line_detection, preprocess, text_extraction
from image.connectivity_recovery import apply_review_decisions, build_review_candidates, review_candidates_with_vlm
from image.junction_detection import draw_junction
from text.circuit_json import (
    build_circuit_json,
    build_topology,
    format_netlist,
    render_reconstructed_circuit,
    save_circuit,
    topology_to_mermaid,
    topology_to_netlist,
)
from text.topology_visualization import render_topology_png


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Recover a circuit graph from an engineering diagram.")
    parser.add_argument("input", nargs="?", type=Path, default=Path("test") / "full-adder-circuit.png", help="Circuit image path.")
    parser.add_argument("--weights", type=Path, default=Path("best.pt"), help="YOLO component-detection weights.")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Directory for reproducible pipeline artifacts.")
    parser.add_argument("--skip-vlm-review", action="store_true", help="Do not call the configured VLM; resolve ambiguous '+' crossings conservatively as crossovers.")
    parser.add_argument("--question", help="Optional VQA question. Requires --knowledge-base and NVIDIA_API_KEY/NVAPI_KEY.")
    parser.add_argument("--knowledge-base", type=Path, default=Path("knowledge_base"), help="Local directory containing PDFs, JSON/Markdown symbol rules, and manuals for hybrid retrieval.")
    parser.add_argument("--rebuild-knowledge-index", action="store_true", help="Rebuild the persisted hybrid retrieval index before VQA.")
    return parser.parse_args(argv)


def _write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write image: {path}")


def _empty_review_result():
    return {"schema_version": "circuit-vlm-junction-crossover/v1", "model": None, "threshold": 0.80, "decisions": [], "failures": []}


def _stage(message):
    print(f"\n[Pipeline] {message}", flush=True)


def run_pipeline(args):
    if not args.input.is_file():
        raise FileNotFoundError(f"Could not load input image: {args.input}")
    if not args.weights.is_file():
        raise FileNotFoundError(f"Could not find YOLO weights: {args.weights}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    connectivity_dir = output_dir / "connectivity"
    connectivity_dir.mkdir(parents=True, exist_ok=True)

    _stage(f"Loading input image: {args.input}")
    source_image = preprocess.load(str(args.input))
    if source_image is None or source_image.size == 0:
        raise FileNotFoundError(f"Could not decode input image: {args.input}")

    _stage("Detecting circuit components with YOLO")
    yolo = component_detection.yolo_component_detection(str(args.weights))
    component_result = yolo.model(source_image, verbose=False)[0]
    print(f"[Pipeline] Detected {len(component_result.boxes)} components.", flush=True)
    _write_image(output_dir / "components_bounding_boxes.png", yolo.draw_yolo(source_image.copy(), result=component_result))

    cleaned_image, components = yolo.remove_components(source_image.copy(), result=component_result)
    _write_image(output_dir / "components_removed.png", cleaned_image)

    _stage("Removing OCR text after component cleanup")
    paddle_ocr = text_extraction.ocr_paddle()
    cleaned_image, texts = paddle_ocr.remove_text(cleaned_image)
    print(f"[Pipeline] Removed {len(texts.get('res', {}).get('rec_boxes', []))} OCR text regions.", flush=True)
    _write_image(output_dir / "text_removed.png", cleaned_image)

    _stage("Extracting and merging candidate wire segments")
    line_detector = line_detection.line_detection_lsd()
    raw_lines = line_detector.predict(cleaned_image)
    _write_image(output_dir / "lines_raw.png", line_detector.draw_lines(cleaned_image.copy(), raw_lines))
    merged_lines = line_detection.merge_segments(raw_lines)
    print(f"[Pipeline] Raw segments: {len(raw_lines)}; merged wire segments: {len(merged_lines)}.", flush=True)
    merged_image = line_detector.draw_lines(cleaned_image.copy(), merged_lines)
    _write_image(output_dir / "lines_merged.png", merged_image)

    _stage("Classifying geometric junctions")
    junction_image, junctions = draw_junction(merged_image.copy(), merged_lines)
    print(f"[Pipeline] Classified {len(junctions)} junctions.", flush=True)
    _write_image(output_dir / "junctions.png", junction_image)
    pd.DataFrame(raw_lines, columns=["x0", "y0", "x1", "y1"]).to_csv(output_dir / "lines_raw.csv", index=False)
    pd.DataFrame(merged_lines, columns=["x0", "y0", "x1", "y1"]).to_csv(output_dir / "lines_merged.csv", index=False)
    junctions.to_csv(output_dir / "junctions.csv", index=False)

    _stage("Creating canonical circuit JSON")
    circuit = build_circuit_json(
        image_path=args.input,
        image=source_image,
        components_result=components,
        lines=merged_lines,
        junctions=junctions,
        ocr_data=texts,
        component_weights_path=args.weights,
    )
    save_circuit(circuit, connectivity_dir / "canonical_circuit_before_review.json")
    candidates = build_review_candidates(circuit)
    print(f"[Pipeline] Ambiguous '+' junctions requiring review: {len(candidates)}.", flush=True)
    (connectivity_dir / "candidate_connectivity_records.json").write_text(json.dumps(candidates, indent=2), encoding="utf-8")

    if args.skip_vlm_review:
        _stage("Skipping VLM crossover review; ambiguous crossings will be conservatively treated as crossovers")
        review_result = _empty_review_result()
        (connectivity_dir / "vlm_review_skipped.json").write_text(json.dumps(review_result, indent=2), encoding="utf-8")
    else:
        _stage("Reviewing ambiguous crossings with the configured VLM when credentials are available")
        review_result = review_candidates_with_vlm(circuit, args.input, connectivity_dir / "vlm_review")
        if review_result["failures"]:
            print(f"[Pipeline] {len(review_result['failures'])} VLM decisions used the conservative crossover fallback.", flush=True)
    circuit = apply_review_decisions(circuit, review_result)
    _stage("Materialising electrical nets and rendering the white-canvas reconstruction")
    topology = build_topology(circuit)
    save_circuit(circuit, connectivity_dir / "canonical_circuit.json")
    (connectivity_dir / "circuit_topology.json").write_text(json.dumps(topology, indent=2), encoding="utf-8")
    mermaid = topology_to_mermaid(circuit, topology)
    mermaid_path = connectivity_dir / "canonical_topology.mmd"
    mermaid_path.write_text(mermaid, encoding="utf-8")
    netlist = topology_to_netlist(circuit, topology)
    (connectivity_dir / "canonical_netlist.json").write_text(json.dumps(netlist, indent=2), encoding="utf-8")
    netlist_text = format_netlist(netlist)
    (connectivity_dir / "canonical_netlist.txt").write_text(netlist_text, encoding="utf-8")
    topology_image_path = connectivity_dir / "mermaid_topology.png"
    renderer = render_topology_png(circuit, topology, mermaid_path, topology_image_path)
    print(f"[Pipeline] Saved Mermaid topology image ({renderer}): {topology_image_path}", flush=True)
    reconstruction_path = connectivity_dir / "reconstructed_circuit.png"
    render_reconstructed_circuit(circuit, reconstruction_path, topology=topology)
    print(f"[Pipeline] Saved reconstructed circuit image: {reconstruction_path}", flush=True)
    print(f"\n[Pipeline] {netlist_text}", flush=True)
    print("[Pipeline] MERMAID (derived from corrected canonical JSON)", flush=True)
    print(mermaid, flush=True)

    vqa_response = None
    if args.question:
        knowledge_base = args.knowledge_base
        if not knowledge_base.is_dir():
            raise FileNotFoundError(f"Knowledge-base directory does not exist: {knowledge_base}")
        from retrieval.knowledge_base import build_or_load_index
        from retrieval.vqa import answer_circuit_question, save_vqa_response

        _stage(f"Building hybrid retrieval context from local knowledge base: {knowledge_base}")
        retriever = build_or_load_index(knowledge_base, output_dir / "retrieval" / "knowledge_index.json", rebuild=args.rebuild_knowledge_index)
        _stage("Sending the question, original image, recovered graph, and retrieved context to the VLM")
        try:
            vqa_response = answer_circuit_question(args.question, circuit, topology, args.input, retriever)
            save_vqa_response(vqa_response, output_dir / "vqa_response.json")
        except RuntimeError as exc:
            vqa_response = {"status": "failed", "error": str(exc)}
            (output_dir / "vqa_error.json").write_text(json.dumps(vqa_response, indent=2), encoding="utf-8")
            print(f"[Pipeline] VQA did not complete: {exc}. The circuit-recovery artifacts are still available.", flush=True)

    summary = {
        "components": len(circuit["components"]),
        "raw_lines": len(raw_lines),
        "merged_lines": len(merged_lines),
        "junctions": len(circuit["junctions"]),
        "ambiguous_candidates": len(candidates),
        "topology_nets": len(topology["nets"]),
        "output_dir": str(output_dir.resolve()),
        "vqa_answered": bool(vqa_response and "answer" in vqa_response),
    }
    if bool(vqa_response and "answer" in vqa_response):
        summary["output"]=vqa_response["answer"]

    _stage(f"Complete. Artifacts written to: {summary['output_dir']}")
    return summary


def main(argv=None):
    args = parse_args(argv)
    summary = run_pipeline(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
