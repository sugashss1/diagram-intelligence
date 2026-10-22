"""Ground-truth-free structural evaluation for reconstructed circuit JSON.

These functions measure output validity and geometric representation consistency.
They must not be reported as connectivity accuracy when reference connectivity
annotations are unavailable.
"""

import json
import statistics
from pathlib import Path

import cv2
import numpy as np

from text.circuit_json import topology_to_mermaid, validate_circuit


def _raster_lines(lines, image_shape, thickness=2):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for line in lines or []:
        x1, y1, x2, y2 = (round(float(value)) for value in line)
        cv2.line(mask, (x1, y1), (x2, y2), 255, thickness, cv2.LINE_AA)
    return mask


def _mask_iou(first, second):
    first, second = first > 0, second > 0
    union = np.count_nonzero(first | second)
    return None if union == 0 else round(float(np.count_nonzero(first & second) / union), 6)


def _symmetric_chamfer(first, second):
    first, second = first > 0, second > 0
    if not np.any(first) or not np.any(second):
        return None
    first_distance = cv2.distanceTransform((~first).astype(np.uint8), cv2.DIST_L2, 3)
    second_distance = cv2.distanceTransform((~second).astype(np.uint8), cv2.DIST_L2, 3)
    return round(float((second_distance[first].mean() + first_distance[second].mean()) / 2), 6)


def evaluate_reconstruction(circuit, topology, raw_lines, merged_lines, image_shape):
    """Evaluate one reconstructed circuit without claiming visual/graph accuracy."""
    errors = validate_circuit(circuit)
    mermaid = topology_to_mermaid(circuit, topology)
    raw_mask = _raster_lines(raw_lines, image_shape)
    merged_mask = _raster_lines(merged_lines, image_shape)
    return {
        "schema_valid": not errors,
        "validation_errors": errors,
        "reference_integrity_pass": not errors,
        "mermaid_nonempty": mermaid.startswith("graph "),
        "raw_to_regenerated_wire_iou": _mask_iou(raw_mask, merged_mask),
        "raw_to_regenerated_chamfer_px": _symmetric_chamfer(raw_mask, merged_mask),
        "ambiguous_crossing_count": sum(
            junction.get("connectivity_status") in {"requires_review", "requires_human_review"}
            for junction in circuit.get("junctions", [])
        ),
        "metric_interpretation": "Structural-consistency and representation-preservation metrics, not connectivity accuracy.",
    }


def summarise_evaluation(rows):
    """Summarise per-image structural results for a held-out corpus."""
    completed = [row for row in rows if row.get("status", "completed") == "completed"]
    if not rows:
        return {"images_requested": 0, "images_completed": 0, "metric_interpretation": "No rows supplied."}
    values = lambda key: [row[key] for row in completed if row.get(key) is not None]
    return {
        "images_requested": len(rows),
        "images_completed": len(completed),
        "images_failed": len(rows) - len(completed),
        "completion_rate": round(len(completed) / len(rows), 6),
        "schema_validity_rate": round(sum(bool(row.get("schema_valid")) for row in completed) / len(completed), 6) if completed else None,
        "reference_integrity_rate": round(sum(bool(row.get("reference_integrity_pass")) for row in completed) / len(completed), 6) if completed else None,
        "mermaid_graph_preservation_rate": round(sum(bool(row.get("mermaid_nonempty")) for row in completed) / len(completed), 6) if completed else None,
        "mean_wire_iou": round(float(np.mean(values("raw_to_regenerated_wire_iou"))), 6) if values("raw_to_regenerated_wire_iou") else None,
        "median_wire_chamfer_px": round(float(statistics.median(values("raw_to_regenerated_chamfer_px"))), 4) if values("raw_to_regenerated_chamfer_px") else None,
        "metric_interpretation": "Downstream measures are ground-truth-free structural consistency and robustness, not connectivity accuracy.",
        "excluded_metrics": ["endpoint attachment rate", "terminal attachment rate", "dangling-terminal rate", "SSIM"],
    }


def write_evaluation(rows, output_dir):
    """Persist raw rows and a paper-ready summary without touching production output."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarise_evaluation(rows)
    (output_dir / "per_image_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (output_dir / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
