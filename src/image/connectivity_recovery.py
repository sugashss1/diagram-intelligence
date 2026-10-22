"""Confidence-gated recovery of ambiguous circuit wire crossings.

Algorithmic geometry identifies a four-way ``+`` but cannot decide whether the
diagram convention means an electrical intersection or two crossing wires. This
module creates review boards and accepts only a strict, confidence-qualified VLM
decision. Ambiguous evidence is resolved conservatively as a crossover: this
avoids inventing an electrical connection that is not visible in the drawing.
"""

import base64
import copy
import json
import os
from pathlib import Path
from urllib import error, request

import cv2


DEFAULT_MODEL = "meta/llama-3.2-11b-vision-instruct"
DEFAULT_INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
VALID_DECISIONS = {"connected_intersection", "crossover", "unclear"}


def build_review_candidates(circuit):
    """Return only the unresolved `+` junctions that need visual review."""
    return [
        {
            "candidate_id": junction["id"],
            "junction_id": junction["id"],
            "kind": "intersection_or_crossover",
            "point": junction["point"],
            "incident_wire_ids": junction["incident_wire_ids"],
            "review_status": "requires_review",
        }
        for junction in circuit.get("junctions", [])
        if junction.get("detected_type") == "+" and junction.get("connectivity_status") == "requires_review"
    ]


def _image_part(path):
    suffix = Path(path).suffix.lower()
    mime_type = ".jpg" if suffix == ".jpg" else suffix.lstrip(".") or "png"
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/{mime_type};base64,{encoded}"}}


def _extract_json_object(text):
    if isinstance(text, dict):
        return text
    decoder = json.JSONDecoder()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, TypeError):
        pass
    for index, character in enumerate(str(text)):
        if character != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _read_response_content(payload):
    content = payload["choices"][0]["message"]["content"]
    if isinstance(content, list):
        return "".join(item.get("text", "") for item in content if isinstance(item, dict))
    return str(content)


def _normalise_decision(payload, candidate, threshold):
    payload = payload or {}
    decision = str(payload.get("decision", "unclear")).strip().lower()
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    high_confidence_decision = decision in {"connected_intersection", "crossover"} and confidence >= threshold
    fallback_to_crossover = not high_confidence_decision
    if fallback_to_crossover:
        # A false merge changes circuit meaning; keeping the two wires separate
        # is the safe correction for unclear, malformed, and low-confidence VLM
        # replies. Preserve the original evidence and confidence for audit.
        decision = "crossover"
    evidence = payload.get("supporting_evidence", {})
    if not isinstance(evidence, dict):
        evidence = {"description": str(evidence)}
    return {
        "review_id": candidate["candidate_id"],
        "candidate_id": candidate["candidate_id"],
        "junction_id": candidate["junction_id"],
        "decision": decision,
        "confidence": round(confidence, 4),
        "graph_action": {
            "connected_intersection": "merge_incident_wires",
            "crossover": "keep_wires_separate",
            "unclear": "retain_for_human_review",
        }[decision],
        "supporting_evidence": {
            "visible_junction_dot": bool(evidence.get("visible_junction_dot", False)),
            "plain_crossing_without_dot": bool(evidence.get("plain_crossing_without_dot", False)),
            "wire_jump_or_bridge_visible": bool(evidence.get("wire_jump_or_bridge_visible", False)),
            "description": str(evidence.get("description", "No confidence-qualified decision available.")),
        },
        "resolution_basis": "conservative_crossover_fallback" if fallback_to_crossover else "vlm_confidence_qualified",
    }


def render_review_board(image, candidate, output_path, crop_radius_px=90):
    """Create a full-context + native-resolution crop board for a crossing."""
    if image is None or image.size == 0:
        raise ValueError("A non-empty source image is required for a review board.")
    point = tuple(round(value) for value in candidate["point"])
    overview = image.copy()
    cv2.circle(overview, point, 18, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.putText(
        overview,
        candidate["candidate_id"],
        (point[0] + 20, max(18, point[1] - 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    x, y = point
    x1, x2 = max(0, x - crop_radius_px), min(image.shape[1], x + crop_radius_px)
    y1, y2 = max(0, y - crop_radius_px), min(image.shape[0], y + crop_radius_px)
    crop = image[y1:y2, x1:x2].copy()
    cv2.circle(crop, (x - x1, y - y1), 18, (0, 0, 255), 3, cv2.LINE_AA)
    target_height = max(overview.shape[0], crop.shape[0])
    if overview.shape[0] != target_height:
        overview = cv2.copyMakeBorder(overview, 0, target_height - overview.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    if crop.shape[0] != target_height:
        crop = cv2.copyMakeBorder(crop, 0, target_height - crop.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    board = cv2.hconcat([overview, cv2.copyMakeBorder(crop, 0, 0, 16, 0, cv2.BORDER_CONSTANT, value=(235, 235, 235))])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), board):
        raise OSError(f"Could not write review board: {output_path}")
    return output_path


def _review_prompt(candidate):
    return f"""You are a precise visual reviewer for an engineering circuit diagram.

Review ID: {candidate['candidate_id']}
Junction ID: {candidate['junction_id']}
Candidate point: {candidate['point']}

The image contains the full diagram at left and a native-resolution crop at right.
Decide only the visual convention at the highlighted '+'.

Definitions:
- connected_intersection: a filled junction dot is visibly centred at the crossing. All incident wires are electrically connected.
- crossover: two wires cross without a filled junction dot, or a wire-jump/bridge indicates that they do not connect.
- unclear: image quality does not let you see whether a junction dot or bridge exists.

Rules:
1. A plain horizontal and vertical wire crossing is not a dot.
2. Do not infer connectivity from circuit function, labels, likely full-adder structure, or detector predictions.
3. Return connected_intersection or crossover only when confidence is at least 0.80. Otherwise return unclear.
4. Return JSON only, exactly with this shape:
{{
  "review_id": "{candidate['candidate_id']}",
  "junction_id": "{candidate['junction_id']}",
  "decision": "connected_intersection|crossover|unclear",
  "confidence": 0.0,
  "supporting_evidence": {{
    "visible_junction_dot": false,
    "plain_crossing_without_dot": false,
    "wire_jump_or_bridge_visible": false,
    "description": "short visual evidence only"
  }}
}}"""


def _post_json(url, api_key, payload, timeout_seconds):
    data = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with request.urlopen(http_request, timeout=timeout_seconds) as response:
        return response.read().decode("utf-8")


def review_candidates_with_vlm(
    circuit,
    source_image_path,
    output_dir,
    api_key=None,
    invoke_url=None,
    model_name=None,
    confidence_threshold=0.80,
    timeout_seconds=180,
):
    """Review candidates with Llama Vision, safely falling back to human review.

    A missing key or any transport/schema failure does not abort the pipeline;
    the affected candidate becomes ``unclear`` with an auditable failure record.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(source_image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not load source image for VLM review: {source_image_path}")
    api_key = api_key or os.environ.get("NVIDIA_API_KEY") or os.environ.get("NVAPI_KEY")
    invoke_url = invoke_url or os.environ.get("NVIDIA_INVOKE_URL", DEFAULT_INVOKE_URL)
    model_name = model_name or os.environ.get("NVIDIA_VLM_MODEL", DEFAULT_MODEL)
    candidates = build_review_candidates(circuit)
    decisions, failures, manifest = [], [], []
    for candidate in candidates:
        board_path = render_review_board(image, candidate, output_dir / f"{candidate['candidate_id']}_review_board.png")
        prompt = _review_prompt(candidate)
        raw_path = output_dir / f"{candidate['candidate_id']}_raw_response.json"
        manifest.append({"candidate_id": candidate["candidate_id"], "junction_id": candidate["junction_id"], "review_board": str(board_path), "raw_response": str(raw_path)})
        if not api_key:
            decisions.append(_normalise_decision(None, candidate, confidence_threshold))
            failures.append({"candidate_id": candidate["candidate_id"], "junction_id": candidate["junction_id"], "error": "NVIDIA_API_KEY/NVAPI_KEY is not configured."})
            continue
        payload = {
            "model": model_name,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 300,
            "stream": False,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, _image_part(board_path)]}],
        }
        try:
            raw = _post_json(invoke_url, api_key, payload, timeout_seconds)
            raw_path.write_text(raw, encoding="utf-8")
            response_payload = json.loads(raw)
            decision_payload = _extract_json_object(_read_response_content(response_payload))
            decisions.append(_normalise_decision(decision_payload, candidate, confidence_threshold))
        except (OSError, ValueError, KeyError, error.URLError, error.HTTPError, json.JSONDecodeError) as exc:
            decisions.append(_normalise_decision(None, candidate, confidence_threshold))
            failures.append({"candidate_id": candidate["candidate_id"], "junction_id": candidate["junction_id"], "error": str(exc), "raw_response_path": str(raw_path)})
    result = {
        "schema_version": "circuit-vlm-junction-crossover/v1",
        "model": model_name,
        "threshold": confidence_threshold,
        "decisions": decisions,
        "failures": failures,
    }
    (output_dir / "prompt_contract.txt").write_text(_review_prompt(candidates[0]) if candidates else "No ambiguous junction candidates.", encoding="utf-8")
    (output_dir / "review_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "junction_crossover_decisions.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def apply_review_decisions(circuit, review_result, confidence_threshold=0.80):
    """Return a copy whose reviewed `+` junctions have final connectivity status.

    A VLM response that is unclear or below threshold is represented as an
    ``assumed_crossover``. It therefore never merges wires, while retaining an
    explicit audit trail that it was a conservative correction rather than a
    confidence-qualified visual conclusion.
    """
    updated = copy.deepcopy(circuit)
    decisions = {item.get("junction_id"): item for item in review_result.get("decisions", [])}
    for junction in updated.get("junctions", []):
        if junction.get("connectivity_status") != "requires_review":
            continue
        decision = decisions.get(junction["id"], {})
        confidence = float(decision.get("confidence", 0.0) or 0.0)
        if decision.get("decision") == "connected_intersection" and confidence >= confidence_threshold:
            junction["connectivity_status"] = "confirmed_connected"
        elif decision.get("decision") == "crossover":
            junction["connectivity_status"] = (
                "assumed_crossover"
                if decision.get("resolution_basis") == "conservative_crossover_fallback"
                else "confirmed_crossover"
            )
        else:
            junction["connectivity_status"] = "assumed_crossover"
        junction["review"] = {
            "decision": decision.get("decision", "unclear"),
            "confidence": round(confidence, 4),
            "graph_action": decision.get("graph_action", "retain_for_human_review"),
            "resolution_basis": decision.get("resolution_basis", "conservative_crossover_fallback"),
        }
    # Keep derived graph matrices coherent with the reviewed canonical records.
    from text.circuit_json import refresh_circuit_matrices

    refresh_circuit_matrices(updated)
    return updated
