"""Canonical circuit JSON, topology construction, and white-canvas rendering.

The image pipeline supplies detector boxes, merged wire segments, junctions, and
OCR output. This module converts those transient objects into a portable,
validated representation that downstream recovery, VQA, and rendering can use
without importing Ultralytics or PaddleOCR result classes.
"""

import json
import math
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


SCHEMA_VERSION = "circuit-connectivity/v1"
TOPOLOGY_SCHEMA_VERSION = "circuit-topology/v1"


class CircuitValidationError(ValueError):
    """Raised when a canonical circuit document violates its invariants."""


class _UnionFind:
    def __init__(self, values):
        self.parent = {value: value for value in values}

    def find(self, value):
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, first, second):
        first, second = self.find(first), self.find(second)
        if first != second:
            self.parent[second] = first


def _number(value, digits=2):
    return round(float(value), digits)


def _point(value):
    return [_number(value[0]), _number(value[1])]


def _distance(first, second):
    return math.hypot(float(first[0]) - float(second[0]), float(first[1]) - float(second[1]))


def _distance_to_box(point, box):
    x1, y1, x2, y2 = (float(value) for value in box)
    dx = max(x1 - point[0], 0.0, point[0] - x2)
    dy = max(y1 - point[1], 0.0, point[1] - y2)
    return math.hypot(dx, dy)


def _normalise_label(value):
    value = re.sub(r"\s+", "_", str(value).strip())
    value = re.sub(r"[^A-Za-z0-9_]", "", value)
    return value or "unlabelled"


def _result_name(result, box):
    class_id = int(box.cls[0])
    return str(result.names[class_id])


def _extract_ocr_records(ocr_data):
    res = (ocr_data or {}).get("res", ocr_data or {})
    boxes = res.get("rec_boxes", [])
    texts = res.get("rec_texts", [])
    scores = res.get("rec_scores", res.get("rec_score", []))
    records = []
    for index, (box, text) in enumerate(zip(boxes, texts), start=1):
        if len(box) != 4 or not str(text).strip():
            continue
        x1, y1, x2, y2 = (float(value) for value in box)
        score = scores[index - 1] if index - 1 < len(scores) else None
        records.append(
            {
                "id": f"ocr-{index:03d}",
                "text": str(text).strip(),
                "bbox": [_number(x1), _number(y1), _number(x2), _number(y2)],
                "score": _number(score, 4) if score is not None else None,
                "center": [_number((x1 + x2) / 2), _number((y1 + y2) / 2)],
            }
        )
    return records


def _attach_terminals(wires, components, attachment_radius_px, cluster_radius_px):
    """Attach wire endpoints to component boxes and cluster same-pin endpoints."""
    endpoint_owner = {}
    groups_by_component = defaultdict(list)

    for wire in wires:
        for endpoint_index, point in enumerate(wire["points"]):
            nearest = min(
                components,
                key=lambda component: _distance_to_box(point, component["bbox"]),
                default=None,
            )
            if nearest is None:
                continue
            distance = _distance_to_box(point, nearest["bbox"])
            if distance > attachment_radius_px:
                continue

            target_group = next(
                (
                    group
                    for group in groups_by_component[nearest["id"]]
                    if _distance(point, group["point"]) <= cluster_radius_px
                ),
                None,
            )
            if target_group is None:
                target_group = {"point": list(point), "endpoints": []}
                groups_by_component[nearest["id"]].append(target_group)
            target_group["endpoints"].append((wire["id"], endpoint_index, point))

    terminals = []
    for component in components:
        groups = groups_by_component[component["id"]]
        groups.sort(key=lambda group: (group["point"][0], group["point"][1]))
        for index, group in enumerate(groups, start=1):
            terminal_id = f"terminal-{component['id'].split('-')[-1]}-{index:02d}"
            attached_wire_ids = sorted({endpoint[0] for endpoint in group["endpoints"]})
            terminals.append(
                {
                    "id": terminal_id,
                    "component_id": component["id"],
                    "point": _point(group["point"]),
                    "attached_wire_ids": attached_wire_ids,
                    "evidence": {
                        "attachment_radius_px": _number(attachment_radius_px),
                        "endpoint_count": len(group["endpoints"]),
                        "cluster_radius_px": _number(cluster_radius_px),
                    },
                }
            )
            for wire_id, endpoint_index, _ in group["endpoints"]:
                endpoint_owner[(wire_id, endpoint_index)] = terminal_id
    return terminals, endpoint_owner


def _build_matrices(wires, junctions, terminals, ports, open_endpoints):
    wire_ids = [wire["id"] for wire in wires]
    nodes = []
    for junction in junctions:
        nodes.append(
            {
                "id": junction["id"],
                "wire_ids": junction["incident_wire_ids"],
                "confirmed": junction["connectivity_status"] == "confirmed_connected",
            }
        )
    for terminal in terminals:
        nodes.append({"id": terminal["id"], "wire_ids": terminal["attached_wire_ids"], "confirmed": True})
    for port in ports:
        nodes.append({"id": port["id"], "wire_ids": [port["wire_id"]], "confirmed": True})
    for endpoint in open_endpoints:
        nodes.append({"id": endpoint["id"], "wire_ids": [endpoint["wire_id"]], "confirmed": False})

    columns = {wire_id: index for index, wire_id in enumerate(wire_ids)}
    geometric = np.zeros((len(nodes), len(wire_ids)), dtype=np.uint8)
    confirmed = np.zeros_like(geometric)
    for row, node in enumerate(nodes):
        for wire_id in node["wire_ids"]:
            if wire_id not in columns:
                continue
            geometric[row, columns[wire_id]] = 1
            if node["confirmed"]:
                confirmed[row, columns[wire_id]] = 1
    adjacency = confirmed @ confirmed.T
    np.fill_diagonal(adjacency, 0)
    return {
        "wire_columns": wire_ids,
        "topology_node_rows": [node["id"] for node in nodes],
        "geometric_incidence_matrix": geometric.tolist(),
        "confirmed_incidence_matrix": confirmed.tolist(),
        "confirmed_topology_adjacency_matrix": adjacency.tolist(),
    }


def refresh_circuit_matrices(circuit):
    """Rebuild JSON-derived matrices after a junction review changes status.

    The matrices are a cached representation of the canonical records, so they
    must never outlive a connectivity decision.  The input is updated in place
    and returned to make the intended pipeline use explicit.
    """
    circuit["matrices"] = _build_matrices(
        circuit.get("wire_segments", []),
        circuit.get("junctions", []),
        circuit.get("terminals", []),
        circuit.get("external_ports", []),
        circuit.get("open_endpoints", []),
    )
    return circuit


def build_circuit_json(
    image_path,
    image,
    components_result,
    lines,
    junctions,
    ocr_data=None,
    component_weights_path=None,
    attachment_radius_px=None,
    terminal_cluster_radius_px=6,
    junction_radius_px=10,
):
    """Create a validated canonical JSON document from committed pipeline outputs.

    A detected ``+`` remains ``requires_review``. It is deliberately not
    materialised as an electrical connection until connectivity recovery gives a
    confidence-qualified connected-intersection decision.
    """
    if image is None or image.size == 0:
        raise ValueError("A non-empty image is required to create circuit JSON.")

    height, width = image.shape[:2]
    attachment_radius_px = attachment_radius_px or float(np.linalg.norm(np.asarray(image.shape[:2])) / 50)
    components = []
    for index, box in enumerate(components_result.boxes, start=1):
        x1, y1, x2, y2 = (float(value) for value in box.xyxy[0])
        components.append(
            {
                "id": f"component-{index:03d}",
                "kind": _result_name(components_result, box),
                "role": "electrical_component",
                "bbox": [_number(x1), _number(y1), _number(x2), _number(y2)],
                "center": [_number((x1 + x2) / 2), _number((y1 + y2) / 2)],
                "confidence": _number(box.conf[0], 4),
            }
        )

    wires = []
    for index, line in enumerate([] if lines is None else lines, start=1):
        x1, y1, x2, y2 = (float(value) for value in line)
        wires.append(
            {
                "id": f"wire-{index:03d}",
                "source_segment_index": index - 1,
                "points": [[_number(x1), _number(y1)], [_number(x2), _number(y2)]],
                "length_px": _number(math.hypot(x2 - x1, y2 - y1)),
            }
        )
    wire_ids = {wire["id"] for wire in wires}

    junction_records = []
    if junctions is not None:
        for index, row in enumerate(junctions.itertuples(index=False), start=1):
            detected_type = str(row.type)
            if detected_type == "+":
                classification, status = "intersection_or_crossover", "requires_review"
            elif detected_type == "T":
                classification, status = "t_intersection", "confirmed_connected"
            else:
                classification, status = "bend", "confirmed_connected"
            incident = [
                f"wire-{int(segment_index) + 1:03d}"
                for segment_index in row.segment_indices
                if f"wire-{int(segment_index) + 1:03d}" in wire_ids
            ]
            junction_records.append(
                {
                    "id": f"junction-{index:03d}",
                    "point": [_number(row.x), _number(row.y)],
                    "detected_type": detected_type,
                    "classification": classification,
                    "connectivity_status": status,
                    "degree": int(row.degree),
                    "source_segment_indices": [int(value) for value in row.segment_indices],
                    "incident_wire_ids": incident,
                    "nearby_crossover_marker_ids": [],
                }
            )

    terminals, endpoint_owner = _attach_terminals(
        wires,
        components,
        attachment_radius_px,
        terminal_cluster_radius_px,
    )
    open_endpoints = []
    connections = []
    confirmed_junctions = [
        junction for junction in junction_records if junction["connectivity_status"] == "confirmed_connected"
    ]
    for wire in wires:
        for endpoint_index, point in enumerate(wire["points"]):
            node_id = endpoint_owner.get((wire["id"], endpoint_index))
            node_kind = "component_terminal" if node_id else None
            status = "confirmed_by_geometry" if node_id else "unresolved"
            if node_id is None:
                nearest = min(confirmed_junctions, key=lambda item: _distance(point, item["point"]), default=None)
                if nearest is not None and _distance(point, nearest["point"]) <= junction_radius_px:
                    node_id, node_kind, status = nearest["id"], "junction", "confirmed_by_geometry"
            if node_id is None:
                node_id = f"open-end-{len(open_endpoints) + 1:03d}"
                node_kind = "open_endpoint"
                open_endpoints.append(
                    {
                        "id": node_id,
                        "wire_id": wire["id"],
                        "endpoint_index": endpoint_index,
                        "point": _point(point),
                        "status": "unresolved",
                    }
                )
            connections.append(
                {
                    "id": f"connection-{len(connections) + 1:03d}",
                    "wire_id": wire["id"],
                    "endpoint_index": endpoint_index,
                    "point": _point(point),
                    "topology_node_id": node_id,
                    "topology_node_kind": node_kind,
                    "status": status,
                }
            )

    ocr_records = _extract_ocr_records(ocr_data)
    labels_by_open_endpoint = defaultdict(list)
    label_radius = max(90.0, min(width, height) * 0.30)
    for record in ocr_records:
        endpoint = min(open_endpoints, key=lambda item: _distance(record["center"], item["point"]), default=None)
        if endpoint is not None:
            distance = _distance(record["center"], endpoint["point"])
            if distance <= label_radius:
                record["assigned_open_endpoint_id"] = endpoint["id"]
                record["distance_to_endpoint_px"] = _number(distance)
                labels_by_open_endpoint[endpoint["id"]].append(record)

    ports, retained_open_endpoints = [], []
    port_by_open_endpoint = {}
    used_port_ids = set()
    for endpoint in open_endpoints:
        labels = sorted(labels_by_open_endpoint.get(endpoint["id"], []), key=lambda item: item["bbox"][0])
        if not labels:
            retained_open_endpoints.append(endpoint)
            continue
        label = " ".join(item["text"] for item in labels)
        base_id = f"port-{_normalise_label(label)}"
        port_id = base_id
        suffix = 2
        while port_id in used_port_ids:
            port_id = f"{base_id}_{suffix}"
            suffix += 1
        used_port_ids.add(port_id)
        direction = "input" if endpoint["point"][0] < width / 2 else "output"
        ports.append(
            {
                "id": port_id,
                "label": label,
                "normalised_label": _normalise_label(label),
                "direction": direction,
                "point": endpoint["point"],
                "wire_id": endpoint["wire_id"],
                "ocr_label_ids": [item["id"] for item in labels],
                "evidence": {
                    "ocr_scores": [item["score"] for item in labels],
                    "ocr_distance_to_endpoint_px": [item["distance_to_endpoint_px"] for item in labels],
                },
                "status": "confirmed_by_ocr",
            }
        )
        port_by_open_endpoint[endpoint["id"]] = port_id

    for connection in connections:
        port_id = port_by_open_endpoint.get(connection["topology_node_id"])
        if port_id:
            connection["topology_node_id"] = port_id
            connection["topology_node_kind"] = "external_port"
            connection["status"] = "confirmed_by_ocr"

    circuit = {
        "schema_version": SCHEMA_VERSION,
        "image": {"path": str(image_path), "width": int(width), "height": int(height)},
        "pipeline": {
            "component_weights": str(component_weights_path) if component_weights_path else None,
            "attachment_radius_px": _number(attachment_radius_px),
            "junction_radius_px": _number(junction_radius_px),
        },
        "components": components,
        "wire_segments": wires,
        "junctions": junction_records,
        "terminals": terminals,
        "open_endpoints": retained_open_endpoints,
        "connections": connections,
        "ocr_records": ocr_records,
        "external_ports": ports,
    }
    circuit["matrices"] = _build_matrices(wires, junction_records, terminals, ports, retained_open_endpoints)
    validate_circuit(circuit, raise_on_error=True)
    return circuit


def validate_circuit(circuit, raise_on_error=False):
    """Return validation errors; optionally raise a concise schema exception."""
    errors = []
    required = {"schema_version", "image", "components", "wire_segments", "junctions", "terminals", "connections"}
    missing = sorted(required.difference(circuit))
    if missing:
        errors.append(f"Missing required fields: {', '.join(missing)}")
    image = circuit.get("image", {})
    if not isinstance(image.get("width"), int) or image["width"] <= 0:
        errors.append("image.width must be a positive integer")
    if not isinstance(image.get("height"), int) or image["height"] <= 0:
        errors.append("image.height must be a positive integer")

    def duplicate_ids(records, label):
        ids = [record.get("id") for record in records]
        if len(ids) != len(set(ids)) or any(identifier is None for identifier in ids):
            errors.append(f"{label} must have unique IDs")

    for field in ("components", "wire_segments", "junctions", "terminals", "connections", "external_ports", "open_endpoints"):
        duplicate_ids(circuit.get(field, []), field)
    wire_ids = {wire.get("id") for wire in circuit.get("wire_segments", [])}
    component_ids = {component.get("id") for component in circuit.get("components", [])}
    for junction in circuit.get("junctions", []):
        unknown = set(junction.get("incident_wire_ids", [])).difference(wire_ids)
        if unknown:
            errors.append(f"{junction['id']} references unknown wires: {sorted(unknown)}")
    for terminal in circuit.get("terminals", []):
        if terminal.get("component_id") not in component_ids:
            errors.append(f"{terminal['id']} references an unknown component")
        unknown = set(terminal.get("attached_wire_ids", [])).difference(wire_ids)
        if unknown:
            errors.append(f"{terminal['id']} references unknown wires: {sorted(unknown)}")
    if raise_on_error and errors:
        raise CircuitValidationError("; ".join(errors))
    return errors


def save_circuit(circuit, path):
    validate_circuit(circuit, raise_on_error=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(circuit, indent=2), encoding="utf-8")
    return path


def load_circuit(path):
    circuit = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_circuit(circuit, raise_on_error=True)
    return circuit


def build_topology(circuit):
    """Materialise electrical nets after connected/crossover review decisions.

    Only junctions whose status is ``confirmed_connected`` merge wires. A
    reviewed crossover is intentionally absent from the net-union operation.
    """
    validate_circuit(circuit, raise_on_error=True)
    wire_ids = [wire["id"] for wire in circuit["wire_segments"]]
    union_find = _UnionFind(wire_ids)
    for junction in circuit["junctions"]:
        if junction.get("connectivity_status") != "confirmed_connected":
            continue
        incident = [wire_id for wire_id in junction["incident_wire_ids"] if wire_id in union_find.parent]
        for wire_id in incident[1:]:
            union_find.union(incident[0], wire_id)

    groups = defaultdict(list)
    for wire_id in wire_ids:
        groups[union_find.find(wire_id)].append(wire_id)
    ordered_groups = sorted((sorted(group) for group in groups.values()), key=lambda group: group[0])
    net_by_wire = {wire_id: f"net-{index:03d}" for index, group in enumerate(ordered_groups, start=1) for wire_id in group}

    terminals_by_wire = defaultdict(list)
    for terminal in circuit.get("terminals", []):
        for wire_id in terminal["attached_wire_ids"]:
            terminals_by_wire[wire_id].append(terminal["id"])
    ports_by_wire = defaultdict(list)
    for port in circuit.get("external_ports", []):
        ports_by_wire[port["wire_id"]].append(port["id"])
    junctions_by_wire = defaultdict(list)
    for junction in circuit.get("junctions", []):
        if junction.get("connectivity_status") == "confirmed_connected":
            for wire_id in junction["incident_wire_ids"]:
                junctions_by_wire[wire_id].append(junction["id"])

    nets = []
    for index, group in enumerate(ordered_groups, start=1):
        terminals = sorted({terminal for wire_id in group for terminal in terminals_by_wire[wire_id]})
        ports = sorted({port for wire_id in group for port in ports_by_wire[wire_id]})
        junctions = sorted({junction for wire_id in group for junction in junctions_by_wire[wire_id]})
        nets.append(
            {
                "id": f"net-{index:03d}",
                "external_port_ids": ports,
                "terminal_ids": terminals,
                "junction_ids": junctions,
                "wire_ids": group,
            }
        )

    terminal_to_net = {
        terminal_id: net_by_wire[wire_id]
        for wire_id, terminal_ids in terminals_by_wire.items()
        for terminal_id in terminal_ids
    }
    port_to_net = {
        port_id: net_by_wire[wire_id]
        for wire_id, port_ids in ports_by_wire.items()
        for port_id in port_ids
    }
    terminals_by_component = defaultdict(list)
    for terminal in circuit.get("terminals", []):
        terminals_by_component[terminal["component_id"]].append(terminal["id"])
    component_net_incidence = []
    for component in circuit.get("components", []):
        terminal_nets = [
            {"terminal_id": terminal_id, "net_id": terminal_to_net[terminal_id]}
            for terminal_id in sorted(terminals_by_component[component["id"]])
            if terminal_id in terminal_to_net
        ]
        component_net_incidence.append(
            {"component_id": component["id"], "kind": component["kind"], "terminal_nets": terminal_nets}
        )
    return {
        "schema_version": TOPOLOGY_SCHEMA_VERSION,
        "source_schema_version": circuit["schema_version"],
        "nets": nets,
        "external_port_nets": port_to_net,
        "component_net_incidence": component_net_incidence,
    }


def topology_to_mermaid(circuit, topology, direction="LR"):
    """Create a deterministic Mermaid graph from canonical JSON topology."""
    component_by_id = {component["id"]: component for component in circuit.get("components", [])}
    terminal_component = {
        terminal["id"]: terminal["component_id"] for terminal in circuit.get("terminals", [])
    }
    port_by_id = {port["id"]: port for port in circuit.get("external_ports", [])}
    lines = [f"graph {direction}"]
    component_node_ids = {}
    for index, component in enumerate(sorted(circuit.get("components", []), key=lambda item: item["id"]), start=1):
        node_id = f"c{index}"
        component_node_ids[component["id"]] = node_id
        label = str(component["kind"]).replace('"', "'")
        lines.append(f'    {node_id}["{label}"]')
    port_node_ids = {}
    for index, port in enumerate(sorted(circuit.get("external_ports", []), key=lambda item: item["id"]), start=1):
        node_id = f"p{index}"
        port_node_ids[port["id"]] = node_id
        label = str(port["label"]).replace('"', "'")
        lines.append(f'    {node_id}(("{label}"))')
    for net in topology.get("nets", []):
        node_ids = [component_node_ids[terminal_component[terminal_id]] for terminal_id in net["terminal_ids"] if terminal_id in terminal_component]
        node_ids += [port_node_ids[port_id] for port_id in net["external_port_ids"] if port_id in port_node_ids]
        node_ids = list(dict.fromkeys(node_ids))
        if len(node_ids) < 2:
            continue
        anchor = node_ids[0]
        for node_id in node_ids[1:]:
            lines.append(f'    {anchor} ---|"{net["id"]}"| {node_id}')
    return "\n".join(lines) + "\n"


def topology_to_netlist(circuit, topology):
    """Create a serialisable netlist derived solely from canonical topology."""
    port_by_id = {port["id"]: port for port in circuit.get("external_ports", [])}
    components = []
    for item in topology.get("component_net_incidence", []):
        components.append(
            {
                "component_id": item["component_id"],
                "kind": item["kind"],
                "terminal_nets": item["terminal_nets"],
            }
        )
    nets = []
    for net in topology.get("nets", []):
        nets.append(
            {
                "net_id": net["id"],
                "ports": [port_by_id[port_id].get("label", port_id) for port_id in net["external_port_ids"] if port_id in port_by_id],
                "terminal_ids": net["terminal_ids"],
                "junction_ids": net["junction_ids"],
            }
        )
    return {
        "schema_version": "circuit-netlist/v1",
        "source_schema_version": circuit["schema_version"],
        "nets": nets,
        "components": components,
    }


def format_netlist(netlist):
    """Render the JSON netlist as concise, deterministic console text."""
    lines = ["NETLIST (derived from corrected canonical JSON)"]
    for net in netlist.get("nets", []):
        endpoints = list(net.get("ports", [])) + list(net.get("terminal_ids", []))
        lines.append(f"{net['net_id']}: " + (", ".join(endpoints) if endpoints else "(no attached ports or terminals)"))
    lines.append("COMPONENT TERMINALS")
    for component in netlist.get("components", []):
        terminal_nets = ", ".join(f"{item['terminal_id']}->{item['net_id']}" for item in component["terminal_nets"])
        lines.append(f"{component['component_id']} ({component['kind']}): {terminal_nets or '(unattached)'}")
    return "\n".join(lines) + "\n"


def render_reconstructed_circuit(circuit, output_path, topology=None):
    """Render canonical JSON as a white-canvas circuit reconstruction image."""
    validate_circuit(circuit, raise_on_error=True)
    width, height = circuit["image"]["width"], circuit["image"]["height"]
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for wire in circuit.get("wire_segments", []):
        start, end = (tuple(map(int, point)) for point in wire["points"])
        cv2.line(canvas, start, end, (90, 90, 90), 2, cv2.LINE_AA)

    connected = {junction["id"] for junction in circuit.get("junctions", []) if junction.get("connectivity_status") == "confirmed_connected"}
    crossover = {
        junction["id"]
        for junction in circuit.get("junctions", [])
        if junction.get("connectivity_status") in {"confirmed_crossover", "assumed_crossover"}
    }
    for junction in circuit.get("junctions", []):
        point = tuple(map(int, junction["point"]))
        if junction["id"] in connected:
            color = (0, 180, 255)
        elif junction["id"] in crossover:
            color = (0, 140, 255)
        else:
            color = (0, 0, 255)
        cv2.circle(canvas, point, 5, color, -1, cv2.LINE_AA)
        cv2.putText(canvas, junction["id"], (point[0] + 5, point[1] - 6), font, 0.32, color, 1, cv2.LINE_AA)

    for component in circuit.get("components", []):
        x1, y1, x2, y2 = (int(round(value)) for value in component["bbox"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"{component['id']}:{component['kind']}", (x1, max(14, y1 - 7)), font, 0.38, (255, 0, 255), 1, cv2.LINE_AA)
    for terminal in circuit.get("terminals", []):
        cv2.circle(canvas, tuple(map(int, terminal["point"])), 4, (255, 255, 0), -1, cv2.LINE_AA)
    for port in circuit.get("external_ports", []):
        point = tuple(map(int, port["point"]))
        cv2.circle(canvas, point, 5, (0, 150, 0), -1, cv2.LINE_AA)
        offset = -80 if port.get("direction") == "input" else 8
        cv2.putText(canvas, port["label"], (point[0] + offset, point[1] - 10), font, 0.5, (0, 130, 0), 2, cv2.LINE_AA)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise OSError(f"Could not write reconstruction image: {output_path}")
    return canvas
