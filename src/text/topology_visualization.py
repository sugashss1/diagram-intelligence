"""Offline PNG rendering for a recovered circuit topology.

When Mermaid CLI is available it renders the saved ``.mmd`` source directly.
Otherwise a deterministic local renderer writes a Mermaid-style topology image,
so the pipeline never depends on a browser or external rendering service.
"""

from __future__ import annotations

import shutil
import subprocess
from collections import defaultdict
from pathlib import Path


def _normalise(value, low, high, default=0.5):
    if high <= low:
        return default
    return (value - low) / (high - low)


def _render_with_mermaid_cli(mermaid_path, output_path):
    """Return True only when a locally installed Mermaid CLI creates the PNG."""
    executable = shutil.which("mmdc")
    if not executable:
        return False
    try:
        completed = subprocess.run(
            [executable, "--input", str(mermaid_path), "--output", str(output_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0 and output_path.is_file()


def _render_locally(circuit, topology, output_path):
    """Render a dark Mermaid-style graph with matplotlib as a portable fallback."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyBboxPatch

    components = list(circuit.get("components", []))
    ports = list(circuit.get("external_ports", []))
    width = max(1, circuit.get("image", {}).get("width", 1))
    height = max(1, circuit.get("image", {}).get("height", 1))
    positions = {}
    for component in components:
        x1, y1, x2, y2 = component["bbox"]
        positions[component["id"]] = (0.08 + 0.84 * _normalise((x1 + x2) / 2, 0, width), 0.08 + 0.84 * (1 - _normalise((y1 + y2) / 2, 0, height)))
    for port in ports:
        x, y = port["point"]
        positions[port["id"]] = (0.04 + 0.92 * _normalise(x, 0, width), 0.05 + 0.90 * (1 - _normalise(y, 0, height)))

    terminal_component = {terminal["id"]: terminal["component_id"] for terminal in circuit.get("terminals", [])}
    net_members = {}
    for net in topology.get("nets", []):
        members = [terminal_component[terminal_id] for terminal_id in net.get("terminal_ids", []) if terminal_id in terminal_component]
        members.extend(port_id for port_id in net.get("external_port_ids", []) if port_id in positions)
        members = list(dict.fromkeys(member for member in members if member in positions))
        if members:
            x = sum(positions[member][0] for member in members) / len(members)
            y = sum(positions[member][1] for member in members) / len(members)
        else:
            x, y = 0.5, 0.5
        net_members[net["id"]] = members
        positions[net["id"]] = (x, y)

    occupied = defaultdict(int)
    for net_id, (x, y) in positions.items():
        if not net_id.startswith("net-"):
            continue
        key = (round(x, 2), round(y, 2))
        offset = occupied[key]
        occupied[key] += 1
        positions[net_id] = (x, min(0.95, max(0.05, y + 0.045 * (offset - 0.5))))

    figure, axis = plt.subplots(figsize=(14, max(7, 14 * height / width)), dpi=160)
    figure.patch.set_facecolor("#171717")
    axis.set_facecolor("#171717")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    for net_id, members in net_members.items():
        net_x, net_y = positions[net_id]
        for member in members:
            member_x, member_y = positions[member]
            axis.plot([member_x, net_x], [member_y, net_y], color="#a5a5a5", linewidth=1.2, zorder=1)

    for component in components:
        x, y = positions[component["id"]]
        patch = FancyBboxPatch((x - 0.055, y - 0.035), 0.11, 0.07, boxstyle="round,pad=0.012,rounding_size=0.02", facecolor="#06345e", edgecolor="#075389", linewidth=1.2, zorder=3)
        axis.add_patch(patch)
        axis.text(x, y, str(component["kind"]), color="#acd7ff", ha="center", va="center", fontsize=12, fontweight="bold", zorder=4)
    for port in ports:
        x, y = positions[port["id"]]
        patch = Circle((x, y), 0.018, facecolor="#06345e", edgecolor="#075389", linewidth=1.2, zorder=3)
        axis.add_patch(patch)
        axis.text(x, y, str(port.get("label", port["id"])), color="#acd7ff", ha="center", va="center", fontsize=10, fontweight="bold", zorder=4)
    for net_id in sorted(net_members):
        x, y = positions[net_id]
        patch = FancyBboxPatch((x - 0.04, y - 0.018), 0.08, 0.036, boxstyle="round,pad=0.008,rounding_size=0.018", facecolor="#071015", edgecolor="#075389", linewidth=1.0, zorder=3)
        axis.add_patch(patch)
        axis.text(x, y, net_id, color="#acd7ff", ha="center", va="center", fontsize=9, fontweight="bold", zorder=4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, facecolor=figure.get_facecolor(), bbox_inches="tight", pad_inches=0.12)
    plt.close(figure)


def render_topology_png(circuit, topology, mermaid_path, output_path):
    """Save the Mermaid topology as PNG, returning the renderer that was used."""
    mermaid_path = Path(mermaid_path)
    output_path = Path(output_path)
    if _render_with_mermaid_cli(mermaid_path, output_path):
        return "mermaid-cli"
    _render_locally(circuit, topology, output_path)
    return "matplotlib-fallback"
