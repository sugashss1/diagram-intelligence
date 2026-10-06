import math

import cv2

import colorsys
import numpy as np

def generate_colors_bgr(n, saturation=0.9, value=0.95):
    """n visually distinct colours as OpenCV BGR int tuples (0-255).
    Uses golden-ratio hue stepping so consecutive colours are far apart
    on the colour wheel, no matter how large n is."""
    golden = 0.618033988749895
    colors = []
    h = 0.0
    for _ in range(n):
        h = (h + golden) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, saturation, value)
        colors.append((int(b * 255), int(g * 255), int(r * 255)))  # RGB -> BGR
    return colors


class segments:
    def __init__(self, name="", endpoints=None,components=None):
        self.name : str=name
        self.endpoints: list=endpoints
        self.components: list=components



class component:
    def __init__(self, name="", coor=None,bounding_box=None, near_text=None, connected=None):
        self.name:str = name
        self.coor:list = coor
        self.bounding_box=bounding_box #x1y1x2y2
        self.near_text:list = near_text 
        self.connected:list = connected  #segments it is connected to

class netlist:
    def __init__(self,image,ocr_data,components_result,lines,junctions):
        self.image=image
        self.image_dim=image.shape
        self.components=self.init_components(components_result)
        self.lines=self.init_segments(lines)
        self.lines=self.merge_via_junctions(junctions)
        self.attach_text(ocr_data)
        self.attach_segments_and_components()
        print(self.to_mermaid())

    def attach_text(self, ocr_data, max_dist=None):
        """Attach each OCR text from paddleOCR's output to its nearest component."""
        if not self.components:
            return

        res = ocr_data.get("res", ocr_data)
        boxes = res.get("rec_boxes", [])
        texts = res.get("rec_texts", [])

        for box, text in zip(boxes, texts):
            x1, y1, x2, y2 = [float(v) for v in box]
            tx, ty = (x1 + x2) / 2, (y1 + y2) / 2

            nearest, best = None, float("inf")
            for comp in self.components:
                d = math.hypot(tx - comp.coor[0], ty - comp.coor[1])
                if d < best:
                    nearest, best = comp, d

            if max_dist is None or best <= max_dist:
                nearest.near_text.append(text)

    def init_components(self,result):
        components = []

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            class_id = int(box.cls[0])
            name = result.names[class_id]

            midpoint=[(x1+x2)/2,(y1+y2)/2]
            midpoint=[int(x) for x in midpoint]

            components.append(
                component(
                    name=name,
                    coor=midpoint,
                    bounding_box=box.xyxy[0],
                    near_text=[],
                    connected=[]
                )
            )

        return components
    
    def init_segments(self, lines):
        segs = []

        if lines is None:
            return segs

        for i, line in enumerate(lines):
            x1, y1, x2, y2 = [int(v) for v in line]

            segs.append(
                segments(
                    name=f"{i}",
                    endpoints=[[x1, y1], [x2, y2]],
                    components=[],  # components it is connected to
                )
            )

        return segs
    
    def merge_via_junctions(self, junctions, junction_types=("L", "T"), radius=10):
        """Group segments that touch through a junction and return one
        `segments` object per group. The name is the member indices,
        e.g. "1 2". start/end are the two farthest-apart endpoints
        of the group. only L and T junctions segments are merged."""

        n = len(self.lines)
        groups = [
            {"members": {i}, "endpoints": [list(p) for p in self.lines[i].endpoints]}
            for i in range(n)
        ]
        owner = list(range(n))  # segment index -> index of the group that holds it

        def dist(p, node):
            return math.hypot(p[0] - node[0], p[1] - node[1])

        def far_points(group, node):
            """Endpoints of the group that are away from the junction."""
            keep = [p for p in group["endpoints"] if dist(p, node) > radius]
            # a lone short segment with both ends at the junction: keep the farthest
            if not keep and len(group["members"]) == 1:
                keep = [max(group["endpoints"], key=lambda p: dist(p, node))]
            return keep

        if junctions is None or len(junctions) == 0:
            return self.lines

        for row in junctions.itertuples():
            if row.type not in junction_types or len(row.segment_indices) < 2:
                continue

            node = (row.x, row.y)
            gids = sorted({owner[i] for i in row.segment_indices})

            new_endpoints = []
            for g in gids:
                for p in far_points(groups[g], node):
                    if p not in new_endpoints:
                        new_endpoints.append(p)

            target = gids[0]
            members = set().union(*(groups[g]["members"] for g in gids))
            for g in gids[1:]:
                groups[g] = None
            groups[target] = {"members": members, "endpoints": new_endpoints}
            for m in members:
                owner[m] = target

        return [
            segments(
                name=" ".join(map(str, sorted(g["members"]))),
                endpoints=g["endpoints"],
                components=[],
            )
            for g in sorted((g for g in groups if g is not None), key=lambda g: min(g["members"]))
        ]
    @staticmethod
    def _dist_point_to_box(px, py, box):
        """Shortest distance from a point to an axis-aligned box (x1, y1, x2, y2).
        Returns 0 if the point is inside the box."""
        x1, y1, x2, y2 = [float(v) for v in box]
        # clamp the point onto the box, then measure to the clamped point
        dx = max(x1 - px, 0.0, px - x2)
        dy = max(y1 - py, 0.0, py - y2)
        return math.hypot(dx, dy)


    def attach_segments_and_components(self, radius=None):
        """Link each segment to the nearest component at each of its endpoints
        (within `radius` px). Fills seg.components and comp.connected."""
        if not self.components or not self.lines:
            return
        if radius==None:
            radius=np.linalg.norm(np.array(self.image_dim)/50)

        for seg in self.lines:
            if seg.components is None:
                seg.components = []

            for ex, ey in seg.endpoints:
                nearest, best = None, float("inf")
                for comp in self.components:
                    d = self._dist_point_to_box(ex, ey, comp.bounding_box)
                    if d < best:
                        nearest, best = comp, d

                if nearest is None or best > radius:
                    continue

                # avoid duplicates (e.g. both endpoints near the same component)
                if nearest not in seg.components:
                    seg.components.append(nearest)
                if nearest.connected is None:
                    nearest.connected = []
                if seg not in nearest.connected:
                    nearest.connected.append(seg)
    
    def draw(self, image, show_text=False,show_segments_name=False):
        """Draw segments (endpoints + index + name), components (center + name), and a
        line from each segment to every component it is connected to.
        Each segment gets its own colour. Returns the annotated image."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        colors = generate_colors_bgr(len(self.lines))

        for idx, (seg, color) in enumerate(zip(self.lines, colors)):
            pts = [tuple(map(int, p)) for p in seg.endpoints]

            # endpoints + segment enumerate number beside each circle
            for p in pts:
                cv2.circle(image, p, 4, color, -1)
                cv2.putText(image, str(idx), (p[0] + 6, p[1] - 6),
                            font, 0.45, color, 1, cv2.LINE_AA)

            # segment name at the centre of ALL endpoints
            if show_segments_name and pts:
                mid = (sum(p[0] for p in pts) // len(pts),
                    sum(p[1] for p in pts) // len(pts))
                cv2.putText(image, f"s{seg.name}", mid, font, 0.4, color, 1, cv2.LINE_AA)

            # line from segment -> connected components (same colour as the segment)
            for comp in seg.components or []:
                cx, cy = comp.coor
                ep = min(pts, key=lambda p: math.hypot(p[0] - cx, p[1] - cy))
                cv2.line(image, ep, (cx, cy), color, 1, cv2.LINE_AA)

        # components: center + name (+ nearby OCR text)
        for comp in self.components:
            cx, cy = comp.coor
            cv2.circle(image, (cx, cy), 5, (255, 0, 255), -1)
            label = comp.name
            if show_text and comp.near_text:
                label += " (" + ", ".join(comp.near_text) + ")"
            cv2.putText(image, label, (cx + 8, cy - 8), font, 0.5, (255, 0, 255), 1, cv2.LINE_AA)

        return image
    
    def sort_left_to_right(self):
        """Sort components and segments in place by x position (then y)."""
        self.components.sort(key=lambda c: (c.coor[0], c.coor[1]))
        self.lines.sort(key=lambda s: self._seg_key(s))

    @staticmethod
    def _seg_key(seg):
        xs = [p[0] for p in seg.endpoints]
        ys = [p[1] for p in seg.endpoints]
        return (sum(xs) / len(xs), sum(ys) / len(ys))  # centre of the net

    def to_mermaid(self, direction="LR", show_text=False, show_dangling=False):
        def esc(s):
            return str(s).replace('"', "'")

        comps_sorted = sorted(self.components, key=lambda c: (c.coor[0], c.coor[1]))
        segs_sorted = sorted(self.lines, key=self._seg_key)

        # ids assigned AFTER sorting: c0 is the leftmost component
        cid = {id(c): f"c{i}" for i, c in enumerate(comps_sorted)}

        out = [f"graph {direction}"]

        for c in comps_sorted:
            label = c.name
            if show_text and c.near_text:
                label += " (" + ", ".join(c.near_text) + ")"
            out.append(f'    {cid[id(c)]}["{esc(label)}"]')

        for i, seg in enumerate(segs_sorted):
            # components of this net, left to right
            comps = sorted(seg.components or [], key=lambda c: (c.coor[0], c.coor[1]))
            ids = [cid[id(c)] for c in comps]
            net_label = f"s{seg.name}"

            if len(ids) == 2:
                out.append(f'    {ids[0]} ---|"{esc(net_label)}"| {ids[1]}')
            elif len(ids) > 2:
                j = f"n{i}"
                out.append(f'    {j}(("{esc(net_label)}"))')
                for c_id in ids:
                    out.append(f"    {j} --- {c_id}")
            elif show_dangling:
                j = f"n{i}"
                out.append(f'    {j}(("{esc(net_label)}")):::dangling')
                for c_id in ids:
                    out.append(f"    {j} -.- {c_id}")

        if show_dangling:
            out.append("    classDef dangling stroke-dasharray:4 3,stroke:#999")

        return "\n".join(out)

if __name__ == "__main__":
    img = cv2.imread("output/text_removed.png")


