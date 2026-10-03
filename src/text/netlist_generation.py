import math

import cv2


class segments:
    def __init__(self, name="", endpoints=None,components=None):
        self.name=name
        self.endpoints=endpoints
        self.components=components



class component:
    def __init__(self, name="", coor=None, near_text=None, connected=None):
        self.name = name
        self.coor = coor
        self.near_text = near_text 
        self.connected = connected  #segments it is connected to

class netlist:
    def __init__(self,ocr_data,components_result,lines,junctions):
        self.components=self.init_components(components_result)
        self.lines=self.init_segments(lines)
        self.lines=self.merge_via_junctions(junctions)
        self.attach_text(ocr_data)

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

        if junctions is not None and len(junctions) > 0:
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


if __name__ == "__main__":
    img = cv2.imread("output/text_removed.png")


