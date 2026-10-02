import math

import cv2


class component:
    def __init__(self, name="", coor=None, near_text=None, connected=None):
        self.name = name
        self.coor = coor
        self.near_text = near_text 
        self.connected = connected  #segments it is connected to

class netlist:
    def __init__(self,ocr_data,components_result,lines,junctions):
        self.components=self.init_components(components_result)
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



if __name__ == "__main__":
    img = cv2.imread("output/text_removed.png")


