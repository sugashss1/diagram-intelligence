from image import preprocess
from cv2 import typing
import cv2
from pytesseract import image_to_data
import pytesseract
import numpy as np
import os
import json
# Must be set BEFORE importing Paddle/PaddleOCR
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"

from paddleocr import PaddleOCR

class ocr_paddle:
    def __init__(self):
        self.ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            engine="paddle",
            enable_mkldnn=False,
            )
    

    def predict(self, img):
        # PaddleOCR expects H x W x C
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        result = self.ocr.predict(img)
        result[0].save_to_json("output")

        data = result[0].json

        if isinstance(data, str):
            data = json.loads(data)

        # Remove OCR detections containing "ww"
        res = data.get("res", {})
        boxes = res.get("rec_boxes", [])
        texts = res.get("rec_texts", [])

        filtered_boxes = []
        filtered_texts = []

        for box, text in zip(boxes, texts):
            #a lot of resistors gets classified as some sequence of text with ww since most words do not contain a w next to another w another we remove the entry
            if "ww" in text.lower():
                continue

            filtered_boxes.append(box)
            filtered_texts.append(text)

        res["rec_boxes"] = filtered_boxes
        res["rec_texts"] = filtered_texts

        return data


    def remove_text(self, img):
        mask = np.zeros(img.shape[:2], dtype=np.uint8)

        data = self.predict(img)

        boxes = data["res"].get("rec_boxes", [])

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            mask[y1:y2, x1:x2] = 255

        return cv2.inpaint(
            img,
            mask,
            3,
            cv2.INPAINT_TELEA
        )


def text_extract(img:typing.MatLike) -> dict:
    data=image_to_data(img,output_type=pytesseract.Output.DATAFRAME)
    
    data = data.dropna(subset=["text"])
    data = data[data["text"].str.strip() != ""]
    data = data[data["conf"] >= 30]
    #removes texts with confidence less than 30 and empty texts found

    return data.to_dict(orient='list')



def draw_ocr(img: typing.MatLike,data: dict) -> typing.MatLike:
     
    for i, text in enumerate(data["text"]):

        x: int = data["left"][i]
        y: int = data["top"][i]
        w: int = data["width"][i]
        h: int = data["height"][i]



        cv2.rectangle(
            img,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2,
        )

        # cv2.putText(
        #     img,
        #     text,
        #     (x, max(y - 5, 15)),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.5,
        #     (0, 0, 255),
        #     1,
        #     cv2.LINE_AA,
        # )

    return img

def remove_text(img: typing.MatLike,data: dict) -> typing.MatLike:
    
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    
    for i, text in enumerate(data["text"]):
        x: int = data["left"][i]
        y: int = data["top"][i]
        w: int = data["width"][i]
        h: int = data["height"][i]
        # Slightly expand the box to catch antialiased text edges
        padding = 2

        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img.shape[1], x + w + padding)
        y2 = min(img.shape[0], y + h + padding)

        mask[y1:y2, x1:x2] = 255

    # Reconstruct the masked regions from surrounding pixels
    result = cv2.inpaint(
        img,
        mask,
        inpaintRadius=5,
        flags=cv2.INPAINT_TELEA
    )

    return result



img=preprocess.load("test/large.jpg")
j=ocr_paddle()

cv2.imwrite("out.png",j.remove_text(img))


# j=text_extract(img)
# cv2.imwrite("bounding.png",draw_ocr(img,j))
# cv2.imwrite("out.png",remove_text(img,j))
    
