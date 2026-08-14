from image import preprocess
from cv2 import typing
import cv2
from pytesseract import image_to_data
import pytesseract
import numpy as np

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



img=preprocess.load("original.png")
j=text_extract(img)
cv2.imwrite("out.png",remove_text(img,j))
    
