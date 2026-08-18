import cv2
import os
import numpy as np

def read_grayscale(path:str) -> (cv2.typing.MatLike | None):
    if not os.path.exists(path):
        print("can not find"+ path)
    
    return cv2.imread(path)

def load(path:str) ->(cv2.typing.MatLike):
    img=read_grayscale(path)
    if(img is None):
        print("can not parse")
        return np.empty((0, 0), dtype=np.uint8)
    # orther preprocess things
    # _,img=cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    
    return img


