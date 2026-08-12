import cv2
import os

def read_grayscale(path:str) -> (cv2.typing.MatLike | None):
    if not os.path.exists(path):
        print("can not find"+ path)
    
    return cv2.imread(path,cv2.IMREAD_GRAYSCALE)

def load(path:str) ->(cv2.typing.MatLike | None):
    img=read_grayscale(path)
    # orther preprocess things
    
    return img


