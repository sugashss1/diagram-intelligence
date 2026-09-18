import cv2

from image import component_detection, line_dectection, preprocess, text_extraction

paddle_ocr=text_extraction.ocr_paddle();
yolo=component_detection.yolo_component_detection("best.pt");
lines_dect=line_dectection.line_detection_hough()

img=preprocess.load("test/large.jpg")

img,components=yolo.remove_components(img)

img,texts=paddle_ocr.remove_text(img)

img,lines=lines_dect.draw_lines(img)

cv2.imwrite("out.png",img)
