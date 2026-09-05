import cv2
import numpy as np
from ultralytics import YOLO


class yolo_component_detection:
    def __init__(self,path:str):
        self.model = YOLO(path)
    
    def predict(self,img:cv2.typing.MatLike):
        return self.model(img)

    def draw_yolo(self,img: cv2.typing.MatLike) -> cv2.typing.MatLike:
        """
        Draw YOLO detections on an image.
        """
        result=self.predict(img)[0]
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            confidence = float(box.conf[0])

            class_id = int(box.cls[0])

            class_name = result.names[class_id]

            label = f"{class_name} {confidence:.2f}"

            cv2.putText(
                img,
                label,
                (x1, max(y1 - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
        return img
    
    def remove_components(self,img: cv2.typing.MatLike):

        result=self.predict(img)[0]
        boxes = result.boxes
        background_color = np.median(img.reshape(-1, 3), axis=0).astype(np.uint8)
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            img[y1:y2, x1:x2]=background_color
        
        return img,result






if __name__ == "__main__":
    img = cv2.imread("test/large.jpg")

    model=yolo_component_detection("best.pt")
    img=model.draw_yolo(img)

    cv2.imwrite("YOLO Detection.png", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
