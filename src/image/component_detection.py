import cv2
import numpy as np
from pathlib import Path
import shutil
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

        temporary_dir = Path("output") / "components temp"
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        temporary_dir.mkdir(parents=True, exist_ok=True)
        preserved_classes = {"crossover", "terminal", "terminals"}
        removed_index = 0

        for box in boxes:
            class_id = int(box.cls[0])
            class_name = str(result.names[class_id]).strip().lower()
            if class_name in preserved_classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1 = max(0, min(x1, img.shape[1]))
            y1 = max(0, min(y1, img.shape[0]))
            x2 = max(x1, min(x2, img.shape[1]))
            y2 = max(y1, min(y2, img.shape[0]))
            component = img[y1:y2, x1:x2].copy()
            component_path = temporary_dir / f"{removed_index:04d}_{class_name}.png"
            cv2.imwrite(str(component_path), component)
            img[y1:y2, x1:x2]=background_color
            removed_index += 1
        
        return img,result






if __name__ == "__main__":
    img = cv2.imread("test/large.jpg")

    model=yolo_component_detection("best.pt")
    img=model.draw_yolo(img)

    cv2.imwrite("YOLO Detection.png", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
