import cv2
import numpy as np
from skimage.morphology import skeletonize


class line_detection_hough:
    def __init__(
        self,
        threshold: int = 50,
        min_line_length: int = 50,
        max_line_gap: int = 30,
    ):
        self.threshold = threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.fld = cv2.ximgproc.createFastLineDetector()

    def predict(self, img: cv2.typing.MatLike):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Otsu threshold
        _, binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )


        # If the lines are black, invert so the foreground is white.
        #
        # Choose the polarity based on which side occupies less area.
        white_pixels = cv2.countNonZero(binary)
        black_pixels = binary.size - white_pixels
        
        if white_pixels > black_pixels:
            binary = cv2.bitwise_not(binary)
        
        cv2.imshow("threshold",binary)
        cv2.waitKey()
        # Skeletonize the filled regions
        skeleton = skeletonize(binary > 0)
        skeleton = (skeleton * 255).astype(np.uint8)
        
        cv2.imshow("skeleton",skeleton)
        cv2.waitKey()
        
        return self.fld.detect(skeleton).astype(int)
        
        # return cv2.HoughLinesP(
        #     skeleton,
        #     rho=1,
        #     theta=np.pi / 180,
        #     threshold=self.threshold,
        #     minLineLength=self.min_line_length,
        #     maxLineGap=self.max_line_gap,
        # )


    def draw_lines(
        self,
        img: cv2.typing.MatLike,
        lines=None,
    ) -> cv2.typing.MatLike:

        if lines is None:
            lines = self.predict(img)
            lines_was_none=True

        output = np.zeros_like(img)
        print(lines)
        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0]:
                cv2.line(
                    output,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 255),
                    2,
                )
        
        if lines_was_none:
            return output,lines

        return output
