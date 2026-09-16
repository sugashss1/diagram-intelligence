import cv2
import numpy as np
from skimage.morphology import skeletonize


class line_detection_lsd:
    def __init__(
        self,
    ):

        self.lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        # self.lsd = cv2.ximgproc.createFastLineDetector(cv2.LSD_REFINE_STD)

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
        
        # Skeletonize the filled regions
        skeleton = skeletonize(binary > 0)
        skeleton = (skeleton * 255).astype(np.uint8)
        
        
        return self.lsd.detect(skeleton)[0].astype(int)
        
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

        self.lsd.drawSegments(img,lines)
        if lines_was_none:
            return img,lines

        return img
