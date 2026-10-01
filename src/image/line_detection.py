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
        if len(img.shape) == 2:
            gray = img
        else:
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
        
        
        lines = self.lsd.detect(skeleton)[0]
        if lines is None:
            return np.empty((0, 4), dtype=np.float32)
        return lines[:, 0, :].astype(np.float32)
        
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

        lines_was_none = lines is None
        if lines_was_none:
            lines = self.predict(img)

        canvas = img.copy()
        if len(lines) > 0:
            self.lsd.drawSegments(canvas, lines)
        if lines_was_none:
            return canvas, lines
        return canvas


def segment_geometry(segment: np.ndarray):
    start, end = segment[:2], segment[2:]
    vector = end - start
    length = np.linalg.norm(vector)
    if length == 0:
        return None
    unit = vector / length
    if unit[0] < 0 or (abs(unit[0]) < 1e-9 and unit[1] < 0):
        unit = -unit
    return unit, np.array([-unit[1], unit[0]]), (start + end) / 2


def merge_segments(
    segments,
    angle_tolerance=5,
    distance_tolerance=7,
    gap_tolerance=14,
):
    """Merge nearby segments with similar direction."""
    segments = np.asarray(segments, dtype=np.float32).reshape(-1, 4)
    pending = [segment.astype(float) for segment in segments if segment_geometry(segment)]
    merged = []

    while pending:
        current = pending.pop(0)
        changed = True
        while changed:
            changed = False
            unit, normal, midpoint = segment_geometry(current)
            current_projection = current.reshape(2, 2) @ unit
            current_low, current_high = current_projection.min(), current_projection.max()

            for index, candidate in enumerate(pending):
                candidate_unit, _, candidate_midpoint = segment_geometry(candidate)
                angle = np.degrees(np.arccos(np.clip(abs(unit @ candidate_unit), 0, 1)))
                offset = abs((candidate_midpoint - midpoint) @ normal)
                candidate_projection = candidate.reshape(2, 2) @ unit
                candidate_low, candidate_high = candidate_projection.min(), candidate_projection.max()
                gap = max(current_low, candidate_low) - min(current_high, candidate_high)

                if angle <= angle_tolerance and offset <= distance_tolerance and gap <= gap_tolerance:
                    low, high = min(current_low, candidate_low), max(current_high, candidate_high)
                    center_offset = ((midpoint @ normal) + (candidate_midpoint @ normal)) / 2
                    current = np.array(
                        [unit * low + normal * center_offset, unit * high + normal * center_offset]
                    ).reshape(4)
                    pending.pop(index)
                    changed = True
                    break
        merged.append(current)

    return np.asarray(merged, dtype=np.float32).reshape(-1, 4)
