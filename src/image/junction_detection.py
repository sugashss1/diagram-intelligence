from itertools import combinations

import numpy as np
import pandas as pd
import cv2


def point_segment_distance(point, segment):
    start, end = segment[:2], segment[2:]
    vector = end - start
    denominator = vector @ vector
    if denominator == 0:
        return np.linalg.norm(point - start), 0.0
    parameter = np.clip(((point - start) @ vector) / denominator, 0, 1)
    return np.linalg.norm(point - (start + parameter * vector)), parameter


def segment_intersection(first, second):
    p, r = first[:2], first[2:] - first[:2]
    q, s = second[:2], second[2:] - second[:2]
    cross = r[0] * s[1] - r[1] * s[0]
    if abs(cross) < 1e-9:
        return None

    q_minus_p = q - p
    first_parameter = (q_minus_p[0] * s[1] - q_minus_p[1] * s[0]) / cross
    second_parameter = (q_minus_p[0] * r[1] - q_minus_p[1] * r[0]) / cross
    if -0.03 <= first_parameter <= 1.03 and -0.03 <= second_parameter <= 1.03:
        return p + first_parameter * r
    return None


def cluster_points(points, radius=10):
    clusters = []
    for point in points:
        for cluster in clusters:
            if np.linalg.norm(point - cluster["center"]) <= radius:
                cluster["points"].append(point)
                cluster["center"] = np.mean(cluster["points"], axis=0)
                break
        else:
            clusters.append({"center": point.copy(), "points": [point.copy()]})
    return [cluster["center"] for cluster in clusters]


def classify_junctions(segments, radius=10, angle_tolerance=18):
    """Classify segment nodes as L, T, or + junctions."""
    segments = np.asarray(segments, dtype=np.float32).reshape(-1, 4)
    if len(segments) == 0:
        return pd.DataFrame(columns=["x", "y", "type", "degree"])

    candidates = [point for segment in segments for point in (segment[:2], segment[2:])]
    candidates += [
        intersection
        for first, second in combinations(segments, 2)
        if (intersection := segment_intersection(first, second)) is not None
    ]

    rows = []
    for node in cluster_points(candidates, radius):
        rays = []
        for segment in segments:
            distance, parameter = point_segment_distance(node, segment)
            if distance > radius:
                continue

            start, end = segment[:2], segment[2:]
            if 0.08 < parameter < 0.92:
                vectors = [start - node, end - node]
            elif parameter <= 0.92:
                vectors = [end - node]
            else:
                vectors = [start - node]

            for vector in vectors:
                if np.linalg.norm(vector) <= 2:
                    continue
                angle = np.degrees(np.arctan2(vector[1], vector[0])) % 360
                if not any(
                    abs((angle - other + 180) % 360 - 180) < angle_tolerance
                    for other in rays
                ):
                    rays.append(angle)

        degree = len(rays)
        if degree >= 4:
            junction_type = "+"
        elif degree == 3:
            junction_type = "T"
        elif degree == 2:
            separation = abs((rays[0] - rays[1] + 180) % 360 - 180)
            if not 45 <= separation <= 135:
                continue
            junction_type = "L"
        else:
            continue

        rows.append({"x": node[0], "y": node[1], "type": junction_type, "degree": degree})

    return pd.DataFrame(rows, columns=["x", "y", "type", "degree"]).drop_duplicates(
        subset=["x", "y", "type"]
    )

def draw_junction(img,lines):
    junctions = classify_junctions(lines)
    junction_image = img
    colors = {"L": (255, 0, 0), "T": (0, 165, 255), "+": (0, 255, 0)}
    for row in junctions.itertuples():
        center = (round(row.x), round(row.y))
        color = colors[row.type]
        cv2.circle(junction_image, center, 7, color, -1)
        cv2.putText(
            junction_image,
            row.type,
            (center[0] + 8, center[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
    return junction_image,junctions
