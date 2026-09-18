from pathlib import Path

import cv2
import pandas as pd

from image import component_detection, line_detection, preprocess, text_extraction
from image.junction_detection import classify_junctions


input_path = Path("test/full-adder-circuit.png")
output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)

image = preprocess.load(str(input_path))
if image is None or image.size == 0:
	raise FileNotFoundError(f"Could not load image: {input_path}")

paddle_ocr = text_extraction.ocr_paddle()
image, texts = paddle_ocr.remove_text(image.copy())
cv2.imwrite(str(output_dir / "text_removed.png"), image)

yolo = component_detection.yolo_component_detection("best.pt")
image, components = yolo.remove_components(image.copy())
cv2.imwrite(str(output_dir / "components_removed.png"), image)

line_detector = line_detection.line_detection_lsd()
raw_lines = line_detector.predict(image)
cv2.imwrite(str(output_dir / "lines_raw.png"), line_detector.draw_lines(image, raw_lines))

merged_lines = line_detection.merge_segments(raw_lines)
merged_image = image.copy()
for x0, y0, x1, y1 in merged_lines:
	cv2.line(
		merged_image,
		(round(x0), round(y0)),
		(round(x1), round(y1)),
		(0, 0, 255),
		2,
		cv2.LINE_AA,
	)
cv2.imwrite(str(output_dir / "lines_merged.png"), merged_image)

junctions = classify_junctions(merged_lines)
junction_image = merged_image.copy()
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

cv2.imwrite(str(output_dir / "junctions.png"), junction_image)
cv2.imwrite("out.png", junction_image)
pd.DataFrame(raw_lines, columns=["x0", "y0", "x1", "y1"]).to_csv(
	output_dir / "lines_raw.csv", index=False
)
pd.DataFrame(merged_lines, columns=["x0", "y0", "x1", "y1"]).to_csv(
	output_dir / "lines_merged.csv", index=False
)
junctions.to_csv(output_dir / "junctions.csv", index=False)

print(f"Input: {input_path}")
print(f"Raw lines: {len(raw_lines)}")
print(f"Merged lines: {len(merged_lines)}")
print(f"Junctions: {len(junctions)}")
