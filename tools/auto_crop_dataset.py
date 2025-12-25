import os
import cv2
from ultralytics import YOLO

# ----------------------------
# CONFIG
# ----------------------------
RAW_DIR = "data/raw_images"
OUT_2W = "data/classify/2W"
OUT_4W = "data/classify/4W"

CONF_THRESH = 0.4

# YOLO COCO class IDs
TWO_WHEELERS = {3}                  # motorcycle
FOUR_WHEELERS = {2, 5, 7}           # car, bus, truck

os.makedirs(OUT_2W, exist_ok=True)
os.makedirs(OUT_4W, exist_ok=True)

# ----------------------------
# LOAD YOLO (cached automatically)
# ----------------------------
model = YOLO("yolov8n.pt")

# ----------------------------
# PROCESS IMAGES
# ----------------------------
img_id = 0

for img_name in os.listdir(RAW_DIR):
    img_path = os.path.join(RAW_DIR, img_name)
    img = cv2.imread(img_path)

    if img is None:
        continue

    results = model(img)[0]

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if conf < CONF_THRESH:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        if cls in TWO_WHEELERS:
            out_path = f"{OUT_2W}/2w_{img_id}.jpg"
        elif cls in FOUR_WHEELERS:
            out_path = f"{OUT_4W}/4w_{img_id}.jpg"
        else:
            continue

        cv2.imwrite(out_path, crop)
        img_id += 1

print("✅ Dataset auto-cropping completed")
print("2W images:", len(os.listdir(OUT_2W)))
print("4W images:", len(os.listdir(OUT_4W)))
