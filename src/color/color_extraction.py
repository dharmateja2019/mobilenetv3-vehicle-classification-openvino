import cv2
import numpy as np

COLOR_RANGES = {
    "White":  ((0, 0, 180), (180, 50, 255)),
    "Black":  ((0, 0, 0),   (180, 255, 60)),
    "Gray":   ((0, 0, 60),  (180, 40, 180)),
    "Red":    ((0, 80, 60), (10, 255, 255)),
    "Blue":   ((90, 80, 60), (130, 255, 255)),
    "Green":  ((40, 80, 60), (80, 255, 255)),
    "Yellow": ((20, 80, 60), (35, 255, 255)),
}

def detect_color(bgr_img):
    if bgr_img is None or bgr_img.size == 0:
        return "Unknown"

    h, w = bgr_img.shape[:2]

    # 🔥 CENTRAL REGION ONLY (THIS IS KEY)
    cx1, cy1 = int(w * 0.25), int(h * 0.25)
    cx2, cy2 = int(w * 0.75), int(h * 0.75)
    body = bgr_img[cy1:cy2, cx1:cx2]

    hsv = cv2.cvtColor(body, cv2.COLOR_BGR2HSV)

    scores = {}
    for color, (low, high) in COLOR_RANGES.items():
        mask = cv2.inRange(
            hsv,
            np.array(low, np.uint8),
            np.array(high, np.uint8)
        )
        scores[color] = cv2.countNonZero(mask)

    best = max(scores, key=scores.get)

    # 🔒 CONFIDENCE GATE
    total_pixels = body.shape[0] * body.shape[1]
    if scores[best] / total_pixels < 0.05:
        return "Unknown"

    return best
