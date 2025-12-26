import cv2
import numpy as np

COLOR_RANGES = {
    "White":  ((0, 0, 180), (180, 50, 255)),
    "Gray":   ((0, 0, 80),  (180, 40, 200)),
    "Red":    ((0, 80, 60), (10, 255, 255)),
    "Yellow": ((20, 80, 60), (35, 255, 255)),
    "Green":  ((40, 80, 60), (80, 255, 255)),
    "Blue":   ((90, 80, 60), (130, 255, 255)),
    "Black":  ((0, 0, 0),   (180, 255, 60)),
}

def detect_color(bgr_img):
    if bgr_img is None or bgr_img.size == 0:
        return "Unknown"

    h, w = bgr_img.shape[:2]

    # ---- CENTRAL REGION ONLY ----
    cx1, cy1 = int(w * 0.25), int(h * 0.25)
    cx2, cy2 = int(w * 0.75), int(h * 0.75)
    body = bgr_img[cy1:cy2, cx1:cx2]

    hsv = cv2.cvtColor(body, cv2.COLOR_BGR2HSV)

    # ---- REMOVE DARK PIXELS (shadows, tyres) ----
    v = hsv[:, :, 2]
    valid_mask = v > 50

    scores = {}
    for color, (low, high) in COLOR_RANGES.items():
        mask = cv2.inRange(
            hsv,
            np.array(low, np.uint8),
            np.array(high, np.uint8),
        )
        mask = cv2.bitwise_and(
            mask, mask,
            mask=(valid_mask.astype(np.uint8) * 255)
        )
        scores[color] = cv2.countNonZero(mask)

    # ---- Penalize black dominance ----
    scores["Black"] *= 0.4

    best = max(scores, key=scores.get)

    total_pixels = body.shape[0] * body.shape[1]
    if scores[best] / total_pixels < 0.05:
        return "Unknown"

    return best
