import cv2
import numpy as np

# HSV color ranges (tuned for vehicles)
COLOR_RANGES = {
    "White":  ((0, 0, 200), (180, 40, 255)),
    "Black":  ((0, 0, 0),   (180, 255, 50)),
    "Gray":   ((0, 0, 50),  (180, 40, 200)),
    "Red":    ((0, 70, 50), (10, 255, 255)),
    "Blue":   ((100, 70, 50), (130, 255, 255)),
    "Green":  ((40, 70, 50),  (80, 255, 255)),
    "Yellow": ((20, 70, 50),  (35, 255, 255)),
}


def detect_color(image_bgr):
    """
    Detect dominant vehicle color using HSV histogram voting
    """
    if image_bgr is None or image_bgr.size == 0:
        return "Unknown"

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    color_pixels = {}

    for color, (lower, upper) in COLOR_RANGES.items():
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        color_pixels[color] = cv2.countNonZero(mask)

    # Return color with max pixels
    detected_color = max(color_pixels, key=color_pixels.get)

    # Optional confidence gate
    if color_pixels[detected_color] < 500:
        return "Unknown"

    return detected_color
