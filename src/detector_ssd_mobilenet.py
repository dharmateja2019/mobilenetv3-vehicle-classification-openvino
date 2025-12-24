import cv2
import numpy as np
import os

# COCO class IDs for vehicles
VEHICLE_CLASS_IDS = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

CONF_THRESHOLD = 0.25


class SSDMobileNetDetector:
    def __init__(self, model_path, config_path):
        self.net = cv2.dnn.readNetFromTensorflow(
            model_path, config_path
        )

    def detect(self, image_bgr):
        """
        Returns list of detections:
        [
          {
            'label': 'car',
            'bbox': (x1, y1, x2, y2),
            'confidence': 0.87
          }
        ]
        """
        h, w = image_bgr.shape[:2]

        blob = cv2.dnn.blobFromImage(
            image_bgr,
            size=(300, 300),
            swapRB=True,
            crop=False
        )

        self.net.setInput(blob)
        detections = self.net.forward()

        results = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            class_id = int(detections[0, 0, i, 1])

            if confidence < CONF_THRESHOLD:
                continue

            if class_id not in VEHICLE_CLASS_IDS:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)

            results.append({
                "label": VEHICLE_CLASS_IDS[class_id],
                "bbox": (x1, y1, x2, y2),
                "confidence": float(confidence)
            })

        return results
