from ultralytics import YOLO

# COCO vehicle class IDs
# 2: car, 3: motorcycle, 5: bus, 7: truck
VEHICLE_CLASSES = {2, 3, 5, 7}

class VehicleDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect(self, image):
        """
        Args:
            image: BGR image (numpy array)

        Returns:
            List of dicts:
            [
              {
                "bbox": (x1, y1, x2, y2),
                "confidence": 0.85
              }
            ]
        """
        results = self.model(image, conf=0.4, verbose=False)[0]
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls)
            if cls_id not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf)

            detections.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": conf
            })

        return detections
