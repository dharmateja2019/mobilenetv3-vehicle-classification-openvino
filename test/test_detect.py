import argparse
import cv2
import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
from src.detect import VehicleDetector

def main():
    parser = argparse.ArgumentParser(description="Test vehicle detection")
    parser.add_argument(
        "--image",
        required=True,
        help="Path to input image"
    )
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise ValueError(f"Failed to load image: {args.image}")

    detector = VehicleDetector()
    detections = detector.detect(image)

    print("Detections:")
    for det in detections:
        print(det)

if __name__ == "__main__":
    main()
