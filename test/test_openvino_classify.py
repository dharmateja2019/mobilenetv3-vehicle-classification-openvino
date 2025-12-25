import argparse
import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
from src.classify.openvino_infer import classify_openvino

if __name__ == "__main__":
    parser = argparse.ArgumentParser("OpenVINO Vehicle Classifier Test")
    parser.add_argument("--image", required=True, help="Path to cropped vehicle image")
    args = parser.parse_args()

    result = classify_openvino(args.image)
    print(result)
