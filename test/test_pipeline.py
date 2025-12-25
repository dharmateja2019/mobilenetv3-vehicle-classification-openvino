import argparse
import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
from src.pipeline import run_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Vehicle Detection + Classification Pipeline")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--out", default="outputs/result.jpg")
    args = parser.parse_args()

    run_pipeline(args.image, args.out)
