import numpy as np
import os

# -----------------------------
# ImageNet labels (for MobileNetV3)
# -----------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGENET_LABELS_FILE = os.path.join(PROJECT_ROOT, "models", "imagenet_classes.txt")

with open(IMAGENET_LABELS_FILE, "r") as f:
    IMAGENET_LABELS = [line.strip() for line in f]

# -----------------------------
# Vehicle recognition labels (future use)
# -----------------------------
COLORS = ["White", "Gray", "Yellow", "Red", "Green", "Blue", "Black"]
TYPES = ["Car", "Bus", "Truck", "Van"]


def postprocess(*outputs):
    """
    Smart postprocess supporting:
    1) ImageNet classification        → postprocess(output)
    2) Vehicle recognition (future)   → postprocess(type_output, color_output)
    """

    # --------------------------------------------------
    # Case 1: Single-head ImageNet classification
    # --------------------------------------------------
    if len(outputs) == 1:
        output = outputs[0]

        # Softmax
        exp = np.exp(output - np.max(output))
        probs = exp / exp.sum()

        idx = int(np.argmax(probs))
        label = IMAGENET_LABELS[idx]
        confidence = float(probs[idx])

        print("Predicted ImageNet class index:", idx)

        return label, confidence

    # --------------------------------------------------
    # Case 2: Multi-head vehicle recognition (later)
    # --------------------------------------------------
    elif len(outputs) == 2:
        type_output, color_output = outputs

        vehicle_type = TYPES[int(np.argmax(type_output))]
        vehicle_color = COLORS[int(np.argmax(color_output))]

        return vehicle_type, vehicle_color

    else:
        raise ValueError("Invalid number of outputs passed to postprocess()")
