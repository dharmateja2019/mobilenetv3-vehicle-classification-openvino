import numpy as np

LABELS = {
    0: "Two-Wheeler",
    1: "Four-Wheeler"
}

def postprocess(output):
    exp = np.exp(output)
    probs = exp / exp.sum()
    idx = int(np.argmax(probs))
    return LABELS.get(idx, "Unknown"), float(probs[idx])
