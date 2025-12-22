import sys
import os
import torch

# Add src/ to Python path (absolute path, not relative guess)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
sys.path.append(SRC_PATH)

from model_loader import load_model

model = load_model()
model.eval()

dummy = torch.randn(1, 3, 224, 224)

onnx_path = os.path.join(PROJECT_ROOT, "models", "mobilenetv3.onnx")

torch.onnx.export(
    model,
    dummy,
    onnx_path,
    input_names=["input"],
    output_names=["output"],
    opset_version=18,
    do_constant_folding=True
)

print(f"ONNX export successful → {onnx_path}")
