import sys
import os
import torch

# Resolve project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

os.makedirs(MODELS_DIR, exist_ok=True)
sys.path.append(SRC_PATH)

from model_loader import load_model

model = load_model()
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

onnx_path = os.path.join(MODELS_DIR, "mobilenetv3.onnx")

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=["input"],
    output_names=["output"],
    opset_version=18,     # 👈 KEY FIX (most stable for OpenVINO)
    do_constant_folding=True
)

print(f"✅ ONNX export successful → {onnx_path}")
