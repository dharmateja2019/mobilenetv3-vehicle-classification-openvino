import torch
from torchvision import models

MODEL_PTH = "models/classify/mobilenetv3_2w4w.pth"
OUT_ONNX = "models/classify/mobilenetv3_2w4w.onnx"

model = models.mobilenet_v3_small(weights=None)
model.classifier[3] = torch.nn.Linear(
    model.classifier[3].in_features, 2
)

model.load_state_dict(torch.load(MODEL_PTH, map_location="cpu"))
model.eval()

dummy = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy,
    OUT_ONNX,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch"},
        "output": {0: "batch"}
    },
    opset_version=13
)

print("✅ ONNX exported:", OUT_ONNX)
