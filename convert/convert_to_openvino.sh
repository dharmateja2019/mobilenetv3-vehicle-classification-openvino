#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Project root: $PROJECT_ROOT"

python -m openvino.tools.mo \
  --input_model "$PROJECT_ROOT/models/mobilenetv3.onnx" \
  --output_dir "$PROJECT_ROOT/models/ir"

echo "✅ OpenVINO IR conversion successful"
