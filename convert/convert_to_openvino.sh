#!/bin/bash
set -e

PROJECT_ROOT=$(dirname "$(dirname "$(realpath "$0")")")

python -m openvino.tools.mo \
  --input_model "$PROJECT_ROOT/models/mobilenetv3.onnx" \
  --output_dir "$PROJECT_ROOT/models/ir"

echo "OpenVINO IR conversion successful"
