# Vehicle Detection & Classification  
YOLO + MobileNetV3 | PyTorch vs OpenVINO

This project implements a **complete end-to-end vehicle detection and classification pipeline** with optimized inference using OpenVINO.  
It supports **CLI and Streamlit GUI**, along with **PyTorch vs OpenVINO performance comparison**.

---

## 🚀 Overview

The system performs:

1. **Vehicle detection** using YOLO
2. **Vehicle classification** using MobileNetV3
   - 2-Wheeler (2W)
   - 4-Wheeler (4W)
3. **Model export & optimization**
   - PyTorch → ONNX → OpenVINO IR
4. **Inference backends**
   - PyTorch
   - OpenVINO
5. **Interfaces**
   - CLI (`cli.py`)
   - Streamlit GUI (`app.py`)

The project follows a **production-style pipeline**, separating inference logic, pipeline orchestration, tests, and UI.

---

## 🧠 Architecture

Input Image
↓
YOLO Detection (src/detect.py)
↓
Vehicle Crops (NumPy arrays)
↓
MobileNetV3 Classifier
├── PyTorch (src/classify/train.py)
└── OpenVINO (src/classify/openvino_infer.py)
↓
Annotated Output + Metrics

yaml
Copy code

---

## ✅ Project Status

### STEP-1: Vehicle Detection — ✅ DONE
- YOLO-based detection
- Bounding box extraction
- Auto-cropping vehicles
- CLI + unit tests

### STEP-2: Vehicle Classification — ✅ DONE
- Dataset created from YOLO crops
- MobileNetV3 trained using PyTorch
- CLI-based classification validation

Dataset summary:
- 2W images: 15
- 4W images: 36

> Dataset is intentionally small to validate the pipeline, not for production accuracy.

---

### STEP-3: Model Export — ✅ DONE
- PyTorch → ONNX (`export_onnx.py`)

### STEP-4: OpenVINO Conversion — ✅ DONE
- ONNX → OpenVINO IR (`openvino_convert.sh`)
- FP16 optimized model

### STEP-5: Inference & Comparison — ✅ DONE
- PyTorch inference
- OpenVINO inference
- Latency & FPS comparison
- Prediction parity verified

### STEP-6: CLI & GUI — ✅ DONE
- Unified pipeline (`src/pipeline.py`)
- CLI with backend/device selection
- Streamlit GUI with side-by-side comparison
- Clean tables and annotated images

---

## 🧪 CLI Usage

### Single Backend
```bash
python3 cli.py --image path/to/image.jpg --backend openvino --device CPU
Compare PyTorch vs OpenVINO
bash
Copy code
python3 cli.py --image path/to/image.jpg --backend compare --device CPU
Outputs:

outputs/result.jpg

outputs/result_pytorch.jpg

outputs/result_openvino.jpg

🖥️ Streamlit GUI
Run:

bash
Copy code
streamlit run app.py
Features:

Backend selection (PyTorch / OpenVINO / Compare)

Device selection

Vehicle table

Annotated images

Performance comparison

⚡ Performance Example
Backend	Latency (ms)	FPS
PyTorch	~435	~2.3
OpenVINO	~244	~4.1
Speedup	~1.8×	—

🧩 Tech Stack
Python 3.11.9
YOLO (Ultralytics)
PyTorch
MobileNetV3
ONNX
onnxscript
OpenVINO Runtime
OpenCV, NumPy
Streamlit

🎯 Key Engineering Decisions
Classifiers accept NumPy arrays or file paths

No runtime dependency on test modules

Single pipeline shared by CLI & GUI

Clean separation of logic and presentation

Optimized inference without accuracy loss

⚠️ Memory & Performance Notes (Important)

This project combines real-time computer vision models with an optional Vision-Language Model (VLM).
These components have very different resource requirements.

Please read this section carefully before running the application.

🧠 Model Categories Used in This Project
1️⃣ Real-time Inference Models (Lightweight)

Used for:

Vehicle detection

Vehicle classification

Color extraction

Technologies

YOLO (detection)

OpenVINO (MobileNetV3 – FP16/FP32)

Characteristics

Optimized for CPU

Static computation graph

Low memory footprint

Memory Usage

FP32 OpenVINO: ~300–400 MB

FP16 OpenVINO: ~200–300 MB

✅ Runs comfortably on 2–4 GB RAM
✅ Stable on low-resource VMs

2️⃣ Vision-Language Model (VLM) – Optional

Used for:

Natural language Q/A on images
(e.g., “How many vehicles are visible?”)

Technology

BLIP VQA (Vision-Language Model)

Characteristics

Large multimodal foundation model

Dynamic attention mechanisms

PyTorch-based

Memory-intensive by design

Memory Usage

Model weights (model.safetensors): ~1.5–2.0 GB

Runtime buffers, attention caches, tokenizer: ~2–3 GB

Total VLM footprint: ~4–6 GB

⚠️ Requires 12–16 GB RAM for stable execution

🚨 Why VLM May Crash or Restart the VM

If VLM is enabled on a low-memory VM (≤ 8 GB RAM), you may experience:

VM freeze

Sudden restart

OOM (Out-Of-Memory) termination during model load

This happens because:

VLM loads large model weights into RAM

PyTorch allocates additional runtime buffers

Combined memory usage exceeds VM capacity

Linux OOM killer terminates the process

❗ This is expected behavior, not a bug.

✅ Recommended Execution Strategy
✔️ On Low-RAM VMs (≤ 8 GB)

Run without VLM enabled

Use:

OpenVINO (FP16 recommended)

YOLO detection

Streamlit UI (CV pipeline only)

streamlit run app.py
# Do NOT enter a VLM question

✔️ On Local Machines / High-RAM Systems (Recommended)

Run with VLM enabled

Ideal environments:

macOS (especially Apple Silicon)

Linux systems with ≥ 16 GB RAM

streamlit run app.py
# Enter a question in the VLM Q/A box

🧩 Design Decision: Why VLM Is Optional

The VLM is intentionally:

Lazy-loaded (only loads when a question is asked)

Excluded from performance metrics

Not required for core functionality

This ensures:

Real-time inference remains fast

Low-resource environments remain stable

Advanced reasoning is available when resources permit

📊 Summary Table
Component	RAM Requirement
OpenVINO FP16 inference	2–4 GB
YOLO + OpenVINO pipeline	4–6 GB
VLM (BLIP Q/A)	12–16 GB
VLM + Streamlit + CV	16 GB (recommended)

### Why YOLO Detection Is Not Converted to OpenVINO

In this project, only the classification model is converted to OpenVINO, while the detection model (YOLO) remains in PyTorch.

This is an intentional design decision:

- The classification model (MobileNetV3) is lightweight and benefits significantly from OpenVINO CPU optimizations (FP16/INT8).
- YOLO detection models are larger, involve dynamic shapes and complex post-processing, and offer limited CPU performance gains from OpenVINO without additional optimization steps.
- Keeping YOLO in PyTorch simplifies development, debugging, and future model updates.

This hybrid approach balances performance, maintainability, and development efficiency.

