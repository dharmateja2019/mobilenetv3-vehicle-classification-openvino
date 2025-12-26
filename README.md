Vehicle Detection & Classification

YOLO + MobileNetV3 | PyTorch vs OpenVINO

This project implements a complete end-to-end vehicle detection and classification pipeline with optimized inference using OpenVINO.
It supports both CLI and Streamlit GUI, along with PyTorch vs OpenVINO performance comparison.

🚀 Overview

The system performs:

Vehicle detection using YOLO

Vehicle classification using MobileNetV3

2-Wheeler (2W)

4-Wheeler (4W)

Model export & optimization

PyTorch → ONNX → OpenVINO IR

Inference backends

PyTorch

OpenVINO

Interfaces

CLI (cli.py)

Streamlit GUI (app.py)

The project follows a production-style pipeline, separating inference logic, pipeline orchestration, tests, and UI.

🧠 Architecture
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

✅ Project Status
STEP-1: Vehicle Detection — ✅ DONE

YOLO-based detection

Bounding box extraction

Auto-cropping vehicles

CLI + unit tests

STEP-2: Vehicle Classification — ✅ DONE

Dataset created from YOLO crops

MobileNetV3 trained using PyTorch

CLI-based classification validation

Dataset summary

2W images: 15

4W images: 36

Dataset is intentionally small to validate the pipeline, not for production accuracy.

STEP-3: Model Export — ✅ DONE

PyTorch → ONNX (export_onnx.py)

STEP-4: OpenVINO Conversion — ✅ DONE

ONNX → OpenVINO IR (openvino_convert.sh)

FP16 optimized model

STEP-5: Inference & Comparison — ✅ DONE

PyTorch inference

OpenVINO inference

Latency & FPS comparison

Prediction parity verified

STEP-6: CLI & GUI — ✅ DONE

Unified pipeline (src/pipeline.py)

CLI with backend/device selection

Streamlit GUI with side-by-side comparison

Clean tables and annotated images

🧪 CLI Usage
Single Backend
python3 cli.py --image path/to/image.jpg --backend openvino --device CPU

Compare PyTorch vs OpenVINO
python3 cli.py --image path/to/image.jpg --backend compare --device CPU


Outputs

outputs/result.jpg
outputs/result_pytorch.jpg
outputs/result_openvino.jpg

🖥️ Streamlit GUI

Run:

streamlit run app.py


Features

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

🧠 Model Categories Used
1️⃣ Real-time Inference Models (Lightweight)

Used for

Vehicle detection

Vehicle classification

Color extraction

Technologies

YOLO (detection)

OpenVINO (MobileNetV3 – FP16 / FP32)

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

Used for

Natural language Q&A on images
(e.g., “How many vehicles are visible?”)

Technology

BLIP VQA (Vision-Language Model)

Characteristics

Large multimodal foundation model

Dynamic attention mechanisms

PyTorch-based

Memory-intensive by design

Memory Usage

Model weights: ~1.5–2.0 GB

Runtime buffers & tokenizer: ~2–3 GB

Total footprint: ~4–6 GB

⚠️ Requires 12–16 GB RAM for stable execution

🚨 Why VLM May Crash or Restart the VM

On low-memory systems (≤ 8 GB RAM), enabling VLM may cause:

VM freeze

Sudden restart

OOM (Out-Of-Memory) termination

Reason

Large model weights loaded into RAM

PyTorch runtime buffer allocation

Linux OOM killer terminates the process

❗ This is expected behavior, not a bug.

✅ Recommended Execution Strategy
✔️ Low-RAM VMs (≤ 8 GB)

Run without VLM

Use:

YOLO detection

OpenVINO FP16 classification

Streamlit UI (CV pipeline only)

streamlit run app.py
# Do NOT enter a VLM question

✔️ High-RAM Systems (≥ 16 GB) — Recommended

Enable VLM

Ideal for:

macOS (Apple Silicon)

Linux workstations

streamlit run app.py
# Enter a VLM question in the UI

🧩 Design Decision: Why VLM Is Optional

Lazy-loaded (only when queried)

Excluded from performance benchmarks

Not required for core CV pipeline

This ensures:

Fast real-time inference

Stability on low-resource systems

Advanced reasoning when resources permit

📊 Memory Requirement Summary
Component	RAM Requirement
OpenVINO FP16 inference	2–4 GB
YOLO + OpenVINO pipeline	4–6 GB
VLM (BLIP Q/A)	12–16 GB
Full system (VLM + UI + CV)	16 GB (recommended)
❓ Why YOLO Detection Is Not Converted to OpenVINO

Only the classification model is converted to OpenVINO.
YOLO detection remains in PyTorch by design.

Reasons:

MobileNetV3 gains significant CPU speedup from OpenVINO

YOLO models have dynamic shapes & complex post-processing

Limited CPU gains without advanced OpenVINO tuning

Easier debugging and faster iteration in PyTorch

This hybrid approach balances:

Performance

Maintainability

Development efficiency

