🚗 Vehicle Detection, Classification & Scene Understanding

YOLO + MobileNetV3 | PyTorch vs OpenVINO | VLM (Qwen2.5)

This project implements a production-style, end-to-end vehicle detection and classification pipeline with optimized inference using OpenVINO, and an optional Vision-Language Model (VLM) for natural-language scene understanding.

It supports:

CLI

Streamlit GUI

Fair PyTorch vs OpenVINO performance comparison

Optional, lightweight VLM reasoning

🚀 Overview

The system performs:

Vehicle Detection

YOLO (Ultralytics)

Vehicle Classification

MobileNetV3

Classes:

2-Wheeler (2W)

4-Wheeler (4W)

Model Export & Optimization

PyTorch → ONNX → OpenVINO IR (FP16)

Inference Backends

PyTorch

OpenVINO (CPU-optimized)

Optional Vision-Language Reasoning (VLM)

Natural-language Q&A over detection results

Powered by Qwen2.5-0.5B-Instruct

Interfaces

CLI (cli.py)

Streamlit GUI (app.py)

The project follows a clean, production-style architecture, separating:

Computer Vision (CV)

Inference optimization

Language reasoning

UI / orchestration

🧠 Architecture
Input Image
   ↓
YOLO Detection (src/detect.py)
   ↓
Vehicle Crops
   ↓
MobileNetV3 Classification
   ├── PyTorch
   └── OpenVINO (FP16)
   ↓
Annotated Output + Metrics
   ↓
(Optional)
Vision-Language Model (Qwen2.5-0.5B)
   ↓
Natural-Language Answers


Important:
VLM runs after CV inference and is not included in latency/FPS metrics.

✅ Project Status
STEP-1: Vehicle Detection — ✅ DONE

YOLO-based detection

Bounding box extraction

Auto-cropping vehicles

CLI + unit tests

STEP-2: Vehicle Classification — ✅ DONE

Dataset created from YOLO crops

MobileNetV3 trained using PyTorch

CLI-based validation

Dataset summary

2W images: 15

4W images: 36

Dataset size is intentionally small to validate the pipeline, not for production accuracy.

STEP-3: Model Export — ✅ DONE

PyTorch → ONNX (export_onnx.py)

STEP-4: OpenVINO Conversion — ✅ DONE

ONNX → OpenVINO IR (openvino_convert.sh)

FP16 optimized model

STEP-5: Inference & Comparison — ✅ DONE

PyTorch inference

OpenVINO inference

Latency & FPS comparison (CV pipeline only)

Prediction parity verified

STEP-6: CLI & GUI — ✅ DONE

Unified pipeline (src/pipeline.py)

Backend/device selection

Streamlit GUI with comparison mode

Clean tables and annotated images

STEP-7: Vision-Language Model (VLM) — ✅ DONE

Lightweight reasoning using Qwen2.5-0.5B-Instruct

Lazy-loaded

Optional (UI toggle)

RAM-safe on 8 GB systems

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

Optional VLM Q&A (toggle-controlled)

⚡ Performance Example (CV Pipeline Only)
Backend	Latency (ms)	FPS
PyTorch	~435	~2.3
OpenVINO	~244	~4.1
Speedup	~1.8×	—

VLM latency is intentionally excluded from benchmarks.

🧩 Tech Stack

Python 3.11.9

YOLO (Ultralytics)

PyTorch

MobileNetV3

ONNX + onnxscript

OpenVINO Runtime

OpenCV, NumPy

Streamlit

Qwen2.5-0.5B-Instruct (VLM)

🎯 Key Engineering Decisions

CV and language reasoning are decoupled

OpenVINO used only where it provides real CPU benefit

VLM operates on structured CV output, not raw images

Single pipeline shared by CLI & GUI

VLM excluded from latency/FPS metrics

Lazy-loading to protect low-RAM systems

⚠️ Memory & Performance Notes (Important)

This project combines:

Real-time CV models

An optional language model

These have very different resource requirements.

🧠 Model Categories Used
1️⃣ Real-time Inference Models (Lightweight)

Used for

Vehicle detection

Vehicle classification

Color extraction

Technologies

YOLO (detection)

OpenVINO (MobileNetV3 – FP16 / FP32)

Memory Usage

FP32 OpenVINO: ~300–400 MB

FP16 OpenVINO: ~200–300 MB

✅ Runs comfortably on 2–4 GB RAM
✅ Stable on low-resource VMs

2️⃣ Vision-Language Model (VLM) — Optional

Technology

Qwen2.5-0.5B-Instruct (text-only reasoning)

Purpose

Natural-language Q&A over detection results

Example:

“What are the colors of the vehicles?”

Memory Usage

Model + runtime: ~0.8–1.2 GB

✅ Safe on 8 GB RAM
✅ Much lighter than traditional VLMs (e.g., BLIP)

🚨 Why VLM Is Optional

Language models dominate latency

They do not benefit from OpenVINO

Not required for core CV functionality

VLM is:

Lazy-loaded

UI-controlled

Excluded from benchmarks

📊 Memory Requirement Summary
Component	RAM Requirement
OpenVINO FP16 inference	2–4 GB
YOLO + OpenVINO pipeline	4–6 GB
Qwen VLM reasoning	~1 GB
Full system (recommended)	8–12 GB
❓ Why YOLO Detection Is Not Converted to OpenVINO

Only classification is converted to OpenVINO.

Reasons

MobileNetV3 gains clear CPU speedup

YOLO has dynamic shapes & complex post-processing

Limited CPU gains without heavy tuning

PyTorch simplifies debugging & iteration

This hybrid approach balances:

Performance

Maintainability

Development efficiency
