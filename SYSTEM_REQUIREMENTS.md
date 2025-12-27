This document describes the system requirements for running the Vehicle Detection, Classification & Scene Understanding project.
The project has been tested and validated on the configuration listed below.

✅ Tested System (Baseline)
The project is confirmed to work correctly on the following system:


Operating System: Ubuntu 24.04.3 LTS
Python: 3.11
RAM: 12 GB
Storage: 150 GB HDD
CPU: Multi-core x86_64 (CPU-only inference)
GPU: ❌ Not required

This configuration supports all features, including:
YOLO detection
MobileNetV3 classification
OpenVINO inference (FP16)
CLI & Streamlit GUI
Optional Vision-Language Model (VLM)

🔹 Minimum Requirements (CV Pipeline Only)
Suitable for running vehicle detection and classification without VLM.
Hardware
CPU: 4 cores (Intel i5 / AMD Ryzen 5 or equivalent)
RAM: 4–6 GB
Disk Space: ~5 GB free
GPU: Not required

Software
OS: Ubuntu 20.04+ / macOS / Linux
Python: 3.10 or 3.11

Supported Features
YOLO detection
MobileNetV3 classification
OpenVINO inference
CLI
Streamlit UI
PyTorch vs OpenVINO comparison

🔹 Recommended Requirements (Full Project)
Recommended for a smooth and complete experience, including optional reasoning.
Hardware
CPU: 6–8 cores recommended
RAM: 8–12 GB
Disk Space: ~8–10 GB free
GPU: Optional (not required)

Supported Features
All CV pipeline features
Streamlit UI
Optional Vision-Language Model (VLM)
Stable inference without memory issues

🧠 Vision-Language Model (VLM) Requirements
The project includes an optional VLM for natural-language reasoning.
VLM Used
Model: Qwen2.5-0.5B-Instruct
Type: Text-only reasoning (no vision encoder)
Framework: PyTorch (CPU)

Memory Usage
Approximate RAM: ~0.8–1.2 GB
Loading behavior: Lazy-loaded (only when a question is asked)

Recommendation
RAM ≥ 8 GB for enabling VLM
VLM is excluded from performance benchmarks

📊 Memory Requirement Summary
ComponentRAM RequirementOpenVINO FP16 inference2–4 GBYOLO + OpenVINO CV pipeline4–6 GBStreamlit UI6–8 GBVLM (Qwen2.5-0.5B)~1 GBFull system (recommended)8–12 GB

⚠️ Notes & Limitations
GPU acceleration is not required
Performance benchmarks measure CV pipeline only
VLM latency is not included in FPS / latency metrics
Large datasets and heavy transformer training are not recommended on CPU-only systems

✅ Summary
This project is designed to:
Run on CPU-only systems
Be stable on low-resource environments
Scale gracefully on higher RAM systems
The tested system (Ubuntu 24.04, Python 3.11, 12 GB RAM) provides a stable and recommended baseline for development and experimentation.

📌 File Placement Recommendation
Place this file at the project root:
SYSTEM_REQUIREMENTS.md

