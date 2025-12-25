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

👤 Author
Dharmateja
QA Engineer | AI & OpenVINO Practitioner

yaml
Copy code
