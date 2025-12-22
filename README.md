# MobileNetV3 Vehicle Classification (2W / 4W) with OpenVINO

## 📌 Overview
This project implements a **vehicle classification system** using **MobileNetV3** to classify images into:
- **Two-Wheeler (2W)**
- **Four-Wheeler (4W)**

The project compares **baseline PyTorch FP32 inference** with **optimized OpenVINO FP16 inference** and presents results through a **Streamlit-based GUI**, including **performance metrics** such as latency and FPS.

---

## 🎯 Objectives
- Build a vision inference pipeline **from scratch**
- Explicitly manage model weights and formats
- Compare **PyTorch vs OpenVINO** inference performance
- Provide a **browser-based GUI**
- Demonstrate real-world **edge AI optimization**

---

## 🧠 Key Technologies
- **PyTorch** – baseline FP32 inference
- **ONNX** – intermediate model format
- **OpenVINO** – optimized FP16 inference on CPU
- **Streamlit** – GUI for inference & comparison
- **MobileNetV3** – lightweight CNN for vision tasks

---

## 🏗 Architecture (High Level)

Input Image
↓
Preprocessing (Resize, Normalize)
↓
Inference Engine
├── PyTorch FP32
└── OpenVINO FP16
↓
Postprocessing (Softmax, Class Mapping)
↓
Performance Metrics (Latency, FPS)
↓
Streamlit GUI (Visualization & Comparison)
---

## 🚀 Features
- Image upload via GUI
- Backend selection:
  - PyTorch FP32
  - OpenVINO FP16
  - Comparison mode
- Real-time performance metrics
- Clean separation of preprocessing, inference, and UI layers

---

## ▶️ How to Run

### 1️⃣ Create virtual environment
```bash
python3 -m venv ov_env
source ov_env/bin/activate
pip install -r requirements.txt
2️⃣ Ensure models exist
python
Copy code
models/
 ├── mobilenetv3.pth
 ├── mobilenetv3.xml
 └── mobilenetv3.bin
3️⃣ Launch Streamlit GUI
bash
Copy code
streamlit run src/app.py
📊 Performance Comparison (Sample)
Backend	Precision	Avg Latency (ms)	FPS
PyTorch	FP32	~40 ms	~25
OpenVINO	FP16	~7 ms	~140

