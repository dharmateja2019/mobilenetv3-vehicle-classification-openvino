Input Image
   ↓
Vehicle Detection (SSD-MobileNet)
   ↓
Crop Vehicle(s)
   ↓
Vehicle Classification (PyTorch / OpenVINO)
   ↓
Vehicle Color Detection (HSV)
   ↓
Optional VLM Reasoning
     ├─ Vehicle-level reasoning
     └─ Full-image fallback (if detection fails)

🚀 Features

SSD-MobileNet vehicle detection (COCO)

MobileNetV3 classification (ImageNet)

Classical CV color detection (HSV)

Refined center-crop color extraction

PyTorch FP32 vs OpenVINO FP16 comparison

Per-backend latency & FPS metrics

CLI for automation & benchmarking

Streamlit GUI with Run button

Real Vision–Language Model (VLM) integration

Clean compare mode (PyTorch vs OpenVINO)

🛠 Requirements

Python 3.11

OS: Linux / macOS / Windows

CPU (GPU / FPGA selectable in GUI, optional)

🧪 Environment Setup
1️⃣ Create virtual environment
python3 -m venv mobilenet_env
source mobilenet_env/bin/activate

2️⃣ Install dependencies
pip install --upgrade pip
pip install torch torchvision
pip install opencv-python
pip install openvino
pip install streamlit pandas numpy
pip install openai

🔑 VLM Setup (Required for VLM)

This project uses a real Vision–Language Model (OpenAI GPT-4o-mini).

Set your API key:

export OPENAI_API_KEY="your_api_key_here"


⚠️ VLM is optional and disabled by default.
It is not used for FPS benchmarking.

📦 Model Downloads (VERY IMPORTANT)
🔹 A. SSD-MobileNet (Vehicle Detection)
mkdir -p models/ssd_mobilenet
cd models/ssd_mobilenet

wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz

mv ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb .
wget https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/ssd_mobilenet_v2_coco_2018_03_29.pbtxt


Final structure:

models/ssd_mobilenet/
├── frozen_inference_graph.pb
└── ssd_mobilenet_v2_coco_2018_03_29.pbtxt

🔹 B. MobileNetV3 OpenVINO Model (Classification)

Place converted OpenVINO IR files here:

models/ir/
├── mobilenetv3.xml
└── mobilenetv3.bin


⚠️ Conversion scripts are intentionally excluded.

📁 Project Structure
project_root/
├── app.py
├── README.md
├── outputs/
├── models/
│   ├── ssd_mobilenet/
│   └── ir/
└── src/
    ├── run_infer_cli.py
    ├── detector_ssd_mobilenet.py
    ├── inference_baseline.py
    ├── inference_openvino.py
    ├── color_extractor.py
    ├── preprocess.py
    ├── postprocess.py
    ├── metrics.py
    └── vlm_reasoner.py

🧪 CLI Usage
▶ Run with OpenVINO
python src/run_infer_cli.py --image path/to/image.jpg --backend openvino

▶ Run with PyTorch
python src/run_infer_cli.py --image path/to/image.jpg --backend pytorch

▶ Compare PyTorch vs OpenVINO
python src/run_infer_cli.py --image path/to/image.jpg --backend compare

▶ Enable VLM reasoning
python src/run_infer_cli.py --image path/to/image.jpg --backend openvino --use-vlm


CLI Output Includes:

   Per-backend latency & FPS

   Per-vehicle type & color

   Optional VLM reasoning

   VLM fallback when detection fails

🖥 Streamlit GUI
streamlit run app.py

GUI Features

   Upload image

   Select backend (PyTorch / OpenVINO / Compare)

   Select device (CPU / GPU / AUTO / FPGA – UI level)

   Run Inference button

   View bounding boxes

   Per-vehicle type & color

   Per-backend latency & FPS

   Optional VLM reasoning section

   Annotated output image

⚠️ Known Limitations (Current Stage)

   SSD-MobileNet may misclassify:

   Front-facing cars

   Scooters in narrow lanes

   ImageNet classifier is not vehicle-specific

   VLM adds latency and should not be used for benchmarking

   These are model limitations, not pipeline bugs.

🔮 Planned Enhancements

   YOLOv8 detector option

   Better 2-wheeler recall

   VLM override policy (trust VLM over CV)

   Video & multi-object tracking

   GPU / FPGA enablement in OpenVINO

   CSV export of benchmark results

👨‍💻 Engineering Notes

This project emphasizes system design over raw accuracy.

It demonstrates:

   Multi-backend inference

   Honest benchmarking

   Explainable AI via VLM

   Clean CLI & GUI separation

   Real-world AI orchestration patterns