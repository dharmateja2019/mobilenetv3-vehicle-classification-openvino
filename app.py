import streamlit as st
import os
import pandas as pd
from PIL import Image
from src.pipeline import run_pipeline

st.set_page_config(page_title="Vehicle AI", layout="wide")
st.title("🚗 Vehicle Detection & Scene Understanding")

# ---------------- Sidebar ----------------
backend = st.sidebar.selectbox(
    "Backend",
    ["openvino", "pytorch", "compare"]
)

device = st.sidebar.selectbox(
    "Device",
    ["CPU"]
)

enable_vlm = st.sidebar.checkbox(
    "Enable VLM (slower, uses more RAM)",
    value=False
)

qa_question = None
if enable_vlm:
    qa_question = st.sidebar.text_input(
        "Ask a question about the image",
        placeholder="e.g. What are the colors of the vehicles?"
    )

uploaded = st.sidebar.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

run_btn = st.sidebar.button("▶ Run Inference")

if not uploaded:
    st.info("⬅ Upload an image to begin")
    st.stop()

# ---------------- Input Image ----------------
os.makedirs("outputs", exist_ok=True)
input_path = "outputs/input.jpg"

image = Image.open(uploaded).convert("RGB")
image.save(input_path)

st.subheader("Input Image")
st.image(image, width=600)

# ---------------- Run Pipeline ----------------
if run_btn:
    with st.spinner("Running inference..."):
        result = run_pipeline(
            image_path=input_path,
            backend=backend,
            device=device,
            qa_question=qa_question.strip() if qa_question else None,
        )

    st.success("Inference completed")

    # ---------- Compare Mode ----------
    if backend == "compare":
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🟦 PyTorch")
            st.metric("Latency (ms)", result["compare"]["pytorch"]["latency_ms"])
            st.metric("FPS", result["compare"]["pytorch"]["fps"])
            st.table(pd.DataFrame(result["compare"]["pytorch"]["vehicles"]))
            st.image("outputs/result_pytorch.jpg")

        with col2:
            st.subheader("🟩 OpenVINO")
            st.metric("Latency (ms)", result["compare"]["openvino"]["latency_ms"])
            st.metric("FPS", result["compare"]["openvino"]["fps"])
            st.table(pd.DataFrame(result["compare"]["openvino"]["vehicles"]))
            st.image("outputs/result_openvino.jpg")

        if enable_vlm:
            answer = result["compare"]["openvino"].get("qa_answer")
            if answer:
                st.subheader("🧠 VLM Answer (OpenVINO)")
                st.success(answer)

    # ---------- Single Backend ----------
    else:
        st.subheader(f"🧠 {backend.upper()} Results")
        st.metric("Latency (ms)", result["latency_ms"])
        st.metric("FPS", result["fps"])
        st.table(pd.DataFrame(result["vehicles"]))
        st.image("outputs/result.jpg")

        if enable_vlm:
            answer = result.get("qa_answer")
            if answer:
                st.subheader("🧠 VLM Answer")
                st.success(answer)
