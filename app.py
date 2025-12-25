import streamlit as st
import os
import pandas as pd
from PIL import Image
from src.pipeline import run_pipeline

st.set_page_config(page_title="Vehicle AI", layout="wide")
st.title("🚗 Vehicle Detection & Scene Understanding")

# ---- Sidebar ----
backend = st.sidebar.selectbox(
    "Backend",
    ["openvino", "pytorch", "compare"]
)

device = st.sidebar.selectbox(
    "Device",
    ["CPU"]
)

qa_question = st.sidebar.text_input(
    "Ask a question about the image (VLM Q/A)",
    placeholder="e.g. How many vehicles are visible?"
)

uploaded = st.sidebar.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

run_btn = st.sidebar.button("▶ Run Inference")

if not uploaded:
    st.info("⬅ Upload an image")
    st.stop()

os.makedirs("outputs", exist_ok=True)
input_path = "outputs/input.jpg"
image = Image.open(uploaded).convert("RGB")
image.save(input_path)

st.subheader("Input Image")
st.image(image, width=600)

if run_btn:
    with st.spinner("Running inference..."):
        result = run_pipeline(
            image_path=input_path,
            backend=backend,
            device=device,
            qa_question=qa_question or None,
        )

    st.success("Inference completed")

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

    else:
        st.subheader(f"🧠 {backend.upper()} Results")
        st.metric("Latency (ms)", result["latency_ms"])
        st.metric("FPS", result["fps"])
        st.table(pd.DataFrame(result["vehicles"]))
        st.image("outputs/result.jpg")

    # ---- VLM Q/A DISPLAY ----
    if qa_question:
        st.subheader("🧠 Visual Question Answering")
        st.markdown(f"**Q:** {qa_question}")
        st.success(result.get("qa_answer"))
