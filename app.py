import streamlit as st
from PIL import Image
from src.inference_baseline import run_baseline
from src.inference_openvino import run_openvino


st.set_page_config(page_title="Vehicle Classification", layout="wide")

st.title("🚗 Vehicle Classification (2W / 4W)")
st.markdown("**PyTorch FP32 vs OpenVINO Performance Comparison**")

uploaded_file = st.file_uploader("Upload vehicle image", type=["jpg", "png", "jpeg"])

backend = st.selectbox(
    "Select backend",
    ["Baseline (PyTorch)", "OpenVINO", "Compare"]
)

device = st.selectbox("Device", ["CPU"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Input Image", use_container_width=True)

    if st.button("Run Inference"):
        if backend == "Baseline (PyTorch)":
            result = run_baseline(image)
            st.success("Inference completed")

            st.json(result)

        elif backend == "OpenVINO":
            result = run_openvino(image, device)
            st.success("Inference completed")

            st.json(result)

        else:
            base = run_baseline(image)
            ov = run_openvino(image, device)

            st.subheader("📊 Performance Comparison")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### PyTorch FP32")
                st.metric("Latency (ms)", f"{base['latency']:.2f}")
                st.metric("FPS", f"{base['fps']:.1f}")
                st.metric("Prediction", base["label"])
                st.metric("Confidence", f"{base['confidence']:.2f}")

            with col2:
                st.markdown("### OpenVINO FP16")
                st.metric("Latency (ms)", f"{ov['latency']:.2f}")
                st.metric("FPS", f"{ov['fps']:.1f}")
                st.metric("Prediction", ov["label"])
                st.metric("Confidence", f"{ov['confidence']:.2f}")
