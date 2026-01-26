import streamlit as st

st.set_page_config(
    page_title="Lesion-Aware DR Grading",
    layout="wide",
)

st.title("🩺 Lesion-Aware Diabetic Retinopathy Grading")
st.markdown("""
This application demonstrates:
- Deep learning–based DR classification
- Class imbalance handling
- Explainability using Grad-CAM
""")

st.info("Use the sidebar to navigate between modules.")
