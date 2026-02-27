import streamlit as st
import torch
import os

# Optional: keep these consistent with your other pages
CKPT_PATH = "outputs/2026-02-27/checkpoints/best_macro_f1_model.pt"
MODEL_NAME = "efficientnet_b0"
CLASS_NAMES = ["No DR (0)", "Mild (1)", "Moderate (2)", "Severe (3)", "Proliferative (4)"]

st.set_page_config(page_title="Dashboard", layout="wide")

st.title("Lesion-Aware DR Grading System")
st.caption("Demo UI: Predict DR grade + Explainability (Grad-CAM++)")

# -------- Status Cards --------
c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Device", "CUDA ✅" if torch.cuda.is_available() else "CPU")

with c2:
    st.metric("Model", MODEL_NAME)

with c3:
    ckpt_ok = os.path.exists(CKPT_PATH)
    st.metric("Checkpoint", "Found ✅" if ckpt_ok else "Missing ❌")

st.divider()

# -------- How to Use --------
st.subheader("How to Use")
st.markdown(
    """
1) Go to **Predict** page → upload a fundus image → get **grade + probabilities**  
2) Go to **Explainability** page → upload the same image → get **Grad-CAM++ heatmap**  
"""
)

# -------- Classes --------
st.subheader("DR Classes")
st.write("The system predicts one of the following grades:")
for name in CLASS_NAMES:
    st.write(f"- {name}")

st.divider()

# -------- Notes / Warnings --------
st.subheader("Notes")
st.warning(
    "This is a student research demo. Not for clinical diagnosis. "
    "Heatmaps are *explanations*, not proof of medical correctness."
)

# -------- Quick Troubleshooting --------
with st.expander("Quick Troubleshooting"):
    st.write("If Checkpoint shows Missing ❌:")
    st.code(
        "1) Verify CKPT_PATH inside Dashboard/Predict/Explainability\n"
        "2) Make sure you run Streamlit from project root\n"
        "3) Confirm the checkpoint file exists in that path"
    )
    st.write("If predictions are wrong or error happens:")
    st.write("- MODEL_NAME must match the model used during training")
    st.write("- IMG_SIZE / preprocessing must match training")