import os
import streamlit as st
import torch

from PIL import Image

import utils


st.set_page_config(page_title="Dashboard", layout="wide")

# Hide sidebar nav (we want 2-page clean flow)
st.markdown(
    """
    <style>
      [data-testid="stSidebar"] {display: none;}
      [data-testid="stSidebarNav"] {display: none;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Session init ----
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_uploaded_name" not in st.session_state:
    st.session_state.last_uploaded_name = ""
if "preferred_layer" not in st.session_state:
    st.session_state.preferred_layer = None


# ---- Header ----
st.title("Lesion-Aware DR Prediction System")
st.caption("DR grading + Grad-CAM++ explainability (student demo, non-clinical)")

# ---- Top status row ----
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Device", "CUDA ✅" if torch.cuda.is_available() else "CPU")
with c2:
    st.metric("Model", utils.MODEL_NAME)
with c3:
    st.metric("Checkpoint", "Found ✅" if os.path.exists(utils.CKPT_PATH) else "Missing ❌")

st.divider()

# ---- Layout: Left main (upload + preview + predict), Right (Rules + View history)
left, right = st.columns([2.2, 1.0], gap="large")

with right:
    st.subheader("Rules / Notes")
    st.info(
        "✅ Upload a clear fundus image\n\n"
        "✅ Supported: JPG / PNG\n\n"
        "✅ Best results when retina is centered and not overexposed\n\n"
        "⚠ This is a student research demo\n\n"
        "⚠ Not for clinical diagnosis\n\n"
        "⚠ Heatmaps are explanations, not proof"
    )

    st.markdown("---")
    if st.button("📚 View Saved Cases", use_container_width=True):
        st.switch_page("pages/2_Analysis.py")

with left:
    st.subheader("Upload Fundus Image")
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if not uploaded:
        st.warning("Upload an image first. No image = no prediction.")
        st.stop()

    st.session_state.last_uploaded_name = uploaded.name

    pil_img = Image.open(uploaded)

    pcol, bcol = st.columns([1.4, 1.0], gap="large")

    with pcol:
        st.markdown("### Preview")
        st.image(pil_img, use_container_width=True)

    with bcol:
        st.markdown("### Actions")

        # Optional: allow user to choose target layer quality
        # We'll populate conv layers after model loads
        try:
            # Load model (cache via Streamlit wrapper below)
            @st.cache_resource
            def _cached_model():
                return utils.load_model_cached()

            model, device = _cached_model()
            convs = utils.list_conv2d_layers(model)
            layer_names = [n for n, _ in convs] if convs else []
        except Exception as e:
            st.error(str(e))
            st.stop()

        if layer_names:
            picked = st.selectbox(
                "Grad-CAM Target Layer (optional)",
                options=["(auto)"] + layer_names,
                index=0,
                help="Auto is usually fine. Picking a slightly earlier conv can reduce border bias sometimes.",
            )
            st.session_state.preferred_layer = None if picked == "(auto)" else picked

        st.markdown("")

        if st.button("🔍 Predict & Analyze", type="primary", use_container_width=True):
            with st.spinner("Running inference…"):
                result = utils.predict(model, device, pil_img)

                # store everything needed for Analysis page
                st.session_state.last_result = {
                    "pred_idx": result["pred_idx"],
                    "pred_name": result["pred_name"],
                    "probs": result["probs"],
                    "raw_rgb": result["raw_rgb"],
                    "input_tensor": result["input_tensor"].detach(),
                }

            st.switch_page("pages/2_Analysis.py")