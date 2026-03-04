import json
import streamlit as st
import numpy as np

import utils

st.set_page_config(page_title="Analysis", layout="wide")

# Hide sidebar nav
st.markdown(
    """
    <style>
      [data-testid="stSidebar"] {display: none;}
      [data-testid="stSidebarNav"] {display: none;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Guard: require a prediction ----
result = st.session_state.get("last_result", None)
if result is None:
    st.warning("No prediction found. Go back and run prediction first.")
    if st.button("Go to Dashboard"):
        st.switch_page("pages/1_Dashboard.py")
    st.stop()

# Load model
@st.cache_resource
def _cached_model():
    return utils.load_model_cached()

model, device = _cached_model()

# Pull stored prediction data
pred_idx = int(result["pred_idx"])
pred_name = result["pred_name"]
probs = np.array(result["probs"], dtype=np.float32)
raw_rgb = result["raw_rgb"]
input_tensor = result["input_tensor"].to(device)

filename_hint = st.session_state.get("last_uploaded_name", "fundus")
default_layer = st.session_state.get("preferred_layer", None)

# ---------------------------
# Controls state (so we can compute CAM BEFORE rendering top buttons)
# ---------------------------
if "analysis_target_class" not in st.session_state:
    st.session_state.analysis_target_class = pred_idx
if "analysis_alpha" not in st.session_state:
    st.session_state.analysis_alpha = 0.45
if "analysis_target_layer" not in st.session_state:
    st.session_state.analysis_target_layer = "(auto)"

# Read controls from session (these are used for CAM compute)
target_class = int(st.session_state.analysis_target_class)
alpha = float(st.session_state.analysis_alpha)
picked_layer = st.session_state.analysis_target_layer

# Build preferred layer name
convs = utils.list_conv2d_layers(model)
layer_names = [n for n, _ in convs] if convs else []
preferred = None if picked_layer == "(auto)" else picked_layer

# ---------------------------
# Compute CAM EARLY (so Save/Download can be at the top)
# ---------------------------
with st.spinner("Preparing analysis…"):
    layer_name, layer_module = utils.choose_target_layer(model, preferred_name=preferred)
    cam_mask = utils.run_gradcam_pp(model, input_tensor, int(target_class), layer_module)
    overlay_rgb, heatmap_rgb = utils.overlay_heatmap(raw_rgb, cam_mask, alpha=float(alpha))

# Prepare export JSON EARLY
export = {
    "filename_hint": filename_hint,
    "pred_idx": pred_idx,
    "pred_name": pred_name,
    "probs": [float(x) for x in probs.tolist()],
    "target_class": int(target_class),
    "target_layer": layer_name,
    "alpha": float(alpha),
}

# ---------------------------
# TOP BAR: Back (left) + Save/Download (right) SAME LINE ✅
# ---------------------------
top_left, top_right = st.columns([3, 1], gap="large")

with top_left:
    if st.button("⬅ Back to Dashboard"):
        st.switch_page("pages/1_Dashboard.py")

with top_right:
    b1, b2 = st.columns(2, gap="small")

    with b1:
        if st.button("💾 Save", use_container_width=True):
            record = utils.save_case(
                filename_hint=filename_hint,
                raw_rgb=raw_rgb,
                heatmap_rgb=heatmap_rgb,
                overlay_rgb=overlay_rgb,
                pred_idx=pred_idx,
                probs=probs,
                target_class=int(target_class),
                target_layer_name=layer_name,
            )
            st.success(f"Saved ✅ {record['case_id']}")

    with b2:
        st.download_button(
            "⬇ JSON",
            data=json.dumps(export, indent=2),
            file_name="dr_result.json",
            mime="application/json",
            use_container_width=True,
        )

# Title
st.title("Prediction Result")
st.divider()

# ---------------------------
# Prediction Summary
# ---------------------------
sum1, sum2 = st.columns([1.1, 1.9], gap="large")

conf = float(probs[pred_idx])

with sum1:
    st.subheader("Predicted Grade")
    st.metric("Prediction", pred_name)

    st.write(f"**Confidence score:** {conf*100:.1f}%")
    st.caption("Confidence = softmax probability of the predicted class.")
    st.progress(min(max(conf, 0.0), 1.0))

    if np.allclose(probs, probs[0], atol=1e-3):
        st.error("All class probabilities are equal (model uncertainty / checkpoint issue).")
    elif conf < 0.35:
        st.warning("Low confidence. Treat this prediction as uncertain.")

with sum2:
    st.subheader("Class Probabilities (sorted)")
    order = np.argsort(-probs)
    for i in order:
        st.write(f"**{utils.CLASS_NAMES[i]}** — {probs[i]*100:.1f}%")
        st.progress(min(max(float(probs[i]), 0.0), 1.0))

st.divider()

# ---------------------------
# Explainability + Controls + Saved Cases
# ---------------------------
left, right = st.columns([2.2, 1.0], gap="large")

with right:
    st.subheader("Explainability Controls")

    # These widgets update session_state, and we force rerun so CAM updates
    new_target = st.selectbox(
        "Generate heatmap for class",
        options=list(range(utils.NUM_CLASSES)),
        index=int(target_class),
        format_func=lambda i: utils.CLASS_NAMES[i],
        key="analysis_target_class",
    )

    new_alpha = st.slider(
        "Overlay strength",
        0.10, 0.80, float(alpha), 0.05,
        key="analysis_alpha",
    )

    if layer_names:
        st.selectbox(
            "Target layer",
            options=["(auto)"] + layer_names,
            index=0 if default_layer is None else (
                layer_names.index(default_layer) + 1 if default_layer in layer_names else 0
            ),
            key="analysis_target_layer",
        )
    else:
        st.session_state.analysis_target_layer = "(auto)"
        st.caption("No Conv2D layers detected.")

    # Rerun if changed (to recompute CAM)
    if (int(new_target) != int(target_class)) or (float(new_alpha) != float(alpha)):
        st.rerun()

    st.markdown("---")
    st.subheader("Saved Cases")
    cases = utils.load_recent_cases(limit=15)
    if cases:
        st.write(f"Recent saved: **{len(cases)}**")
        with st.expander("Show recent cases"):
            for c in cases[:10]:
                st.write(f"- {c['timestamp']} — **{c['pred_name']}** — {c['filename_hint']}")
    else:
        st.caption("No saved cases yet.")

with left:
    st.subheader("Why did the model predict this?")
    st.caption("Grad-CAM++ highlights regions contributing to the selected class score.")

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown("### Input")
        st.image(raw_rgb, use_container_width=True)
    with c2:
        st.markdown("### Heatmap")
        st.image(heatmap_rgb, use_container_width=True)
    with c3:
        st.markdown("### Overlay")
        st.image(overlay_rgb, use_container_width=True)

st.divider()

# Bottom highlighted reminder (like before)
st.info("⚠ Research demo only — not medical advice. Heatmaps show model attention, not clinical truth.")