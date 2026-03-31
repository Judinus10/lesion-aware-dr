import numpy as np
import streamlit as st

import utils_classification as cls_utils


def render_sorted_probabilities(probs: np.ndarray):
    order = np.argsort(-probs)
    for i in order:
        st.write(f"**{cls_utils.CLASS_NAMES[i]}** — {probs[i] * 100:.1f}%")
        st.progress(min(max(float(probs[i]), 0.0), 1.0))


def render_eye_prediction_summary_two_col(eye: str, c: dict):
    conf = float(c["probs"][c["pred_idx"]])

    st.markdown(f"## {eye.title()} Eye")
    left_col, right_col = st.columns([1.15, 1.85], gap="large")

    with left_col:
        st.markdown("### Prediction")
        st.markdown(f"# {c['pred_name']}")
        st.write(f"**Confidence score:** {conf * 100:.1f}%")
        st.progress(min(max(conf, 0.0), 1.0))

        if np.allclose(c["probs"], c["probs"][0], atol=1e-3):
            st.error("All class probabilities are equal (model issue).")
        elif conf < 0.35:
            st.warning("Low confidence. Treat as uncertain.")

    with right_col:
        st.markdown("### Class Probabilities (sorted)")
        render_sorted_probabilities(c["probs"])