import numpy as np
import streamlit as st

import utils_classification as cls_utils


def render_eye_controls(eye: str):
    st.markdown(f"### {eye.title()} Eye Controls")

    st.selectbox(
        f"{eye.title()} explainability method",
        options=["GradCAM++", "ScoreCAM"],
        key=f"{eye}_analysis_cam_method",
        help="GradCAM++ is the baseline. ScoreCAM is slower but useful for comparison.",
    )

    st.selectbox(
        f"{eye.title()} target class",
        options=list(range(cls_utils.NUM_CLASSES)),
        format_func=lambda i: cls_utils.CLASS_NAMES[i],
        key=f"{eye}_analysis_target_class",
    )

    st.slider(
        f"{eye.title()} overlay strength",
        min_value=0.10,
        max_value=0.80,
        step=0.05,
        key=f"{eye}_analysis_alpha",
    )


def render_original_single(c: dict, primary_eye: str):
    st.markdown("### Original Image")
    col1, col2 = st.columns([1.85, 0.95], gap="large")
    with col1:
        st.image(c["raw_rgb"], use_container_width=True)
    with col2:
        conf = float(c["probs"][c["pred_idx"]])
        st.markdown("### View Summary")
        st.markdown(
            f"""
            <div class="pred-card">
                <b>Prediction:</b> {c["pred_name"]}<br>
                <b>Confidence:</b> {conf*100:.1f}%<br><br>
                This is the untouched input image used by both the classification and segmentation models.
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_original_pair(computed: dict):
    st.markdown("### Original Images")
    col1, col2 = st.columns(2, gap="large")

    if "right" in computed:
        with col1:
            st.markdown("#### Right Eye")
            st.image(computed["right"]["raw_rgb"], use_container_width=True)

    if "left" in computed:
        with col2:
            st.markdown("#### Left Eye")
            st.image(computed["left"]["raw_rgb"], use_container_width=True)


def render_explainability_single(c: dict, primary_eye: str):
    st.markdown(f"### {c['cam_method_ui']} View")
    st.caption(f"Powered by the classification model for the {primary_eye.title()} eye.")

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown(f"#### {c['cam_method_ui']} Heatmap")
        st.image(c["heatmap_rgb"], use_container_width=True)
    with col2:
        st.markdown(f"#### {c['cam_method_ui']} Overlay")
        st.image(c["cam_overlay_rgb"], use_container_width=True)


def render_explainability_pair_vertical(computed: dict):
    st.markdown("### Explainability Comparison")

    if "right" in computed:
        c = computed["right"]
        row_left, row_right = st.columns([2.0, 0.95], gap="large")

        with row_left:
            st.markdown("#### Right Eye")
            img1, img2 = st.columns(2, gap="medium")
            with img1:
                st.image(c["heatmap_rgb"], use_container_width=True)
            with img2:
                st.image(c["cam_overlay_rgb"], use_container_width=True)

        with row_right:
            render_eye_controls("right")

        st.markdown("---")

    if "left" in computed:
        c = computed["left"]
        row_left, row_right = st.columns([2.0, 0.95], gap="large")

        with row_left:
            st.markdown("#### Left Eye")
            img1, img2 = st.columns(2, gap="medium")
            with img1:
                st.image(c["heatmap_rgb"], use_container_width=True)
            with img2:
                st.image(c["cam_overlay_rgb"], use_container_width=True)

        with row_right:
            render_eye_controls("left")


def render_ex_single(c: dict, primary_eye: str):
    st.markdown("### Exudates (EX)")
    st.caption(f"Powered by the segmentation model for the {primary_eye.title()} eye.")
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("#### EX Mask")
        st.image((c["seg"]["ex_mask"] * 255).astype(np.uint8), use_container_width=True, clamp=True)
    with col2:
        st.markdown("#### EX Overlay")
        st.image(c["seg"]["ex_overlay"], use_container_width=True)


def render_ex_pair(computed: dict):
    st.markdown("### Exudates (EX)")
    col1, col2 = st.columns(2, gap="large")

    if "right" in computed:
        c = computed["right"]
        with col1:
            st.markdown("#### Right Eye")
            a, b = st.columns(2, gap="medium")
            with a:
                st.image((c["seg"]["ex_mask"] * 255).astype(np.uint8), use_container_width=True, clamp=True)
            with b:
                st.image(c["seg"]["ex_overlay"], use_container_width=True)

    if "left" in computed:
        c = computed["left"]
        with col2:
            st.markdown("#### Left Eye")
            a, b = st.columns(2, gap="medium")
            with a:
                st.image((c["seg"]["ex_mask"] * 255).astype(np.uint8), use_container_width=True, clamp=True)
            with b:
                st.image(c["seg"]["ex_overlay"], use_container_width=True)


def render_he_single(c: dict, primary_eye: str):
    st.markdown("### Haemorrhages (HE)")
    st.caption(f"Powered by the segmentation model for the {primary_eye.title()} eye.")
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("#### HE Mask")
        st.image((c["seg"]["he_mask"] * 255).astype(np.uint8), use_container_width=True, clamp=True)
    with col2:
        st.markdown("#### HE Overlay")
        st.image(c["seg"]["he_overlay"], use_container_width=True)


def render_he_pair(computed: dict):
    st.markdown("### Haemorrhages (HE)")
    col1, col2 = st.columns(2, gap="large")

    if "right" in computed:
        c = computed["right"]
        with col1:
            st.markdown("#### Right Eye")
            a, b = st.columns(2, gap="medium")
            with a:
                st.image((c["seg"]["he_mask"] * 255).astype(np.uint8), use_container_width=True, clamp=True)
            with b:
                st.image(c["seg"]["he_overlay"], use_container_width=True)

    if "left" in computed:
        c = computed["left"]
        with col2:
            st.markdown("#### Left Eye")
            a, b = st.columns(2, gap="medium")
            with a:
                st.image((c["seg"]["he_mask"] * 255).astype(np.uint8), use_container_width=True, clamp=True)
            with b:
                st.image(c["seg"]["he_overlay"], use_container_width=True)