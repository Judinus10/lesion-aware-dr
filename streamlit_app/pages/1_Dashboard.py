import streamlit as st
from PIL import Image

import utils_classification as cls_utils


st.set_page_config(page_title="Dashboard", layout="wide")

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
if "eye_results" not in st.session_state:
    st.session_state.eye_results = {}
if "analysis_input_mode" not in st.session_state:
    st.session_state.analysis_input_mode = "single"
if "primary_eye" not in st.session_state:
    st.session_state.primary_eye = None
if "last_uploaded_name" not in st.session_state:
    st.session_state.last_uploaded_name = ""
if "last_uploaded_names" not in st.session_state:
    st.session_state.last_uploaded_names = {"right": "", "left": ""}
if "single_eye_confirm_pending" not in st.session_state:
    st.session_state.single_eye_confirm_pending = False
if "single_eye_target" not in st.session_state:
    st.session_state.single_eye_target = None


# ---- Cached classification model ----
@st.cache_resource
def _cached_cls_model():
    return cls_utils.load_model_cached()


def show_center_loader(message="Preparing analysis..."):
    loader = st.empty()
    loader.markdown(
        f"""
        <div style="
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 260px;
            width: 100%;
        ">
            <div style="
                display: flex;
                align-items: center;
                gap: 12px;
                padding: 18px 26px;
                border-radius: 14px;
                background: rgba(255,255,255,0.02);
                border: 1px solid rgba(255,255,255,0.08);
                font-size: 20px;
                font-weight: 500;
            ">
                <div class="custom-loader"></div>
                <span>{message}</span>
            </div>
        </div>

        <style>
        .custom-loader {{
            width: 22px;
            height: 22px;
            border: 3px solid rgba(255,255,255,0.18);
            border-top: 3px solid rgba(255,255,255,0.95);
            border-radius: 50%;
            animation: spin 0.9s linear infinite;
        }}

        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    return loader


def clear_loader(loader):
    loader.empty()


def reset_single_eye_confirmation():
    st.session_state.single_eye_confirm_pending = False
    st.session_state.single_eye_target = None


def pil_from_upload(uploaded_file):
    if uploaded_file is None:
        return None
    return Image.open(uploaded_file).convert("RGB")


def build_eye_result(result_dict, uploaded_name):
    return {
        "pred_idx": result_dict["pred_idx"],
        "pred_name": result_dict["pred_name"],
        "probs": result_dict["probs"],
        "raw_rgb": result_dict["raw_rgb"],
        "input_tensor": result_dict["input_tensor"].detach(),
        "uploaded_name": uploaded_name,
    }


def run_prediction_and_switch(right_img, left_img, model, device):
    eye_results = {}

    if right_img is not None:
        right_pred = cls_utils.predict(model, device, right_img)
        eye_results["right"] = build_eye_result(
            right_pred,
            st.session_state.last_uploaded_names.get("right", "right_eye"),
        )

    if left_img is not None:
        left_pred = cls_utils.predict(model, device, left_img)
        eye_results["left"] = build_eye_result(
            left_pred,
            st.session_state.last_uploaded_names.get("left", "left_eye"),
        )

    st.session_state.eye_results = eye_results

    available_eyes = list(eye_results.keys())
    if len(available_eyes) == 2:
        st.session_state.analysis_input_mode = "pair"
        st.session_state.primary_eye = "right"
        st.session_state.last_result = eye_results["right"]
        st.session_state.last_uploaded_name = st.session_state.last_uploaded_names.get("right", "")
    else:
        only_eye = available_eyes[0]
        st.session_state.analysis_input_mode = "single"
        st.session_state.primary_eye = only_eye
        st.session_state.last_result = eye_results[only_eye]
        st.session_state.last_uploaded_name = st.session_state.last_uploaded_names.get(only_eye, "")

    reset_single_eye_confirmation()
    st.switch_page("pages/2_Analysis.py")


# ---- Header ----
st.markdown(
    """
    <div style='text-align: center; padding-top: 10px; padding-bottom: 10px;'>
        <h1 style='margin-bottom: 5px; text-align: center;'>
            Lesion-Aware DR Prediction System
        </h1>
        <p style='font-size:16px; color: #9CA3AF; text-align: center; margin-bottom: 0;'>
            Multi-model student research demo: DR grading + explainability + lesion segmentation
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

# ---- Layout ----
left, right = st.columns([2.2, 1.0], gap="large")

with right:
    st.subheader("Rules / Notes")
    st.info(
        "✅ You can upload both Right and Left eye images\n\n"
        "✅ Single-eye prediction is also allowed\n\n"
        "✅ Supported: JPG / PNG\n\n"
        "✅ Best results when retina is centered and not overexposed\n\n"
        "⚠ This is a student research demo\n\n"
        "⚠ Not for clinical diagnosis\n\n"
        "⚠ Explainability and segmentation are support tools, not proof"
    )

    st.markdown("---")
    if st.button("View Saved Cases", key="view_saved_cases_btn", use_container_width=True):
        st.switch_page("pages/3_Saved_Cases.py")

with left:
    st.subheader("Upload Fundus Images")

    up1, up2 = st.columns(2, gap="large")

    with up1:
        uploaded_right = st.file_uploader(
            "Right Eye Image",
            type=["jpg", "jpeg", "png"],
            key="right_eye_uploader",
        )

    with up2:
        uploaded_left = st.file_uploader(
            "Left Eye Image",
            type=["jpg", "jpeg", "png"],
            key="left_eye_uploader",
        )

    right_img = pil_from_upload(uploaded_right)
    left_img = pil_from_upload(uploaded_left)

    st.session_state.last_uploaded_names = {
        "right": uploaded_right.name if uploaded_right else "",
        "left": uploaded_left.name if uploaded_left else "",
    }

    current_uploaded_count = int(right_img is not None) + int(left_img is not None)
    if current_uploaded_count != 1:
        reset_single_eye_confirmation()

    if right_img is None and left_img is None:
        st.warning("Upload at least one image. Right, Left, or both.")
        st.stop()

    # ---- Preview ----
    st.markdown("### Preview")

    both_uploaded = right_img is not None and left_img is not None

    if both_uploaded:
        pc1, pc2 = st.columns(2, gap="large")
        with pc1:
            st.markdown("#### Right Eye")
            st.image(right_img, use_container_width=True)
        with pc2:
            st.markdown("#### Left Eye")
            st.image(left_img, use_container_width=True)

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        _, action_center, _ = st.columns([1, 2, 1])
        with action_center:
            try:
                model, device = _cached_cls_model()
            except Exception as e:
                st.error(str(e))
                st.stop()

            predict_clicked = st.button(
                "Predict & Analyze",
                key="predict_analyze_pair",
                type="primary",
                use_container_width=True,
            )

            if predict_clicked:
                loader = show_center_loader("Preparing paired-eye analysis...")
                run_prediction_and_switch(right_img, left_img, model, device)
                clear_loader(loader)

    else:
        if right_img is not None:
            pc1, pc2 = st.columns([1.4, 1.0], gap="large")
            with pc1:
                st.markdown("#### Right Eye")
                st.image(right_img, use_container_width=True)
        else:
            pc1, pc2 = st.columns([1.4, 1.0], gap="large")
            with pc1:
                st.markdown("#### Left Eye")
                st.image(left_img, use_container_width=True)

        with pc2:
            st.markdown("### Actions")

            try:
                model, device = _cached_cls_model()
            except Exception as e:
                st.error(str(e))
                st.stop()

            predict_clicked = st.button(
                "Predict & Analyze",
                key="predict_analyze_single",
                type="primary",
                use_container_width=True,
            )

            if predict_clicked:
                single_side = "right" if right_img is not None else "left"
                st.session_state.single_eye_confirm_pending = True
                st.session_state.single_eye_target = single_side

            if st.session_state.single_eye_confirm_pending:
                target_side = st.session_state.single_eye_target

                if target_side == "right" and right_img is not None:
                    st.warning(
                        f"Only the Right eye image was uploaded "
                        f"({st.session_state.last_uploaded_names.get('right', 'right_eye')}). "
                        f"Do you want to continue with single-eye prediction?"
                    )

                    c1, c2 = st.columns(2, gap="small")
                    with c1:
                        if st.button(
                            "Continue",
                            key="confirm_single_right",
                            use_container_width=True,
                        ):
                            loader = show_center_loader("Preparing single-eye analysis...")
                            run_prediction_and_switch(right_img, None, model, device)
                            clear_loader(loader)

                    with c2:
                        if st.button(
                            "Cancel",
                            key="cancel_single_right",
                            use_container_width=True,
                        ):
                            reset_single_eye_confirmation()

                elif target_side == "left" and left_img is not None:
                    st.warning(
                        f"Only the Left eye image was uploaded "
                        f"({st.session_state.last_uploaded_names.get('left', 'left_eye')}). "
                        f"Do you want to continue with single-eye prediction?"
                    )

                    c1, c2 = st.columns(2, gap="small")
                    with c1:
                        if st.button(
                            "Continue",
                            key="confirm_single_left",
                            use_container_width=True,
                        ):
                            loader = show_center_loader("Preparing single-eye analysis...")
                            run_prediction_and_switch(None, left_img, model, device)
                            clear_loader(loader)

                    with c2:
                        if st.button(
                            "Cancel",
                            key="cancel_single_left",
                            use_container_width=True,
                        ):
                            reset_single_eye_confirmation()