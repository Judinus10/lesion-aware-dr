from typing import Dict, List, Tuple

import numpy as np
import streamlit as st
from PIL import Image

import utils_classification as cls_utils
import utils_segmentation as seg_utils


@st.cache_resource
def get_cached_cls_model():
    return cls_utils.load_model_cached()


@st.cache_resource
def get_cached_seg_model():
    return seg_utils.load_model_cached()


def prepare_eye_results_from_legacy() -> Tuple[str, Dict]:
    analysis_input_mode = st.session_state.get("analysis_input_mode", "single")
    eye_results = st.session_state.get("eye_results", {})
    legacy_result = st.session_state.get("last_result", None)

    if not eye_results and legacy_result is not None:
        primary_eye = st.session_state.get("primary_eye", "right")
        eye_results = {primary_eye: legacy_result}
        st.session_state.eye_results = eye_results

    return analysis_input_mode, eye_results


def ensure_analysis_session_defaults(eye_results: Dict) -> None:
    if "analysis_view_mode" not in st.session_state:
        st.session_state.analysis_view_mode = "Original"

    if "_pending_analysis_view_mode" in st.session_state:
        st.session_state.analysis_view_mode = st.session_state["_pending_analysis_view_mode"]
        del st.session_state["_pending_analysis_view_mode"]

    if "show_pdf_dialog" not in st.session_state:
        st.session_state.show_pdf_dialog = False

    if "generated_pdf_bytes" not in st.session_state:
        st.session_state.generated_pdf_bytes = None

    if "generated_pdf_name" not in st.session_state:
        st.session_state.generated_pdf_name = None

    if "show_save_case_dialog" not in st.session_state:
        st.session_state.show_save_case_dialog = False

    save_defaults = {
        "save_patient_id": st.session_state.get("saved_patient_id", ""),
        "save_patient_name": st.session_state.get("saved_patient_name", ""),
        "save_patient_age": st.session_state.get("saved_patient_age", ""),
        "save_patient_gender": st.session_state.get("saved_patient_gender", ""),
        "save_patient_notes": st.session_state.get("saved_patient_notes", ""),
    }
    for key, value in save_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    pdf_defaults = {
        "pdf_report_title": "DR Analysis Report",
        "pdf_patient_name": "",
        "pdf_patient_id": "",
        "pdf_clinician_name": "",
        "pdf_institution_name": "",
        "pdf_notes": "",
        "pdf_include_report_details": True,
        "pdf_include_prediction_summary": True,
        "pdf_include_probabilities": True,
        "pdf_include_original_image": False,
        "pdf_include_gradcam_heatmap": True,
        "pdf_include_gradcam_overlay": True,
        "pdf_include_exudates_mask": False,
        "pdf_include_exudates_overlay": True,
        "pdf_include_haemorrhages_mask": False,
        "pdf_include_haemorrhages_overlay": True,
        "pdf_include_disclaimer": True,
    }
    for key, value in pdf_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    for eye in ["right", "left"]:
        if eye in eye_results:
            pred_idx = int(eye_results[eye]["pred_idx"])
            if f"{eye}_analysis_target_class" not in st.session_state:
                st.session_state[f"{eye}_analysis_target_class"] = pred_idx
            if f"{eye}_analysis_alpha" not in st.session_state:
                st.session_state[f"{eye}_analysis_alpha"] = 0.45
            if f"{eye}_analysis_cam_method" not in st.session_state:
                st.session_state[f"{eye}_analysis_cam_method"] = "GradCAM++"

            st.session_state[f"{eye}_analysis_target_layer"] = "(auto)"


def get_available_eyes_and_primary(eye_results: Dict) -> Tuple[List[str], str, str]:
    available_eyes = [eye for eye in ["right", "left"] if eye in eye_results]
    analysis_input_mode = "pair" if len(available_eyes) == 2 else "single"

    primary_eye = st.session_state.get("primary_eye", available_eyes[0])
    if primary_eye not in available_eyes:
        primary_eye = available_eyes[0]

    return available_eyes, primary_eye, analysis_input_mode


def validate_layer_names(cls_model, available_eyes: List[str]) -> None:
    convs = cls_utils.list_conv2d_layers(cls_model)
    layer_names = [n for n, _ in convs] if convs else []

    for eye in available_eyes:
        layer_key = f"{eye}_analysis_target_layer"
        if st.session_state[layer_key] != "(auto)" and st.session_state[layer_key] not in layer_names:
            st.session_state[layer_key] = "(auto)"


def compute_outputs_per_eye(
    eye_results: Dict,
    available_eyes: List[str],
    cls_model,
    cls_device,
    seg_model,
    seg_device,
) -> Dict:
    computed = {}

    for eye in available_eyes:
        eye_result = eye_results[eye]
        pred_idx = int(eye_result["pred_idx"])
        pred_name = eye_result["pred_name"]
        probs = np.array(eye_result["probs"], dtype=np.float32)
        raw_rgb = eye_result["raw_rgb"]
        input_tensor = eye_result["input_tensor"].to(cls_device)
        uploaded_name = eye_result.get("uploaded_name", f"{eye}_eye")

        target_class = int(st.session_state[f"{eye}_analysis_target_class"])
        alpha = float(st.session_state[f"{eye}_analysis_alpha"])

        cam_method_ui = st.session_state[f"{eye}_analysis_cam_method"]
        cam_method = "gradcampp" if cam_method_ui == "GradCAM++" else "scorecam"

        layer_name, layer_module = cls_utils.choose_target_layer(
            cls_model,
            preferred_name=None,
        )

        cam_mask = cls_utils.run_cam(
            model=cls_model,
            input_tensor=input_tensor,
            target_class=target_class,
            target_layer_module=layer_module,
            method=cam_method,
        )

        cam_overlay_rgb, heatmap_rgb = cls_utils.overlay_heatmap(
            raw_rgb,
            cam_mask,
            alpha=alpha,
        )

        pil_for_seg = Image.fromarray(raw_rgb)
        seg_result = seg_utils.predict_segmentation(seg_model, seg_device, pil_for_seg)

        computed[eye] = {
            "pred_idx": pred_idx,
            "pred_name": pred_name,
            "probs": probs,
            "raw_rgb": raw_rgb,
            "input_tensor": input_tensor,
            "uploaded_name": uploaded_name,
            "target_class": target_class,
            "alpha": alpha,
            "picked_layer": "(auto)",
            "layer_name": layer_name,
            "cam_method_ui": cam_method_ui,
            "heatmap_rgb": heatmap_rgb,
            "cam_overlay_rgb": cam_overlay_rgb,
            "seg": seg_result,
        }

    return computed