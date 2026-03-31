import os
import sys

import streamlit as st


CURRENT_DIR = os.path.dirname(__file__)
STREAMLIT_APP_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if STREAMLIT_APP_DIR not in sys.path:
    sys.path.insert(0, STREAMLIT_APP_DIR)

from analyse.analysis_compute import (
    compute_outputs_per_eye,
    ensure_analysis_session_defaults,
    get_available_eyes_and_primary,
    get_cached_cls_model,
    get_cached_seg_model,
    prepare_eye_results_from_legacy,
    validate_layer_names,
)
from analyse.analysis_components_saved_cases import render_saved_case_card
from analyse.analysis_components_summary import render_eye_prediction_summary_two_col
from analyse.analysis_components_views import (
    render_ex_pair,
    render_ex_single,
    render_explainability_pair_vertical,
    render_explainability_single,
    render_eye_controls,
    render_he_pair,
    render_he_single,
    render_original_pair,
    render_original_single,
)
from analyse.analysis_dialogs import render_pdf_report_dialog, render_save_case_dialog
from analyse.analysis_styles import apply_analysis_styles, clear_fixed_loader, show_fixed_loader
from saved_cases_store import apply_case_to_session, list_saved_cases, load_case_bundle


st.set_page_config(page_title="Analysis", layout="wide")
apply_analysis_styles()


def open_saved_case(patient_id: str, case_id: str):
    loaded = load_case_bundle(patient_id, case_id)
    apply_case_to_session(st.session_state, loaded)


analysis_input_mode, eye_results = prepare_eye_results_from_legacy()

if not eye_results:
    st.warning("No prediction found. Go back and run prediction first.")
    if st.button("Go to Dashboard", key="analysis_go_dashboard_empty"):
        st.switch_page("pages/1_Dashboard.py")
    st.stop()

available_eyes, primary_eye, analysis_input_mode = get_available_eyes_and_primary(eye_results)

if not available_eyes:
    st.error("Prediction data is invalid.")
    st.stop()

cls_model, cls_device = get_cached_cls_model()
seg_model, seg_device = get_cached_seg_model()

ensure_analysis_session_defaults(eye_results)
validate_layer_names(cls_model, available_eyes)

loader = show_fixed_loader("Preparing multi-model analysis...")

computed = compute_outputs_per_eye(
    eye_results=eye_results,
    available_eyes=available_eyes,
    cls_model=cls_model,
    cls_device=cls_device,
    seg_model=seg_model,
    seg_device=seg_device,
)

clear_fixed_loader(loader)

top_left, top_right = st.columns([3, 1], gap="large")

with top_left:
    if st.button("Back to Dashboard", key="analysis_back_dashboard_btn"):
        st.switch_page("pages/1_Dashboard.py")

with top_right:
    b1, b2 = st.columns(2, gap="small")

    with b1:
        if st.button("Save Case", key="analysis_save_case_btn", use_container_width=True):
            st.session_state.show_save_case_dialog = True

    with b2:
        if st.button("PDF Report", key="analysis_pdf_report_btn", use_container_width=True):
            st.session_state.show_pdf_dialog = True
            st.session_state.generated_pdf_bytes = None
            st.session_state.generated_pdf_name = None

st.title("Prediction Result")
st.divider()

if analysis_input_mode == "pair":
    if "right" in computed:
        render_eye_prediction_summary_two_col("right", computed["right"])
        st.markdown("---")

    if "left" in computed:
        render_eye_prediction_summary_two_col("left", computed["left"])
else:
    render_eye_prediction_summary_two_col(primary_eye, computed[primary_eye])

st.divider()

st.subheader("Why did the model predict this?")

_, switch_mid, _ = st.columns([1, 6.8, 1])

with switch_mid:
    st.markdown(
        '<div class="switch-note">Switch between classifier explainability and segmentation-based lesion views.</div>',
        unsafe_allow_html=True,
    )

    view_mode = st.radio(
        "View Mode",
        options=[
            "Original",
            "Explainability",
            "Exudates (EX)",
            "Haemorrhages (HE)",
        ],
        key="analysis_view_mode",
        horizontal=True,
        label_visibility="collapsed",
    )

st.markdown("")

show_side_controls = view_mode == "Explainability" and analysis_input_mode == "single"

if show_side_controls:
    left, right = st.columns([2.0, 0.95], gap="large")
else:
    left = st.container()
    right = None

with left:
    if analysis_input_mode == "single":
        c = computed[primary_eye]

        if view_mode == "Original":
            render_original_single(c, primary_eye)
        elif view_mode == "Explainability":
            render_explainability_single(c, primary_eye)
        elif view_mode == "Exudates (EX)":
            render_ex_single(c, primary_eye)
        elif view_mode == "Haemorrhages (HE)":
            render_he_single(c, primary_eye)
    else:
        if view_mode == "Original":
            render_original_pair(computed)
        elif view_mode == "Explainability":
            render_explainability_pair_vertical(computed)
        elif view_mode == "Exudates (EX)":
            render_ex_pair(computed)
        elif view_mode == "Haemorrhages (HE)":
            render_he_pair(computed)

if show_side_controls and right is not None:
    with right:
        st.subheader("Explainability Controls")
        render_eye_controls(primary_eye)

st.markdown("---")
st.subheader("Saved Cases")
st.caption("Last 5 persistent cases. These remain available even after you close and reopen the app.")

saved_cases = list_saved_cases(limit=5)

if saved_cases:
    for idx, case in enumerate(saved_cases):
        col_a, col_b = st.columns([5.5, 1.15], gap="large")

        with col_a:
            st.markdown(render_saved_case_card(case, idx), unsafe_allow_html=True)

        with col_b:
            st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
            if st.button(
                "Open",
                key=f"analysis_open_saved_{case['patient_id']}_{case['case_id']}",
                use_container_width=True,
            ):
                open_saved_case(case["patient_id"], case["case_id"])
                st.rerun()
else:
    st.caption("No saved cases yet.")

st.divider()
st.info("Research demo only — not medical advice. Explainability and lesion masks are support outputs, not clinical truth.")

if st.session_state.show_pdf_dialog:
    render_pdf_report_dialog(
        computed=computed,
        available_eyes=available_eyes,
        analysis_input_mode=analysis_input_mode,
    )

if st.session_state.show_save_case_dialog:
    render_save_case_dialog(
        analysis_input_mode=analysis_input_mode,
        primary_eye=primary_eye,
    )