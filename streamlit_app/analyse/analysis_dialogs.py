from datetime import datetime

import streamlit as st

import utils_classification as cls_utils
from pdf_report import build_pdf_report
from saved_cases_store import save_case_bundle


def toast_success(message: str):
    st.toast(message, icon="✅")


def toast_warning(message: str):
    st.toast(message, icon="⚠️")


def toast_error(message: str):
    st.toast(message, icon="❌")


def toast_info(message: str):
    st.toast(message, icon="ℹ️")


def save_current_case(analysis_input_mode: str, primary_eye: str):
    patient_id = st.session_state.save_patient_id.strip()
    if not patient_id:
        raise ValueError("Patient ID is required.")

    saved = save_case_bundle(
        eye_results=st.session_state.eye_results,
        analysis_input_mode=analysis_input_mode,
        primary_eye=primary_eye,
        patient_id=patient_id,
        patient_name=st.session_state.save_patient_name.strip(),
        age=st.session_state.save_patient_age.strip(),
        gender=st.session_state.save_patient_gender.strip(),
        notes=st.session_state.save_patient_notes.strip(),
        max_cases=5,
    )

    st.session_state.saved_patient_id = st.session_state.save_patient_id
    st.session_state.saved_patient_name = st.session_state.save_patient_name
    st.session_state.saved_patient_age = st.session_state.save_patient_age
    st.session_state.saved_patient_gender = st.session_state.save_patient_gender
    st.session_state.saved_patient_notes = st.session_state.save_patient_notes

    return saved


def current_pdf_options():
    return {
        "report_title": st.session_state.pdf_report_title,
        "patient_name": st.session_state.pdf_patient_name,
        "patient_id": st.session_state.pdf_patient_id,
        "clinician_name": st.session_state.pdf_clinician_name,
        "institution_name": st.session_state.pdf_institution_name,
        "notes": st.session_state.pdf_notes,
        "include_report_details": st.session_state.pdf_include_report_details,
        "include_prediction_summary": st.session_state.pdf_include_prediction_summary,
        "include_probabilities": st.session_state.pdf_include_probabilities,
        "include_original_image": st.session_state.pdf_include_original_image,
        "include_gradcam_heatmap": st.session_state.pdf_include_gradcam_heatmap,
        "include_gradcam_overlay": st.session_state.pdf_include_gradcam_overlay,
        "include_exudates_mask": st.session_state.pdf_include_exudates_mask,
        "include_exudates_overlay": st.session_state.pdf_include_exudates_overlay,
        "include_haemorrhages_mask": st.session_state.pdf_include_haemorrhages_mask,
        "include_haemorrhages_overlay": st.session_state.pdf_include_haemorrhages_overlay,
        "include_disclaimer": st.session_state.pdf_include_disclaimer,
    }


def render_save_case_dialog(analysis_input_mode: str, primary_eye: str):
    @st.dialog("Save Case")
    def _dialog():
        st.write("Enter patient details before saving this case. Patient ID is required.")

        col1, col2 = st.columns(2)
        with col1:
            st.text_input("Patient ID *", key="save_patient_id")
            st.text_input("Patient Name", key="save_patient_name")
            st.text_input("Age", key="save_patient_age")
        with col2:
            st.selectbox(
                "Gender",
                options=["", "Male", "Female", "Other"],
                key="save_patient_gender",
            )
            st.text_area("Notes", key="save_patient_notes", height=110)

        st.markdown("---")
        b1, b2 = st.columns(2)

        with b1:
            if st.button("Confirm Save", key="confirm_save_case_btn", use_container_width=True):
                try:
                    saved = save_current_case(analysis_input_mode, primary_eye)
                    st.session_state.show_save_case_dialog = False
                    st.session_state.analysis_page_toast = {
                        "level": "success",
                        "message": f"Saved. Patient ID: {saved['patient_id']} | {saved['summary']}",
                    }
                    st.rerun()
                except Exception as e:
                    toast_error(str(e))

        with b2:
            if st.button("Close", key="close_save_case_btn", use_container_width=True):
                st.session_state.show_save_case_dialog = False
                st.rerun()

    _dialog()


def render_pdf_report_dialog(
    computed: dict,
    available_eyes: list,
    analysis_input_mode: str,
):
    @st.dialog("PDF Report Options")
    def _dialog():
        st.write("Choose what should be included in the report before generating the PDF.")

        st.text_input("Report title", key="pdf_report_title")
        c1, c2 = st.columns(2)
        with c1:
            st.text_input("Patient name", key="pdf_patient_name")
            st.text_input("Patient ID", key="pdf_patient_id")
        with c2:
            st.text_input("Clinician / Examiner", key="pdf_clinician_name")
            st.text_input("Institution", key="pdf_institution_name")

        st.text_area("Notes", key="pdf_notes", height=80)

        st.markdown("### Include sections")
        s1, s2 = st.columns(2)
        with s1:
            st.checkbox("Report details", key="pdf_include_report_details")
            st.checkbox("Prediction summary", key="pdf_include_prediction_summary")
            st.checkbox("Class probabilities", key="pdf_include_probabilities")
            st.checkbox("Disclaimer", key="pdf_include_disclaimer")
        with s2:
            st.checkbox("Original image", key="pdf_include_original_image")
            st.checkbox("Grad-CAM heatmap", key="pdf_include_gradcam_heatmap")
            st.checkbox("Grad-CAM overlay", key="pdf_include_gradcam_overlay")
            st.checkbox("Exudates mask", key="pdf_include_exudates_mask")
            st.checkbox("Exudates overlay", key="pdf_include_exudates_overlay")
            st.checkbox("Haemorrhages mask", key="pdf_include_haemorrhages_mask")
            st.checkbox("Haemorrhages overlay", key="pdf_include_haemorrhages_overlay")

        st.markdown("---")
        btn1, btn2 = st.columns(2)

        with btn1:
            if st.button("Generate PDF", key="generate_pdf_btn", use_container_width=True):
                options = current_pdf_options()
                pdf_bytes = build_pdf_report(
                    computed=computed,
                    available_eyes=available_eyes,
                    analysis_input_mode=analysis_input_mode,
                    class_names=cls_utils.CLASS_NAMES,
                    selected_view_mode=st.session_state.analysis_view_mode,
                    options=options,
                )
                file_name = f"dr_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                st.session_state.generated_pdf_bytes = pdf_bytes
                st.session_state.generated_pdf_name = file_name
                toast_success("PDF generated. Download it below.")

        with btn2:
            if st.button("Close", key="close_pdf_dialog_btn", use_container_width=True):
                st.session_state.show_pdf_dialog = False
                st.rerun()

        if st.session_state.generated_pdf_bytes is not None:
            st.download_button(
                "Download PDF Report",
                data=st.session_state.generated_pdf_bytes,
                file_name=st.session_state.generated_pdf_name or "dr_report.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="download_pdf_report_btn",
            )

    _dialog()