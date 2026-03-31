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


def inject_save_dialog_error_styles():
    st.markdown(
        """
        <style>
        .save-form-error-text {
            color: #f87171;
            font-size: 0.84rem;
            font-weight: 600;
            margin-top: -0.35rem;
            margin-bottom: 0.25rem;
        }

        .save-form-warning-box {
            background: rgba(127, 29, 29, 0.18);
            border: 1px solid rgba(248, 113, 113, 0.55);
            color: #fecaca;
            border-radius: 12px;
            padding: 0.8rem 0.95rem;
            margin-bottom: 0.8rem;
            line-height: 1.5;
            font-size: 0.95rem;
            font-weight: 500;
        }

        .save-field-wrap {
            margin-bottom: 0.1rem;
        }

        div[data-testid="stTextInput"] input {
            border-radius: 10px;
        }

        .missing-patient-id div[data-testid="stTextInput"] input,
        .missing-patient-name div[data-testid="stTextInput"] input,
        .missing-age div[data-testid="stTextInput"] input {
            border: 1.5px solid #ef4444 !important;
            box-shadow: 0 0 0 1px #ef4444 !important;
        }

        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            border-radius: 10px;
        }

        .missing-gender div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            border: 1.5px solid #ef4444 !important;
            box-shadow: 0 0 0 1px #ef4444 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def normalize_text(value) -> str:
    return str(value or "").strip()


def validate_save_case_inputs() -> dict:
    patient_id = normalize_text(st.session_state.get("save_patient_id", ""))
    patient_name = normalize_text(st.session_state.get("save_patient_name", ""))
    age = normalize_text(st.session_state.get("save_patient_age", ""))
    gender = normalize_text(st.session_state.get("save_patient_gender", ""))

    errors = {
        "patient_id": not bool(patient_id),
        "patient_name": not bool(patient_name),
        "age": not bool(age),
        "gender": not bool(gender),
    }

    return {
        "patient_id": patient_id,
        "patient_name": patient_name,
        "age": age,
        "gender": gender,
        "notes": normalize_text(st.session_state.get("save_patient_notes", "")),
        "errors": errors,
        "missing_fields": [
            label
            for key, label in [
                ("patient_id", "Patient ID"),
                ("patient_name", "Patient Name"),
                ("age", "Age"),
                ("gender", "Gender"),
            ]
            if errors[key]
        ],
    }


def save_current_case(analysis_input_mode: str, primary_eye: str):
    data = validate_save_case_inputs()

    if data["missing_fields"]:
        missing_text = ", ".join(data["missing_fields"])
        raise ValueError(f"Please enter the missing data: {missing_text}.")

    saved = save_case_bundle(
        eye_results=st.session_state.eye_results,
        analysis_input_mode=analysis_input_mode,
        primary_eye=primary_eye,
        patient_id=data["patient_id"],
        patient_name=data["patient_name"],
        age=data["age"],
        gender=data["gender"],
        notes=data["notes"],
        max_cases=5,
    )

    st.session_state.saved_patient_id = data["patient_id"]
    st.session_state.saved_patient_name = data["patient_name"]
    st.session_state.saved_patient_age = data["age"]
    st.session_state.saved_patient_gender = data["gender"]
    st.session_state.saved_patient_notes = data["notes"]

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
        inject_save_dialog_error_styles()

        if "save_case_missing_fields" not in st.session_state:
            st.session_state.save_case_missing_fields = []

        st.write("Enter patient details before saving this case. Patient ID is required.")

        missing_fields = st.session_state.get("save_case_missing_fields", [])

        if missing_fields:
            st.markdown(
                f"""
                <div class="save-form-warning-box">
                    Missing required data: {", ".join(missing_fields)}.<br>
                    Fill the highlighted fields and try again.
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ---------- ROW 1 : Patient ID | Gender ----------
        row1_col1, row1_col2 = st.columns([1, 1], gap="small")

        with row1_col1:
            patient_id_class = "missing-patient-id" if "Patient ID" in missing_fields else ""
            st.markdown(f'<div class="{patient_id_class} save-field-wrap">', unsafe_allow_html=True)
            st.text_input("Patient ID *", key="save_patient_id")
            st.markdown("</div>", unsafe_allow_html=True)
            if "Patient ID" in missing_fields:
                st.markdown(
                    '<div class="save-form-error-text">Patient ID is required.</div>',
                    unsafe_allow_html=True,
                )

        with row1_col2:
            gender_class = "missing-gender" if "Gender" in missing_fields else ""
            st.markdown(f'<div class="{gender_class} save-field-wrap">', unsafe_allow_html=True)
            st.selectbox(
                "Gender",
                options=["", "Male", "Female", "Other"],
                key="save_patient_gender",
            )
            st.markdown("</div>", unsafe_allow_html=True)
            if "Gender" in missing_fields:
                st.markdown(
                    '<div class="save-form-error-text">Gender is required.</div>',
                    unsafe_allow_html=True,
                )

        # ---------- ROW 2 : Patient Name | Age ----------
        row2_col1, row2_col2 = st.columns([1, 1], gap="small")

        with row2_col1:
            patient_name_class = "missing-patient-name" if "Patient Name" in missing_fields else ""
            st.markdown(f'<div class="{patient_name_class} save-field-wrap">', unsafe_allow_html=True)
            st.text_input("Patient Name", key="save_patient_name")
            st.markdown("</div>", unsafe_allow_html=True)
            if "Patient Name" in missing_fields:
                st.markdown(
                    '<div class="save-form-error-text">Patient Name is required.</div>',
                    unsafe_allow_html=True,
                )

        with row2_col2:
            age_class = "missing-age" if "Age" in missing_fields else ""
            st.markdown(f'<div class="{age_class} save-field-wrap">', unsafe_allow_html=True)
            st.text_input("Age", key="save_patient_age")
            st.markdown("</div>", unsafe_allow_html=True)
            if "Age" in missing_fields:
                st.markdown(
                    '<div class="save-form-error-text">Age is required.</div>',
                    unsafe_allow_html=True,
                )

        # ---------- ROW 3 : Notes ----------
        st.text_area("Notes", key="save_patient_notes", height=110)

        st.markdown("---")
        b1, b2 = st.columns(2)

        with b1:
            if st.button("Confirm Save", key="confirm_save_case_btn", use_container_width=True):
                try:
                    if "eye_results" not in st.session_state or not st.session_state.eye_results:
                        toast_error("No analysis result found to save.")
                        return

                    data = validate_save_case_inputs()

                    if data["missing_fields"]:
                        st.session_state.save_case_missing_fields = data["missing_fields"]
                        toast_warning("Enter the missing required data before saving.")
                        st.rerun()

                    saved = save_current_case(analysis_input_mode, primary_eye)
                    st.session_state.save_case_missing_fields = []
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
                st.session_state.save_case_missing_fields = []
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
        # ---------- DEFAULT PDF CHECKBOX STATES ----------
        st.session_state.setdefault("pdf_include_report_details", True)
        st.session_state.setdefault("pdf_include_prediction_summary", True)
        st.session_state.setdefault("pdf_include_probabilities", True)
        st.session_state.setdefault("pdf_include_disclaimer", True)

        st.session_state.setdefault("pdf_include_original_image", False)
        st.session_state.setdefault("pdf_include_gradcam_heatmap", False)
        st.session_state.setdefault("pdf_include_gradcam_overlay", False)
        st.session_state.setdefault("pdf_include_exudates_mask", False)
        st.session_state.setdefault("pdf_include_exudates_overlay", False)
        st.session_state.setdefault("pdf_include_haemorrhages_mask", False)
        st.session_state.setdefault("pdf_include_haemorrhages_overlay", False)

        st.write("Choose what should be included in the report before generating the PDF.")

        st.text_input("Report title", key="pdf_report_title")

        row1_col1, row1_col2 = st.columns([1, 1], gap="small")
        with row1_col1:
            st.text_input("Patient name", key="pdf_patient_name")
        with row1_col2:
            st.text_input("Clinician / Examiner", key="pdf_clinician_name")

        row2_col1, row2_col2 = st.columns([1, 1], gap="small")
        with row2_col1:
            st.text_input("Patient ID", key="pdf_patient_id")
        with row2_col2:
            st.text_input("Institution", key="pdf_institution_name")

        st.text_area("Notes", key="pdf_notes", height=100)

        st.markdown("### Include sections")
        s1, s2 = st.columns([1, 1], gap="large")
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