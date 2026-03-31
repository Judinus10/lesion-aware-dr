import time
import streamlit as st

from saved_cases_store import (
    apply_case_to_session,
    delete_case_bundle,
    list_saved_cases,
    load_case_bundle,
    update_case_details,
)


st.set_page_config(page_title="Saved Cases", layout="wide")


# ---------------- TOAST HELPERS ----------------
def toast_success(message: str):
    st.toast(message, icon="✅")


def toast_warning(message: str):
    st.toast(message, icon="⚠️")


def toast_error(message: str):
    st.toast(message, icon="❌")


def toast_info(message: str):
    st.toast(message, icon="ℹ️")


# ---------------- LOADER ----------------
def show_loader(message: str = "Loading..."):
    st.markdown(
        f"""
        <style>
        .saved-cases-loader-overlay {{
            position: fixed;
            inset: 0;
            background: rgba(2, 6, 23, 0.82);
            backdrop-filter: blur(3px);
            z-index: 999999;
            display: flex;
            align-items: center;
            justify-content: center;
        }}

        .saved-cases-loader-box {{
            min-width: 280px;
            max-width: 420px;
            background: rgba(15, 23, 42, 0.96);
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 20px;
            padding: 1.4rem 1.5rem 1.2rem 1.5rem;
            box-shadow: 0 18px 60px rgba(0,0,0,0.35);
            text-align: center;
        }}

        .saved-cases-loader-spinner {{
            width: 52px;
            height: 52px;
            margin: 0 auto 0.9rem auto;
            border-radius: 50%;
            border: 4px solid rgba(255,255,255,0.12);
            border-top: 4px solid #4f8cff;
            animation: savedCasesSpin 0.9s linear infinite;
        }}

        .saved-cases-loader-text {{
            color: #e5e7eb;
            font-size: 1rem;
            font-weight: 600;
            letter-spacing: 0.01em;
        }}

        @keyframes savedCasesSpin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        </style>

        <div class="saved-cases-loader-overlay">
            <div class="saved-cases-loader-box">
                <div class="saved-cases-loader-spinner"></div>
                <div class="saved-cases-loader-text">{message}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------- EDIT DIALOG ----------------
def render_edit_case_dialog(case: dict, idx: int):
    case_id = case.get("case_id", "")
    edit_key_prefix = f"edit_case_{case_id}"

    @st.dialog("Edit Patient Details")
    def _dialog():
        st.write("Edit patient details only. Eye images and prediction results cannot be changed here.")

        patient_id_default = case.get("patient_id", "") or ""
        patient_name_default = case.get("patient_name", "") or ""
        age_default = case.get("age", "") or ""
        gender_default = case.get("gender", "") or ""
        notes_default = case.get("notes", "") or ""

        c1, c2 = st.columns(2)
        with c1:
            patient_id = st.text_input("Patient ID *", value=patient_id_default, key=f"{edit_key_prefix}_patient_id")
            patient_name = st.text_input("Patient Name", value=patient_name_default, key=f"{edit_key_prefix}_patient_name")
            age = st.text_input("Age", value=age_default, key=f"{edit_key_prefix}_age")
        with c2:
            gender_options = ["", "Male", "Female", "Other"]
            try:
                gender_index = gender_options.index(gender_default)
            except ValueError:
                gender_index = 0

            gender = st.selectbox(
                "Gender",
                options=gender_options,
                index=gender_index,
                key=f"{edit_key_prefix}_gender",
            )
            notes = st.text_area("Notes", value=notes_default, height=120, key=f"{edit_key_prefix}_notes")

        st.markdown("---")
        b1, b2 = st.columns(2)

        with b1:
            if st.button("Save Changes", key=f"{edit_key_prefix}_save_btn", use_container_width=True):
                try:
                    if not str(patient_id).strip():
                        toast_warning("Patient ID is required.")
                        return

                    show_loader("Updating patient details...")
                    update_case_details(
                        case_id,
                        patient_id=str(patient_id).strip(),
                        patient_name=str(patient_name).strip(),
                        age=str(age).strip(),
                        gender=str(gender).strip(),
                        notes=str(notes).strip(),
                    )
                    st.session_state.saved_cases_page_toast = {
                        "level": "success",
                        "message": f"Updated Case {idx + 1} details.",
                    }
                    time.sleep(0.35)
                    st.rerun()
                except Exception as e:
                    toast_error(f"Failed to update patient details: {e}")

        with b2:
            if st.button("Close", key=f"{edit_key_prefix}_close_btn", use_container_width=True):
                st.rerun()

    _dialog()


# ---------------- CARD HTML ----------------
def render_saved_case_card(case: dict, idx: int) -> str:
    patient_id = case.get("patient_id") or "-"
    patient_name = case.get("patient_name") or "-"
    age = case.get("age") or "-"
    saved_at = case.get("saved_at") or "-"
    gender = case.get("gender") or "-"
    mode = str(case.get("analysis_input_mode", "-")).title()
    summary = case.get("summary") or "-"

    return f"""
    <div class="saved-card">
        <div class="saved-grid">
            <div class="saved-col">
                <div class="saved-title">Case {idx + 1}</div>
                <div class="saved-row"><span class="saved-label">Patient ID:</span> {patient_id}</div>
                <div class="saved-row"><span class="saved-label">Patient Name:</span> {patient_name}</div>
                <div class="saved-row"><span class="saved-label">Age:</span> {age}</div>
            </div>
            <div class="saved-col">
                <div class="saved-row"><span class="saved-label">Saved:</span> {saved_at}</div>
                <div class="saved-row"><span class="saved-label">Gender:</span> {gender}</div>
                <div class="saved-row"><span class="saved-label">Mode:</span> {mode}</div>
                <div class="saved-summary"><span class="saved-label">Summary:</span> {summary}</div>
            </div>
        </div>
    </div>
    """


# ---------------- PAGE CSS ----------------
st.markdown(
    """
    <style>
      [data-testid="stSidebar"] {display: none;}
      [data-testid="stSidebarNav"] {display: none;}

      .block-container {
          padding-top: 1rem;
          padding-bottom: 2rem;
          max-width: 1280px;
      }

      .saved-top-space {
          margin-top: 0.7rem;
          margin-bottom: 1.4rem;
      }

      .saved-header {
          text-align: center;
          margin-top: 0.2rem;
          margin-bottom: 1.4rem;
      }

      .saved-header-title {
          font-size: 3rem;
          font-weight: 800;
          line-height: 1.1;
          color: #f8fafc;
          margin-bottom: 0.45rem;
      }

      .saved-header-subtitle {
          font-size: 1rem;
          color: #94a3b8;
          margin-bottom: 0;
      }

      .saved-card {
          padding: 1rem 1.15rem;
          border: 1px solid rgba(255,255,255,0.08);
          border-radius: 18px;
          background: rgba(255,255,255,0.02);
      }

      .saved-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 1.4rem;
          align-items: start;
      }

      .saved-col {
          display: flex;
          flex-direction: column;
          gap: 0.42rem;
      }

      .saved-title {
          font-size: 1.1rem;
          font-weight: 700;
          color: #f8fafc;
          margin-bottom: 0.3rem;
      }

      .saved-row {
          font-size: 0.97rem;
          color: #dbe4ee;
          line-height: 1.5;
          word-break: break-word;
      }

      .saved-label {
          color: #94a3b8;
          font-weight: 600;
      }

      .saved-summary {
          font-size: 0.97rem;
          color: #f8fafc;
          line-height: 1.55;
          word-break: break-word;
      }

      .saved-actions-wrap {
          display: flex;
          flex-direction: column;
          justify-content: center;
          gap: 0.55rem;
          height: 100%;
          min-height: 100%;
      }

      .saved-divider {
          height: 1px;
          background: rgba(255,255,255,0.06);
          margin: 1rem 0 1.25rem 0;
      }

      .empty-state-box {
          padding: 1.2rem 1rem;
          border: 1px solid rgba(255,255,255,0.08);
          border-radius: 16px;
          background: rgba(255,255,255,0.02);
          text-align: center;
          color: #94a3b8;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------- TOAST FROM SESSION ----------------
if "saved_cases_page_toast" in st.session_state:
    toast = st.session_state.pop("saved_cases_page_toast")
    level = toast.get("level", "info")
    message = toast.get("message", "")

    if message:
        if level == "success":
            toast_success(message)
        elif level == "warning":
            toast_warning(message)
        elif level == "error":
            toast_error(message)
        else:
            toast_info(message)


st.markdown('<div class="saved-top-space"></div>', unsafe_allow_html=True)

back_col_1, back_col_2, back_col_3 = st.columns([1.15, 2.8, 3.05])

with back_col_1:
    if st.button("⬅ Back to Dashboard", use_container_width=True):
        show_loader("Returning to Dashboard...")
        time.sleep(0.45)
        st.switch_page("pages/1_Dashboard.py")


st.markdown(
    """
    <div class="saved-header">
        <div class="saved-header-title">Saved Cases</div>
        <div class="saved-header-subtitle">
            Last 5 saved analysis results. Open any case to revisit it in the Analysis page.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()


try:
    saved_cases = list_saved_cases(limit=5)
except Exception as e:
    saved_cases = []
    toast_error(f"Failed to load saved cases: {e}")

if not saved_cases:
    toast_warning("No saved cases found.")
    st.markdown(
        """
        <div class="empty-state-box">
            No saved cases yet.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


for idx, case in enumerate(saved_cases):
    case_id = case.get("case_id", f"case_{idx + 1}")

    row_left, row_right = st.columns([5.1, 1.35], vertical_alignment="center")

    with row_left:
        st.markdown(render_saved_case_card(case, idx), unsafe_allow_html=True)

    with row_right:
        st.markdown('<div class="saved-actions-wrap">', unsafe_allow_html=True)

        if st.button("Open", key=f"open_case_{case_id}", use_container_width=True):
            try:
                loaded = load_case_bundle(case_id)

                if not loaded:
                    toast_error("Selected case could not be loaded.")
                else:
                    show_loader("Opening saved case...")
                    st.session_state["pending_loaded_case"] = loaded
                    st.session_state["analysis_page_toast"] = {
                        "level": "success",
                        "message": f"Opened Case {idx + 1}.",
                    }
                    time.sleep(0.45)
                    st.switch_page("pages/2_Analysis.py")
            except Exception as e:
                toast_error(f"Something went wrong while opening the saved case: {e}")

        if st.button("Edit", key=f"edit_case_{case_id}", use_container_width=True):
            render_edit_case_dialog(case, idx)

        if st.button("Delete", key=f"delete_case_{case_id}", use_container_width=True):
            try:
                show_loader("Deleting saved case...")
                deleted = delete_case_bundle(case_id)

                if deleted:
                    st.session_state.saved_cases_page_toast = {
                        "level": "success",
                        "message": f"Deleted Case {idx + 1}.",
                    }
                else:
                    st.session_state.saved_cases_page_toast = {
                        "level": "warning",
                        "message": "Saved case folder was not found.",
                    }

                time.sleep(0.35)
                st.rerun()
            except Exception as e:
                toast_error(f"Failed to delete saved case: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    if idx < len(saved_cases) - 1:
        st.markdown('<div class="saved-divider"></div>', unsafe_allow_html=True)