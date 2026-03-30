import streamlit as st

from saved_cases_store import list_saved_cases, load_case_bundle, apply_case_to_session


st.set_page_config(page_title="Saved Cases", layout="wide")

st.markdown(
    """
    <style>
      [data-testid="stSidebar"] {display: none;}
      [data-testid="stSidebarNav"] {display: none;}

      .case-card {
          padding: 1rem 1.1rem;
          border: 1px solid rgba(255,255,255,0.08);
          border-radius: 16px;
          background: rgba(255,255,255,0.02);
          margin-bottom: 0.85rem;
      }

      .case-title {
          font-size: 1.05rem;
          font-weight: 700;
          margin-bottom: 0.35rem;
      }

      .case-meta {
          color: #9aa4b2;
          font-size: 0.92rem;
          margin-bottom: 0.25rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Saved Cases")
st.caption("Last 5 saved analysis results. Open any case to revisit it in the Analysis page.")

top_left, top_right = st.columns([1, 4])
with top_left:
    if st.button("⬅ Back to Dashboard", use_container_width=True):
        st.switch_page("pages/1_Dashboard.py")

st.divider()

saved_cases = list_saved_cases(limit=5)

if not saved_cases:
    st.info("No saved cases yet.")
    st.stop()

for idx, case in enumerate(saved_cases):
    left, right = st.columns([4.6, 1.2], gap="large")

    with left:
        st.markdown(
            f"""
            <div class="case-card">
                <div class="case-title">Case {idx + 1}</div>
                <div class="case-meta"><b>Saved:</b> {case.get("saved_at", "-")}</div>
                <div class="case-meta"><b>Mode:</b> {case.get("analysis_input_mode", "-").title()}</div>
                <div><b>Summary:</b> {case.get("summary", "-")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        if st.button("Open", key=f"open_case_{case['case_id']}", use_container_width=True):
            loaded = load_case_bundle(case["case_id"])
            apply_case_to_session(st.session_state, loaded)
            st.switch_page("pages/2_Analysis.py")