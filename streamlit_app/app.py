import streamlit as st

st.set_page_config(page_title="Lesion-Aware DR System", layout="wide")

# Hide Streamlit default sidebar navigation (we are doing our own flow)
st.markdown(
    """
    <style>
      [data-testid="stSidebar"] {display: none;}
      [data-testid="stSidebarNav"] {display: none;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Always go to Dashboard as the "home"
if "booted" not in st.session_state:
    st.session_state.booted = True
    st.switch_page("pages/1_Dashboard.py")

st.title("Lesion-Aware DR Grading System")
st.info("Redirecting to Dashboard…")