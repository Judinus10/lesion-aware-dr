import streamlit as st


ANALYSIS_CSS = """
<style>
  [data-testid="stSidebar"] {display: none;}
  [data-testid="stSidebarNav"] {display: none;}

  .switch-note {
      color: #9aa4b2;
      font-size: 0.95rem;
      margin-bottom: 0.55rem;
      text-align: center;
  }

  .pred-card {
      padding: 0.95rem 1rem;
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 16px;
      background: rgba(255,255,255,0.02);
  }

  .saved-card {
      padding: 0.85rem 1rem;
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 14px;
      background: rgba(255,255,255,0.02);
      margin-bottom: 0.65rem;
  }

  .saved-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1rem 2rem;
  }

  .saved-col {
      line-height: 1.65;
  }

  .saved-label {
      font-weight: 700;
  }

  div[data-testid="stRadio"] {
      display: flex !important;
      justify-content: center !important;
      align-items: center !important;
      width: 100% !important;
      margin-top: 0.25rem;
      margin-bottom: 1.25rem;
  }

  div[data-testid="stRadio"] > div {
      display: flex !important;
      justify-content: center !important;
      width: 100% !important;
  }

  div[role="radiogroup"] {
      display: inline-flex !important;
      justify-content: center !important;
      align-items: stretch !important;
      flex-wrap: nowrap;
      gap: 0 !important;
      border: 1px solid rgba(80, 225, 255, 0.55);
      border-radius: 10px;
      overflow: hidden;
      background: rgba(80, 225, 255, 0.05);
      margin: 0 auto !important;
  }

  div[role="radiogroup"] > label {
      margin: 0 !important;
      border: none !important;
      border-right: 1px solid rgba(80, 225, 255, 0.35) !important;
      border-radius: 0 !important;
      padding: 0.72rem 1.55rem !important;
      min-width: 155px;
      justify-content: center !important;
      align-items: center !important;
      background: transparent !important;
      transition: background 0.2s ease, color 0.2s ease;
  }

  div[role="radiogroup"] > label:last-child {
      border-right: none !important;
  }

  div[role="radiogroup"] > label:hover {
      background: rgba(80, 225, 255, 0.12) !important;
  }

  div[role="radiogroup"] > label[data-selected="true"] {
      background: rgb(88, 230, 255) !important;
  }

  div[role="radiogroup"] > label[data-selected="true"] p {
      color: #06263a !important;
      font-weight: 700 !important;
  }

  div[role="radiogroup"] > label p {
      color: #dffbff !important;
      font-size: 1rem !important;
      font-weight: 500 !important;
      margin: 0 !important;
  }

  div[role="radiogroup"] input,
  div[role="radiogroup"] input[type="radio"],
  div[role="radiogroup"] svg,
  div[role="radiogroup"] [data-testid="stMarkdownContainer"] + div,
  div[role="radiogroup"] label > div:first-child {
      display: none !important;
      visibility: hidden !important;
      width: 0 !important;
      height: 0 !important;
      min-width: 0 !important;
      min-height: 0 !important;
      margin: 0 !important;
      padding: 0 !important;
  }

  div[data-testid="stImage"] img {
      max-height: 520px !important;
      width: auto !important;
      max-width: 100% !important;
      object-fit: contain !important;
      display: block !important;
      margin-left: auto !important;
      margin-right: auto !important;
  }

  .loader-overlay {
      position: fixed;
      inset: 0;
      display: flex;
      justify-content: center;
      align-items: center;
      background: rgba(3, 8, 20, 0.72);
      z-index: 999999;
      backdrop-filter: blur(2px);
  }

  .loader-box {
      display: flex;
      align-items: center;
      gap: 14px;
      padding: 18px 26px;
      border-radius: 16px;
      background: rgba(15, 23, 42, 0.96);
      border: 1px solid rgba(255,255,255,0.09);
      box-shadow: 0 10px 30px rgba(0,0,0,0.30);
      font-size: 20px;
      font-weight: 500;
  }

  .custom-loader {
      width: 24px;
      height: 24px;
      border: 3px solid rgba(255,255,255,0.18);
      border-top: 3px solid rgba(255,255,255,0.95);
      border-radius: 50%;
      animation: spin 0.9s linear infinite;
  }

  @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
  }
</style>
"""


def apply_analysis_styles() -> None:
    st.markdown(ANALYSIS_CSS, unsafe_allow_html=True)


def show_fixed_loader(message: str = "Preparing analysis..."):
    holder = st.empty()
    holder.markdown(
        f"""
        <div class="loader-overlay">
            <div class="loader-box">
                <div class="custom-loader"></div>
                <span>{message}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return holder


def clear_fixed_loader(holder) -> None:
    holder.empty()