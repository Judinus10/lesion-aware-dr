import streamlit as st

st.header("📊 Model Overview")

st.markdown("""
**Backbone:** EfficientNet-B0  
**Dataset:** EyePACS  
**Classes:** 5 (No DR → Proliferative DR)  
**Loss:** Class-weighted / Focal (configurable)  
""")

col1, col2, col3 = st.columns(3)

col1.metric("Train Accuracy", "—")
col2.metric("Validation F1", "—")
col3.metric("Best Epoch", "—")

st.warning("Metrics will be populated after training.")
