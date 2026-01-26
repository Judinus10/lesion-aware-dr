import streamlit as st
from PIL import Image

st.header("🖼️ DR Image Prediction")

uploaded_file = st.file_uploader(
    "Upload a fundus image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.success("Image loaded successfully")

    st.button("Run Prediction (coming soon)")
