import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
import timm

import albumentations as A
from albumentations.pytorch import ToTensorV2


# -------------------------
# SETTINGS (EDIT THESE)
# -------------------------
CKPT_PATH = "outputs/2026-02-27/checkpoints/best_macro_f1_model.pt"  # change if needed
MODEL_NAME = "efficientnet_b0"  # change to your trained model name if different
NUM_CLASSES = 5
CLASS_NAMES = ["No DR (0)", "Mild (1)", "Moderate (2)", "Severe (3)", "Proliferative (4)"]
IMG_SIZE = 512  # must match training


# -------------------------
# Helpers
# -------------------------
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    # Handle different checkpoint formats
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict):
        # sometimes directly the state dict
        state = ckpt
    else:
        raise ValueError("Checkpoint format not recognized.")

    # remove possible 'module.' prefix
    new_state = {}
    for k, v in state.items():
        nk = k.replace("module.", "")
        new_state[nk] = v

    model.load_state_dict(new_state, strict=False)
    model.eval().to(device)

    return model, device


def get_transform():
    return A.Compose(
        [
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def preprocess(pil_img: Image.Image):
    img = np.array(pil_img.convert("RGB"))
    aug = get_transform()(image=img)
    x = aug["image"].unsqueeze(0)  # [1,3,H,W]
    return x, img


# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="DR Predict", layout="wide")
st.title("DR Grading - Predict")

st.warning("This is a student project demo. Not for clinical use.")

uploaded = st.file_uploader("Upload a fundus image (JPG/PNG)", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns([1, 1])

if uploaded:
    pil_img = Image.open(uploaded)
    with col1:
        st.subheader("Input")
        st.image(pil_img, use_container_width=True)

    model, device = load_model()
    x, raw_rgb = preprocess(pil_img)
    x = x.to(device)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
        pred_idx = int(np.argmax(probs))
        pred_name = CLASS_NAMES[pred_idx]

    with col2:
        st.subheader("Prediction")
        st.metric("Predicted Class", pred_name)
        st.write("**Probabilities**")
        for i, p in enumerate(probs):
            st.write(f"- {CLASS_NAMES[i]}: **{p:.4f}**")

else:
    st.info("Upload an image to get prediction + probabilities.")