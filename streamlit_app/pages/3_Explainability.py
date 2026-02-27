import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
import timm

import albumentations as A
from albumentations.pytorch import ToTensorV2

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


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
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        raise ValueError("Checkpoint format not recognized.")

    new_state = {k.replace("module.", ""): v for k, v in state.items()}
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


def find_last_conv_layer(model: torch.nn.Module):
    # Pick the last Conv2d layer automatically (works for most CNN backbones)
    last = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    if last is None:
        raise ValueError("No Conv2d layer found. You must manually set the target layer.")
    return last


def overlay_heatmap(rgb_img, cam_mask, alpha=0.45):
    # rgb_img: HxWx3 in RGB, uint8
    # cam_mask: HxW float [0..1]
    heatmap = (cam_mask * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = (alpha * heatmap + (1 - alpha) * rgb_img).astype(np.uint8)
    return overlay, heatmap


# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="DR Explainability", layout="wide")
st.title("Explainability - Grad-CAM++")

uploaded = st.file_uploader("Upload a fundus image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if not uploaded:
    st.info("Upload an image to generate a Grad-CAM++ heatmap.")
    st.stop()

pil_img = Image.open(uploaded)
model, device = load_model()

x, raw_rgb = preprocess(pil_img)
x = x.to(device)

# Predict first
with torch.no_grad():
    logits = model(x)
    probs = F.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
    pred_idx = int(np.argmax(probs))

st.write(f"**Predicted:** {CLASS_NAMES[pred_idx]} (p={probs[pred_idx]:.4f})")

# Grad-CAM++ for predicted class (or user-selected)
target_class = st.selectbox(
    "Generate heatmap for class",
    options=list(range(NUM_CLASSES)),
    format_func=lambda i: CLASS_NAMES[i],
    index=pred_idx,
)

try:
    target_layer = find_last_conv_layer(model)
except Exception as e:
    st.error(str(e))
    st.stop()

cam = GradCAMPlusPlus(model=model, target_layers=[target_layer])

targets = [ClassifierOutputTarget(int(target_class))]

# cam returns [B,H,W]
grayscale_cam = cam(input_tensor=x, targets=targets)[0]

overlay, heatmap = overlay_heatmap(raw_rgb, grayscale_cam)

c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("Input")
    st.image(raw_rgb, use_container_width=True)
with c2:
    st.subheader("Heatmap")
    st.image(heatmap, use_container_width=True)
with c3:
    st.subheader("Overlay")
    st.image(overlay, use_container_width=True)