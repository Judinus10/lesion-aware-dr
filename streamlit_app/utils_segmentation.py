import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp


# -------------------------
# SETTINGS
# -------------------------
SEG_MODEL_NAME = "unet_resnet34"
SEG_NUM_CLASSES = 2
SEG_CLASS_NAMES = ["Exudates (EX)", "Haemorrhages (HE)"]
SEG_IMG_SIZE = 512
SEG_THRESHOLD = 0.50

APP_DIR = Path(__file__).resolve().parent
SEG_CKPT_PATH = str(APP_DIR / "outputs" / "models" / "segmentation" / "best_model.pt")


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transform():
    return A.Compose(
        [
            A.Resize(SEG_IMG_SIZE, SEG_IMG_SIZE),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def pil_to_rgb_np(pil_img: Image.Image) -> np.ndarray:
    return np.array(pil_img.convert("RGB"))


def preprocess(pil_img: Image.Image):
    rgb = pil_to_rgb_np(pil_img)
    aug = get_transform()(image=rgb)
    x = aug["image"].unsqueeze(0)
    return x, rgb


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model", "model_state", "model_state_dict"]:
            if key in ckpt:
                return ckpt[key]
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError("Segmentation checkpoint format not recognized.")


def _clean_state_dict(state):
    return {k.replace("module.", ""): v for k, v in state.items()}


def build_model():
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=SEG_NUM_CLASSES,
    )


def load_model_cached():
    device = get_device()
    model = build_model()

    if not os.path.exists(SEG_CKPT_PATH):
        raise FileNotFoundError(f"Segmentation checkpoint not found: {SEG_CKPT_PATH}")

    ckpt = torch.load(SEG_CKPT_PATH, map_location="cpu")
    state = _extract_state_dict(ckpt)
    state = _clean_state_dict(state)

    missing, unexpected = model.load_state_dict(state, strict=False)

    if missing:
        head_like = [k for k in missing if "segmentation_head" in k or "decoder" in k or "encoder" in k]
        if head_like:
            raise RuntimeError(
                "Segmentation checkpoint did not load cleanly.\n"
                f"Missing keys (sample): {missing[:10]}\n"
                f"Unexpected keys (sample): {unexpected[:10]}"
            )

    model.eval().to(device)
    return model, device


def resize_mask_to_raw(mask: np.ndarray, raw_rgb: np.ndarray) -> np.ndarray:
    return cv2.resize(
        mask.astype(np.uint8),
        (raw_rgb.shape[1], raw_rgb.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )


def make_color_mask(binary_mask: np.ndarray, color_rgb: tuple[int, int, int]) -> np.ndarray:
    color_mask = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 3), dtype=np.uint8)
    color_mask[binary_mask > 0] = color_rgb
    return color_mask


def blend_overlay(base_rgb: np.ndarray, color_mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    return cv2.addWeighted(base_rgb, 1.0, color_mask, alpha, 0)


def predict_segmentation(model, device, pil_img: Image.Image, threshold: float = SEG_THRESHOLD):
    x, raw_rgb = preprocess(pil_img)
    x = x.to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).squeeze(0).detach().cpu().numpy()

    # Channel 0 = EX, Channel 1 = HE
    ex_prob = probs[0]
    he_prob = probs[1]

    ex_mask = (ex_prob >= threshold).astype(np.uint8)
    he_mask = (he_prob >= threshold).astype(np.uint8)

    ex_mask_raw = resize_mask_to_raw(ex_mask, raw_rgb)
    he_mask_raw = resize_mask_to_raw(he_mask, raw_rgb)
    combined_mask_raw = np.clip(ex_mask_raw + he_mask_raw, 0, 1).astype(np.uint8)

    ex_color = (0, 255, 255)     # cyan-ish
    he_color = (255, 80, 80)     # red-ish
    combined_color = (255, 200, 0)

    ex_color_mask = make_color_mask(ex_mask_raw, ex_color)
    he_color_mask = make_color_mask(he_mask_raw, he_color)

    combined_color_mask = np.zeros_like(ex_color_mask)
    combined_color_mask[ex_mask_raw > 0] = ex_color
    combined_color_mask[he_mask_raw > 0] = he_color

    ex_overlay = blend_overlay(raw_rgb, ex_color_mask, alpha=0.40)
    he_overlay = blend_overlay(raw_rgb, he_color_mask, alpha=0.40)
    combined_overlay = blend_overlay(raw_rgb, combined_color_mask, alpha=0.45)

    return {
        "raw_rgb": raw_rgb,
        "ex_prob": ex_prob,
        "he_prob": he_prob,
        "ex_mask": ex_mask_raw,
        "he_mask": he_mask_raw,
        "combined_mask": combined_mask_raw,
        "ex_overlay": ex_overlay,
        "he_overlay": he_overlay,
        "combined_overlay": combined_overlay,
    }