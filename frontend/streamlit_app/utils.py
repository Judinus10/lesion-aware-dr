import os
import json
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn.functional as F
import timm

import albumentations as A
from albumentations.pytorch import ToTensorV2

from pytorch_grad_cam import GradCAMPlusPlus, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# -------------------------
# SETTINGS
# -------------------------
MODEL_NAME = "efficientnet_b0"
NUM_CLASSES = 5
CLASS_NAMES = ["No DR (0)", "Mild (1)", "Moderate (2)", "Severe (3)", "Proliferative (4)"]
IMG_SIZE = 512  # must match training

# Absolute-safe path: streamlit_app/utils.py -> streamlit_app -> outputs -> ...
APP_DIR = Path(__file__).resolve().parent
CKPT_PATH = str(APP_DIR / "outputs" / "2026-02-27" / "checkpoints" / "best_macro_f1_model.pt")


# -------------------------
# Paths for saved cases
# -------------------------
CASES_DIR = str(APP_DIR / "cases")
IMAGES_DIR = str(Path(CASES_DIR) / "images")
CAMS_DIR = str(Path(CASES_DIR) / "cams")
OVERLAYS_DIR = str(Path(CASES_DIR) / "overlays")
CASES_FILE = str(Path(CASES_DIR) / "cases.jsonl")


def ensure_dirs():
    os.makedirs(CASES_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(CAMS_DIR, exist_ok=True)
    os.makedirs(OVERLAYS_DIR, exist_ok=True)


# -------------------------
# Model + Preprocess
# -------------------------
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transform():
    return A.Compose(
        [
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def pil_to_rgb_np(pil_img: Image.Image) -> np.ndarray:
    return np.array(pil_img.convert("RGB"))


def preprocess(pil_img: Image.Image):
    rgb = pil_to_rgb_np(pil_img)
    aug = get_transform()(image=rgb)
    x = aug["image"].unsqueeze(0)  # [1, 3, H, W]
    return x, rgb


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
        if "model" in ckpt:
            return ckpt["model"]
        if "model_state" in ckpt:
            return ckpt["model_state"]
        if "model_state_dict" in ckpt:
            return ckpt["model_state_dict"]

    raise ValueError(
        f"Checkpoint format not recognized. Keys: "
        f"{list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}"
    )


def _clean_state_dict(state):
    return {k.replace("module.", ""): v for k, v in state.items()}


def debug_state_dict_summary(model: torch.nn.Module, state: dict):
    missing, unexpected = model.load_state_dict(state, strict=False)
    return missing, unexpected


def load_model_cached():
    device = get_device()
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)

    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    state = _extract_state_dict(ckpt)
    state = _clean_state_dict(state)

    remapped = {}

    for k, v in state.items():
        nk = k

        for prefix in ["model.", "net.", "module."]:
            if nk.startswith(prefix):
                nk = nk[len(prefix):]

        nk = nk.replace("head.conv_head.", "conv_head.")
        nk = nk.replace("head.classifier.", "classifier.")
        nk = nk.replace("fc.", "classifier.")
        nk = nk.replace("classifier.1.", "classifier.")

        remapped[nk] = v

    missing, unexpected = model.load_state_dict(remapped, strict=False)

    head_missing = [k for k in missing if ("conv_head" in k or "classifier" in k or "fc" in k)]
    if head_missing:
        print("=== CHECKPOINT LOAD DEBUG ===")
        print("CKPT:", CKPT_PATH)
        print("MISSING HEAD:", head_missing[:20])
        print("UNEXPECTED:", unexpected[:20])
        print("TIP: Your checkpoint head naming doesn't match this timm model.")
        print("=============================")
        raise RuntimeError(
            "Checkpoint loaded but classifier head weights are missing.\n"
            f"Missing head keys (sample): {head_missing[:10]}\n\n"
            "This means your checkpoint model head naming differs.\n"
            "Fix options:\n"
            "1) Use the correct MODEL_NAME used in training\n"
            "2) Or share the printed 'UNEXPECTED' keys so we map correctly\n"
        )

    model.eval().to(device)
    return model, device


def predict(model, device, pil_img: Image.Image):
    x, raw_rgb = preprocess(pil_img)
    x = x.to(device)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
        pred_idx = int(np.argmax(probs))

    return {
        "pred_idx": pred_idx,
        "pred_name": CLASS_NAMES[pred_idx],
        "probs": probs,
        "raw_rgb": raw_rgb,
        "input_tensor": x,
        "logits": logits.squeeze(0).detach().cpu().numpy(),
    }


# -------------------------
# CAM helpers
# -------------------------
def list_conv2d_layers(model: torch.nn.Module):
    layers = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            layers.append((name, m))
    return layers


def choose_target_layer(model: torch.nn.Module, preferred_name: str | None = None):
    convs = list_conv2d_layers(model)
    if not convs:
        raise ValueError("No Conv2d layer found in model.")

    if preferred_name:
        for n, m in convs:
            if n == preferred_name:
                return n, m

    for n, m in convs[::-1]:
        if "conv_head" in n:
            return n, m

    return convs[-1][0], convs[-1][1]


def overlay_heatmap(rgb_img: np.ndarray, cam_mask: np.ndarray, alpha: float = 0.45):
    cam_mask = np.clip(cam_mask, 0.0, 1.0)

    if cam_mask.shape[:2] != rgb_img.shape[:2]:
        cam_mask = cv2.resize(
            cam_mask,
            (rgb_img.shape[1], rgb_img.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    heat = (cam_mask * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = (alpha * heatmap + (1 - alpha) * rgb_img).astype(np.uint8)
    return overlay, heatmap


def run_cam(
    model,
    input_tensor,
    target_class: int,
    target_layer_module,
    method: str = "gradcampp",
):
    method = method.lower().strip()

    if method == "gradcampp":
        cam_extractor = GradCAMPlusPlus(model=model, target_layers=[target_layer_module])
    elif method == "scorecam":
        cam_extractor = ScoreCAM(model=model, target_layers=[target_layer_module])
    else:
        raise ValueError(f"Unsupported CAM method: {method}")

    targets = [ClassifierOutputTarget(int(target_class))]
    grayscale_cam = cam_extractor(input_tensor=input_tensor, targets=targets)[0]
    return grayscale_cam


# Backward-compatible wrapper so old code won’t break
def run_gradcam_pp(model, input_tensor, target_class: int, target_layer_module):
    return run_cam(
        model=model,
        input_tensor=input_tensor,
        target_class=target_class,
        target_layer_module=target_layer_module,
        method="gradcampp",
    )


# -------------------------
# Save case (json + images)
# -------------------------
def save_case(
    filename_hint: str,
    raw_rgb: np.ndarray,
    heatmap_rgb: np.ndarray,
    overlay_rgb: np.ndarray,
    pred_idx: int,
    probs: np.ndarray,
    target_class: int,
    target_layer_name: str,
    cam_method: str = "GradCAM++",
):
    ensure_dirs()

    case_id = str(uuid.uuid4())[:8]
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    safe_name = "".join([c for c in filename_hint if c.isalnum() or c in ("_", "-", ".")])[:60]
    if not safe_name:
        safe_name = "fundus"

    base = f"{ts}_{case_id}_{safe_name}"

    img_path = os.path.join(IMAGES_DIR, f"{base}.png")
    cam_path = os.path.join(CAMS_DIR, f"{base}_heatmap.png")
    overlay_path = os.path.join(OVERLAYS_DIR, f"{base}_overlay.png")

    Image.fromarray(raw_rgb).save(img_path)
    Image.fromarray(heatmap_rgb).save(cam_path)
    Image.fromarray(overlay_rgb).save(overlay_path)

    record = {
        "case_id": case_id,
        "timestamp": ts,
        "filename_hint": filename_hint,
        "pred_idx": int(pred_idx),
        "pred_name": CLASS_NAMES[int(pred_idx)],
        "probs": [float(x) for x in probs.tolist()],
        "target_class": int(target_class),
        "target_layer": target_layer_name,
        "cam_method": cam_method,
        "paths": {
            "input": img_path,
            "heatmap": cam_path,
            "overlay": overlay_path,
        },
    }

    with open(CASES_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return record


def load_recent_cases(limit=30):
    ensure_dirs()

    if not os.path.exists(CASES_FILE):
        return []

    rows = []
    with open(CASES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass

    return rows[-limit:][::-1]