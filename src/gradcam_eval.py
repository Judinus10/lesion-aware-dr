import os
import argparse
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import DataLoader
import timm

import albumentations as A
from albumentations.pytorch import ToTensorV2

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# -----------------------------
# Dataset (same style as your eval.py)
# -----------------------------
class EyePacsDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path: str, image_dir: str, image_col: str, label_col: str, transform=None):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.image_col = image_col
        self.label_col = label_col
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        fname = str(row[self.image_col])
        label = int(row[self.label_col])

        img_path = os.path.join(self.image_dir, fname)
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            x = self.transform(image=rgb)["image"]
        else:
            x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

        return x, label, fname, rgb  # return original RGB for overlay


def build_transform(image_size: int):
    # ImageNet normalization (matches EfficientNet pretrained)
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def load_cfg(cfg_path: str) -> dict:
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt

    new_state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(new_state, strict=True)


def find_last_conv_layer(model: torch.nn.Module) -> torch.nn.Module:
    last_conv = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise RuntimeError("No Conv2d layer found in model. Can't run Grad-CAM.")
    return last_conv


def overlay_cam_on_image(rgb_uint8: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    rgb_uint8: HxWx3 RGB in [0..255]
    cam: HxW in [0..1]
    """
    h, w = rgb_uint8.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    cam_uint8 = np.uint8(255 * cam_resized)

    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    overlay = (1 - alpha) * rgb_uint8.astype(np.float32) + alpha * heatmap_rgb.astype(np.float32)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg_path", type=str, required=True)
    ap.add_argument("--ckpt_path", type=str, required=True)
    ap.add_argument("--split", type=str, default="val", choices=["val", "train"])
    ap.add_argument("--method", type=str, default="gradcampp", choices=["gradcam", "gradcampp"])
    ap.add_argument("--num_images", type=int, default=24, help="How many images to export heatmaps for")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = load_cfg(args.cfg_path)

    backbone = cfg["model"]["backbone"]
    num_classes = int(cfg["model"]["num_classes"])

    image_size = int(cfg["data"]["image_size"])
    image_dir = cfg["data"]["image_dir"]
    image_col = cfg["data"]["image_col"]
    label_col = cfg["data"]["label_col"]

    csv_path = cfg["data"]["val_csv"] if args.split == "val" else cfg["data"]["train_csv"]

    outputs_dir = Path(cfg["paths"]["outputs_dir"])
    out_dir = outputs_dir / "gradcam" / f"{args.method}_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[GradCAM] device: {device}")
    print(f"[GradCAM] csv: {csv_path}")
    print(f"[GradCAM] image_dir: {image_dir}")
    print(f"[GradCAM] ckpt: {args.ckpt_path}")
    print(f"[GradCAM] out_dir: {out_dir}")

    # Data
    tfm = build_transform(image_size)
    ds = EyePacsDataset(csv_path, image_dir, image_col, label_col, transform=tfm)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type=="cuda"))

    # Model (same style as your eval.py)
    model = timm.create_model(backbone, pretrained=False, num_classes=num_classes).to(device)
    load_checkpoint(model, args.ckpt_path, device)
    model.eval()

    # Target layer (auto)
    target_layer = find_last_conv_layer(model)
    print(f"[GradCAM] Using target layer: {target_layer}")

    # Choose CAM method
    cam_class = GradCAMPlusPlus if args.method == "gradcampp" else GradCAM
    cam = cam_class(model=model, target_layers=[target_layer])

    exported = 0

    # We also compute predictions to label files nicely
    softmax = torch.nn.Softmax(dim=1)

    for xb, yb, names, rgbs in dl:
        if exported >= args.num_images:
            break

        xb = xb.to(device, non_blocking=True)

        # Forward for preds
        with torch.no_grad():
            logits = model(xb)
            probs = softmax(logits)
            pred = torch.argmax(probs, dim=1).cpu().numpy()

        y_true = yb.numpy()

        # Grad-CAM needs targets per sample:
        # We’ll use predicted class for visualization (common practice)
        targets = [ClassifierOutputTarget(int(c)) for c in pred]

        # IMPORTANT: no torch.no_grad here
        grayscale_cams = cam(input_tensor=xb, targets=targets)  # shape: (B, H, W) in [0..1]

        for i in range(len(names)):
            if exported >= args.num_images:
                break

            fname = names[i]
            rgb = rgbs[i].numpy()  # already RGB uint8-ish? could be int
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)

            overlay = overlay_cam_on_image(rgb, grayscale_cams[i], alpha=0.45)

            # Save with info in name
            out_name = f"{exported:04d}__true{y_true[i]}__pred{pred[i]}__{Path(fname).stem}.png"
            out_path = out_dir / out_name

            # write as BGR for cv2
            bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_path), bgr)

            exported += 1

    print(f"[GradCAM] Done. Exported {exported} images to: {out_dir}")


if __name__ == "__main__":
    main()
import os
import argparse
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import DataLoader
import timm

import albumentations as A
from albumentations.pytorch import ToTensorV2

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# -----------------------------
# Dataset (same style as your eval.py)
# -----------------------------
class EyePacsDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path: str, image_dir: str, image_col: str, label_col: str, transform=None):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.image_col = image_col
        self.label_col = label_col
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        fname = str(row[self.image_col])
        label = int(row[self.label_col])

        img_path = os.path.join(self.image_dir, fname)
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            x = self.transform(image=rgb)["image"]
        else:
            x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

        return x, label, fname, rgb  # return original RGB for overlay


def build_transform(image_size: int):
    # ImageNet normalization (matches EfficientNet pretrained)
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def load_cfg(cfg_path: str) -> dict:
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt

    new_state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(new_state, strict=True)


def find_last_conv_layer(model: torch.nn.Module) -> torch.nn.Module:
    last_conv = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise RuntimeError("No Conv2d layer found in model. Can't run Grad-CAM.")
    return last_conv


def overlay_cam_on_image(rgb_uint8: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    rgb_uint8: HxWx3 RGB in [0..255]
    cam: HxW in [0..1]
    """
    h, w = rgb_uint8.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    cam_uint8 = np.uint8(255 * cam_resized)

    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    overlay = (1 - alpha) * rgb_uint8.astype(np.float32) + alpha * heatmap_rgb.astype(np.float32)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg_path", type=str, required=True)
    ap.add_argument("--ckpt_path", type=str, required=True)
    ap.add_argument("--split", type=str, default="val", choices=["val", "train"])
    ap.add_argument("--method", type=str, default="gradcampp", choices=["gradcam", "gradcampp"])
    ap.add_argument("--num_images", type=int, default=24, help="How many images to export heatmaps for")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = load_cfg(args.cfg_path)

    backbone = cfg["model"]["backbone"]
    num_classes = int(cfg["model"]["num_classes"])

    image_size = int(cfg["data"]["image_size"])
    image_dir = cfg["data"]["image_dir"]
    image_col = cfg["data"]["image_col"]
    label_col = cfg["data"]["label_col"]

    csv_path = cfg["data"]["val_csv"] if args.split == "val" else cfg["data"]["train_csv"]

    outputs_dir = Path(cfg["paths"]["outputs_dir"])
    out_dir = outputs_dir / "gradcam" / f"{args.method}_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[GradCAM] device: {device}")
    print(f"[GradCAM] csv: {csv_path}")
    print(f"[GradCAM] image_dir: {image_dir}")
    print(f"[GradCAM] ckpt: {args.ckpt_path}")
    print(f"[GradCAM] out_dir: {out_dir}")

    # Data
    tfm = build_transform(image_size)
    ds = EyePacsDataset(csv_path, image_dir, image_col, label_col, transform=tfm)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type=="cuda"))

    # Model (same style as your eval.py)
    model = timm.create_model(backbone, pretrained=False, num_classes=num_classes).to(device)
    load_checkpoint(model, args.ckpt_path, device)
    model.eval()

    # Target layer (auto)
    target_layer = find_last_conv_layer(model)
    print(f"[GradCAM] Using target layer: {target_layer}")

    # Choose CAM method
    cam_class = GradCAMPlusPlus if args.method == "gradcampp" else GradCAM
    cam = cam_class(model=model, target_layers=[target_layer])

    exported = 0

    # We also compute predictions to label files nicely
    softmax = torch.nn.Softmax(dim=1)

    for xb, yb, names, rgbs in dl:
        if exported >= args.num_images:
            break

        xb = xb.to(device, non_blocking=True)

        # Forward for preds
        with torch.no_grad():
            logits = model(xb)
            probs = softmax(logits)
            pred = torch.argmax(probs, dim=1).cpu().numpy()

        y_true = yb.numpy()

        # Grad-CAM needs targets per sample:
        # We’ll use predicted class for visualization (common practice)
        targets = [ClassifierOutputTarget(int(c)) for c in pred]

        # IMPORTANT: no torch.no_grad here
        grayscale_cams = cam(input_tensor=xb, targets=targets)  # shape: (B, H, W) in [0..1]

        for i in range(len(names)):
            if exported >= args.num_images:
                break

            fname = names[i]
            rgb = rgbs[i].numpy()  # already RGB uint8-ish? could be int
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)

            overlay = overlay_cam_on_image(rgb, grayscale_cams[i], alpha=0.45)

            # Save with info in name
            out_name = f"{exported:04d}__true{y_true[i]}__pred{pred[i]}__{Path(fname).stem}.png"
            out_path = out_dir / out_name

            # write as BGR for cv2
            bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_path), bgr)

            exported += 1

    print(f"[GradCAM] Done. Exported {exported} images to: {out_dir}")


if __name__ == "__main__":
    main()
