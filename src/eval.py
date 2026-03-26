from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

import timm
import yaml

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt


class DRCSVDataset(Dataset):
    """
    Generic CSV dataset for evaluation.

    Supports:
    1) Full-path mode:
       image_path,label,dataset
       /full/path/to/image.png,0,eyepacs

    2) Relative-name mode:
       image_id,label
       img001.png,0
       + image_dir in config
    """

    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        image_col: str,
        label_col: str,
        image_size: int,
        dataset_col: Optional[str] = None,
        transform=None,
    ):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.image_dir = str(image_dir).strip()
        self.image_col = str(image_col)
        self.label_col = str(label_col)
        self.dataset_col = str(dataset_col) if dataset_col else None
        self.image_size = int(image_size)
        self.transform = transform

        missing = []
        if self.image_col not in self.df.columns:
            missing.append(self.image_col)
        if self.label_col not in self.df.columns:
            missing.append(self.label_col)

        if missing:
            raise ValueError(
                f"Missing required columns in CSV {csv_path}. "
                f"Missing={missing}, found={list(self.df.columns)}"
            )

    def __len__(self):
        return len(self.df)

    def _resolve_path(self, raw_value: str) -> Path:
        raw_value = str(raw_value)

        if self.image_col == "image_path":
            return Path(raw_value)

        if not self.image_dir:
            raise ValueError(
                f"image_dir is empty, but image_col='{self.image_col}' is not full-path mode."
            )

        return Path(self.image_dir) / raw_value

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        raw_img_value = row[self.image_col]
        label = int(row[self.label_col])

        img_path = self._resolve_path(raw_img_value)

        try:
            img = Image.open(img_path).convert("RGB")
            img = np.array(img)
        except Exception as e:
            print(f"[WARN] Failed to read image: {img_path} | {e}")
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        if self.transform is not None:
            img = self.transform(image=img)["image"]

        meta_name = str(raw_img_value)
        return img, label, meta_name


def build_transform(image_size: int):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def load_cfg(cfg_path: str) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    new_state = {}
    for k, v in state.items():
        new_state[k.replace("module.", "")] = v

    model.load_state_dict(new_state, strict=True)


def plot_confusion_matrix(cm: np.ndarray, out_path: str, class_names=None):
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()

    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def predict_with_rule(
    logits: torch.Tensor,
    prediction_rule: str = "raw_logits_argmax",
    class1_threshold: float = 0.20,
) -> tuple[torch.Tensor, torch.Tensor]:
    probs = torch.softmax(logits, dim=1)

    if prediction_rule == "raw_logits_argmax":
        pred = torch.argmax(logits, dim=1)

    elif prediction_rule == "class1_threshold":
        pred = torch.argmax(probs, dim=1)

        # Force class 1 if its probability is above threshold
        class1_mask = probs[:, 1] >= class1_threshold
        pred = pred.clone()
        pred[class1_mask] = 1

    else:
        raise ValueError(
            f"Unknown prediction_rule='{prediction_rule}'. "
            f"Use 'raw_logits_argmax' or 'class1_threshold'."
        )

    return pred, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg_path", type=str, required=True)
    ap.add_argument("--ckpt_path", type=str, default=None)
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=2)

    # NEW
    ap.add_argument(
        "--prediction_rule",
        type=str,
        default="raw_logits_argmax",
        choices=["raw_logits_argmax", "class1_threshold"],
        help="Prediction rule used during evaluation.",
    )
    ap.add_argument(
        "--class1_threshold",
        type=float,
        default=0.20,
        help="Threshold for forcing class 1 when using prediction_rule=class1_threshold.",
    )

    args = ap.parse_args()

    cfg = load_cfg(args.cfg_path)

    backbone = cfg["model"]["backbone"]
    num_classes = int(cfg["model"]["num_classes"])

    image_size = int(cfg["data"]["image_size"])
    image_dir = cfg["data"].get("image_dir", "")
    image_col = cfg["data"]["image_col"]
    label_col = cfg["data"]["label_col"]
    dataset_col = cfg["data"].get("dataset_col", None)

    train_csv = cfg["data"]["train_csv"]
    val_csv = cfg["data"]["val_csv"]
    test_csv = cfg["data"].get("test_csv", None)

    outputs_dir = cfg["paths"]["outputs_dir"]

    loss_name = str(cfg.get("loss", {}).get("name", "ce")).lower()

    if args.split == "train":
        csv_path = train_csv
    elif args.split == "val":
        csv_path = val_csv
    else:
        if not test_csv:
            raise ValueError("Requested split='test' but cfg.data.test_csv is missing.")
        csv_path = test_csv

    if args.ckpt_path is None:
        ckpt_dir = cfg["paths"]["checkpoints_dir"]
        args.ckpt_path = str(Path(ckpt_dir) / "best_macro_f1_model.pt")

    ckpt_name = Path(args.ckpt_path).stem

    if args.prediction_rule == "class1_threshold":
        eval_name = f"{args.split}_{ckpt_name}_{args.prediction_rule}_{args.class1_threshold:.2f}"
    else:
        eval_name = f"{args.split}_{ckpt_name}_{args.prediction_rule}"

    eval_dir = Path(outputs_dir) / "eval" / eval_name
    eval_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EVAL] device: {device}")
    print(f"[EVAL] split: {args.split}")
    print(f"[EVAL] csv: {csv_path}")
    print(f"[EVAL] image_dir: {image_dir}")
    print(f"[EVAL] ckpt: {args.ckpt_path}")
    print(f"[EVAL] loss_name: {loss_name}")
    print(f"[EVAL] prediction_rule: {args.prediction_rule}")
    if args.prediction_rule == "class1_threshold":
        print(f"[EVAL] class1_threshold: {args.class1_threshold:.2f}")

    tfm = build_transform(image_size)
    ds = DRCSVDataset(
        csv_path=csv_path,
        image_dir=image_dir,
        image_col=image_col,
        label_col=label_col,
        dataset_col=dataset_col,
        image_size=image_size,
        transform=tfm,
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = timm.create_model(backbone, pretrained=False, num_classes=num_classes)
    model.to(device)
    load_checkpoint(model, args.ckpt_path, device)
    model.eval()

    y_true = []
    y_pred = []
    probs_all = []
    names_all = []

    with torch.no_grad():
        for x, y, names in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(x)
            else:
                logits = model(x)

            pred, probs = predict_with_rule(
                logits=logits,
                prediction_rule=args.prediction_rule,
                class1_threshold=args.class1_threshold,
            )

            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())
            probs_all.extend(probs.cpu().numpy().tolist())
            names_all.extend(list(names))

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    report = classification_report(y_true, y_pred, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    print("\n[EVAL RESULTS]")
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro F1 : {macro_f1:.4f}")
    print("\nClassification report:\n")
    print(report)

    (eval_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    metrics = {
        "split": args.split,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "num_samples": len(y_true),
        "ckpt_path": args.ckpt_path,
        "csv_path": csv_path,
        "image_dir": image_dir,
        "backbone": backbone,
        "loss_name": loss_name,
        "prediction_rule": args.prediction_rule,
        "class1_threshold": args.class1_threshold if args.prediction_rule == "class1_threshold" else None,
    }
    (eval_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    plot_confusion_matrix(cm, str(eval_dir / "confusion_matrix.png"))

    pred_df = pd.DataFrame({
        "image_ref": names_all,
        "y_true": y_true,
        "y_pred": y_pred,
    })

    probs_np = np.array(probs_all)
    for c in range(probs_np.shape[1]):
        pred_df[f"prob_{c}"] = probs_np[:, c]

    pred_df.to_csv(eval_dir / "predictions.csv", index=False)

    print(f"\nSaved to: {eval_dir}")


if __name__ == "__main__":
    main()