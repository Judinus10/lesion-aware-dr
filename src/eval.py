import os
import json
import argparse
from pathlib import Path

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


class EyePacsDataset(Dataset):
    def __init__(self, csv_path: str, image_dir: str, image_col: str, label_col: str, image_size: int, transform=None):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.image_col = image_col
        self.label_col = label_col
        self.image_size = int(image_size)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        fname = row[self.image_col]
        label = int(row[self.label_col])

        img_path = os.path.join(self.image_dir, fname)

        try:
            img = Image.open(img_path).convert("RGB")
            img = np.array(img)
        except Exception as e:
            print(f"[WARN] Failed to read image: {img_path} | {e}")
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        if self.transform is not None:
            img = self.transform(image=img)["image"]

        return img, label, fname


def build_transform(image_size: int):
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


def get_train_stats(train_csv: str, label_col: str, num_classes: int, device: torch.device):
    train_df = pd.read_csv(train_csv)

    class_counts = (
        train_df[label_col]
        .value_counts()
        .sort_index()
        .reindex(range(num_classes), fill_value=0)
        .values.astype(np.float64)
    )

    class_priors = class_counts / class_counts.sum()
    log_priors = np.log(class_priors + 1e-12)

    log_priors_t = torch.tensor(log_priors, dtype=torch.float32, device=device)
    class_counts_t = torch.tensor(class_counts, dtype=torch.float32, device=device)

    return log_priors_t, class_counts_t, class_counts, class_priors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg_path", type=str, required=True)
    ap.add_argument("--ckpt_path", type=str, default=None)
    ap.add_argument("--split", type=str, default="val", choices=["val", "train"])
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    cfg = load_cfg(args.cfg_path)

    backbone = cfg["model"]["backbone"]
    num_classes = int(cfg["model"]["num_classes"])

    image_size = int(cfg["data"]["image_size"])
    image_dir = cfg["data"]["image_dir"]
    image_col = cfg["data"]["image_col"]
    label_col = cfg["data"]["label_col"]

    train_csv = cfg["data"]["train_csv"]
    val_csv = cfg["data"]["val_csv"]

    outputs_dir = cfg["paths"]["outputs_dir"]

    loss_name = cfg.get("loss", {}).get("name", "ce")
    tau = float(cfg.get("loss", {}).get("tau", 1.0))

    csv_path = val_csv if args.split == "val" else train_csv

    if args.ckpt_path is None:
        ckpt_dir = cfg["paths"]["checkpoints_dir"]
        args.ckpt_path = str(Path(ckpt_dir) / "best_macro_f1_model.pt")

    ckpt_name = Path(args.ckpt_path).stem
    eval_dir = Path(outputs_dir) / "eval" / ckpt_name
    eval_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EVAL] device: {device}")
    print(f"[EVAL] csv: {csv_path}")
    print(f"[EVAL] image_dir: {image_dir}")
    print(f"[EVAL] ckpt: {args.ckpt_path}")
    print(f"[EVAL] loss_name: {loss_name}")
    if loss_name == "logit_adjusted_ce":
        print(f"[EVAL] tau: {tau}")

    log_priors_t, class_counts_t, class_counts, class_priors = get_train_stats(
        train_csv=train_csv,
        label_col=label_col,
        num_classes=num_classes,
        device=device,
    )

    print(f"[EVAL] class_counts: {class_counts.tolist()}")
    print(f"[EVAL] class_priors: {[round(float(x), 6) for x in class_priors]}")

    tfm = build_transform(image_size)
    ds = EyePacsDataset(
        csv_path=csv_path,
        image_dir=image_dir,
        image_col=image_col,
        label_col=label_col,
        image_size=image_size,
        transform=tfm,
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
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

            if loss_name == "logit_adjusted_ce":
                adjusted_logits = logits + tau * log_priors_t.unsqueeze(0)
            elif loss_name == "balanced_softmax":
                adjusted_logits = logits + torch.log(class_counts_t.unsqueeze(0) + 1e-12)
            else:
                adjusted_logits = logits

            probs = torch.softmax(adjusted_logits, dim=1)
            pred = torch.argmax(adjusted_logits, dim=1)

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

    (eval_dir / "classification_report.txt").write_text(report)

    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "num_samples": len(y_true),
        "ckpt_path": args.ckpt_path,
        "csv_path": csv_path,
        "image_dir": image_dir,
        "backbone": backbone,
        "loss_name": loss_name,
        "tau": tau if loss_name == "logit_adjusted_ce" else None,
        "class_counts": class_counts.tolist(),
        "class_priors": class_priors.tolist(),
    }
    (eval_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    plot_confusion_matrix(cm, str(eval_dir / "confusion_matrix.png"))

    pred_df = pd.DataFrame({
        "image_id": names_all,
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