from __future__ import annotations

import argparse
import yaml
from omegaconf import OmegaConf

import numpy as np
import torch
from tqdm import tqdm

from segmentation.src.models import build_model
from segmentation.src.data.dr_datamodule import DRDataModule
from segmentation.src.losses.segmentation import get_loss


def load_config(cfg_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)
    return OmegaConf.create(cfg_dict)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, default="segmentation/configs/base.yaml")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    return parser.parse_args()


@torch.no_grad()
def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    dims = (0, 2, 3)
    intersection = (preds * targets).sum(dim=dims)
    union = preds.sum(dim=dims) + targets.sum(dim=dims) - intersection
    denom = preds.sum(dim=dims) + targets.sum(dim=dims)

    iou = (intersection + 1e-6) / (union + 1e-6)
    dice = (2 * intersection + 1e-6) / (denom + 1e-6)

    return {
        "iou_ex": float(iou[0].item()),
        "iou_he": float(iou[1].item()),
        "dice_ex": float(dice[0].item()),
        "dice_he": float(dice[1].item()),
        "mean_iou": float(iou.mean().item()),
        "mean_dice": float(dice.mean().item()),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold: float):
    model.eval()
    running_loss = 0.0
    metric_list = []

    for batch in tqdm(loader, desc="Eval"):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, masks)

        running_loss += loss.item()
        metric_list.append(compute_metrics(logits, masks, threshold=threshold))

    avg_loss = running_loss / max(len(loader), 1)
    keys = metric_list[0].keys() if metric_list else []
    avg_metrics = {k: float(np.mean([m[k] for m in metric_list])) for k in keys}

    return avg_loss, avg_metrics


def main():
    args = parse_args()
    cfg = load_config(args.cfg_path)

    device = torch.device(
        "cuda" if str(cfg.device).lower() == "cuda" and torch.cuda.is_available() else "cpu"
    )

    dm = DRDataModule(cfg)
    loader = dm.val_dataloader() if args.split == "val" else dm.test_dataloader()

    model = build_model(cfg).to(device)
    criterion = get_loss(cfg)

    ckpt = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    loss, metrics = evaluate(
        model,
        loader,
        criterion,
        device,
        threshold=float(cfg.data.threshold),
    )

    print(f"\n[{args.split.upper()} RESULTS]")
    print(f"loss      : {loss:.4f}")
    print(f"mean_dice : {metrics['mean_dice']:.4f}")
    print(f"mean_iou  : {metrics['mean_iou']:.4f}")
    print(f"dice_ex   : {metrics['dice_ex']:.4f}")
    print(f"dice_he   : {metrics['dice_he']:.4f}")
    print(f"iou_ex    : {metrics['iou_ex']:.4f}")
    print(f"iou_he    : {metrics['iou_he']:.4f}")


if __name__ == "__main__":
    main()