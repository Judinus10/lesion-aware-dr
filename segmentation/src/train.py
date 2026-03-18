from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime

import yaml
from omegaconf import OmegaConf

import numpy as np
import torch
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from segmentation.src.utils.logger import get_logger
from segmentation.src.utils.seed import set_seed
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
    parser.add_argument("--resume_ckpt", type=str, default=None)
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


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for batch in tqdm(loader, desc="Train", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, criterion, device, threshold: float):
    model.eval()
    running_loss = 0.0
    metric_list = []

    for batch in tqdm(loader, desc="Val", leave=False):
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

    set_seed(int(cfg.seed))

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(cfg.paths.outputs_dir) / timestamp
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger("seg_train", str(run_dir / "train.log"))
    logger.info("Starting segmentation training")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    device = torch.device(
        "cuda" if str(cfg.device).lower() == "cuda" and torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Using device: {device}")

    dm = DRDataModule(cfg)
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    model = build_model(cfg).to(device)
    criterion = get_loss(cfg)

    optimizer_name = str(cfg.training.optimizer).lower()
    if optimizer_name == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=float(cfg.training.lr),
            weight_decay=float(cfg.training.weight_decay),
        )
    elif optimizer_name == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=float(cfg.training.lr),
            weight_decay=float(cfg.training.weight_decay),
        )
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.training.optimizer}")

    scheduler_name = str(cfg.training.scheduler).lower()
    if scheduler_name == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=int(cfg.training.epochs))
    else:
        scheduler = None

    start_epoch = 0
    best_score = -1.0

    if args.resume_ckpt:
        ckpt = torch.load(args.resume_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler and ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_score = float(ckpt.get("best_score", -1.0))
        logger.info(f"Resumed from checkpoint: {args.resume_ckpt}")

    for epoch in range(start_epoch, int(cfg.training.epochs)):
        logger.info(f"Epoch [{epoch + 1}/{cfg.training.epochs}]")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = validate(
            model,
            val_loader,
            criterion,
            device,
            threshold=float(cfg.data.threshold),
        )

        if scheduler is not None:
            scheduler.step()

        logger.info(
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"mean_dice={val_metrics['mean_dice']:.4f} | "
            f"mean_iou={val_metrics['mean_iou']:.4f} | "
            f"dice_ex={val_metrics['dice_ex']:.4f} | "
            f"dice_he={val_metrics['dice_he']:.4f}"
        )

        latest_ckpt = ckpt_dir / "last.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "best_score": best_score,
                "cfg": OmegaConf.to_container(cfg, resolve=True),
            },
            latest_ckpt,
        )

        current_score = val_metrics["mean_dice"]
        if current_score > best_score:
            best_score = current_score
            best_ckpt = ckpt_dir / "best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    "best_score": best_score,
                    "cfg": OmegaConf.to_container(cfg, resolve=True),
                },
                best_ckpt,
            )
            logger.info(f"Saved best checkpoint to: {best_ckpt}")

    logger.info("Training completed")


if __name__ == "__main__":
    main()