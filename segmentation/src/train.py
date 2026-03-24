from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import SegmentationDataset
from src.losses.segmentation import BCEDiceLoss
from src.models.unet import UNet
from src.utils.logger import get_logger
from src.utils.metrics import compute_batch_metrics
from src.utils.seed import set_seed


def load_config(cfg_path: str) -> dict:
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def build_model(cfg: dict) -> torch.nn.Module:
    return UNet(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        base_channels=cfg["model"]["base_channels"],
    )


def build_optimizer(model: torch.nn.Module, cfg: dict):
    return AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )


def build_scheduler(optimizer, cfg: dict):
    return CosineAnnealingLR(
        optimizer,
        T_max=int(cfg["training"]["epochs"]),
        eta_min=float(cfg["training"].get("min_lr", 1e-6)),
    )


def make_loader(csv_path: str, cfg: dict, is_train: bool) -> DataLoader:
    ds = SegmentationDataset(
        csv_path=csv_path,
        image_size=int(cfg["data"]["image_size"]),
        is_train=is_train,
    )
    return DataLoader(
        ds,
        batch_size=int(cfg["data"]["batch_size"]),
        shuffle=is_train,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=bool(cfg["data"]["pin_memory"]),
        drop_last=False,
    )


def run_one_epoch(model, loader, criterion, optimizer, device, threshold, train: bool):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_metrics = {
        "dice_ex": 0.0,
        "dice_he": 0.0,
        "dice_mean": 0.0,
        "iou_ex": 0.0,
        "iou_he": 0.0,
        "iou_mean": 0.0,
    }

    num_batches = 0
    pbar = tqdm(loader, leave=False)

    for images, targets, _ in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, targets)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        metrics = compute_batch_metrics(logits.detach(), targets, threshold=threshold)

        total_loss += float(loss.item())
        for k in total_metrics:
            total_metrics[k] += metrics[k]

        num_batches += 1
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            dice=f"{metrics['dice_mean']:.4f}",
            iou=f"{metrics['iou_mean']:.4f}",
        )

    avg_loss = total_loss / max(1, num_batches)
    avg_metrics = {k: v / max(1, num_batches) for k, v in total_metrics.items()}

    return avg_loss, avg_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, default="configs/base.yaml")
    args = parser.parse_args()

    cfg = load_config(args.cfg_path)
    set_seed(int(cfg.get("seed", 42)))

    output_dir = Path(cfg["paths"]["outputs_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger("seg_train", str(output_dir / "train.log"))

    device_name = cfg.get("device", "cuda")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")

    logger.info("Starting segmentation training")
    logger.info(f"Using device: {device}")
    logger.info(f"Config:\n{yaml.dump(cfg, sort_keys=False)}")

    train_loader = make_loader(cfg["paths"]["train_csv"], cfg, is_train=True)
    val_loader = make_loader(cfg["paths"]["val_csv"], cfg, is_train=False)

    model = build_model(cfg).to(device)
    criterion = BCEDiceLoss(
        bce_weight=float(cfg["loss"]["bce_weight"]),
        dice_weight=float(cfg["loss"]["dice_weight"]),
        smooth=float(cfg["loss"].get("smooth", 1.0)),
    )
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    best_val_dice = -1.0
    history = []

    epochs = int(cfg["training"]["epochs"])
    threshold = float(cfg["data"]["threshold"])

    for epoch in range(1, epochs + 1):
        logger.info(f"Epoch [{epoch}/{epochs}]")

        train_loss, train_metrics = run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            threshold=threshold,
            train=True,
        )

        val_loss, val_metrics = run_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            threshold=threshold,
            train=False,
        )

        scheduler.step()

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_record)

        logger.info(
            f"train_loss={train_loss:.4f} | "
            f"train_dice={train_metrics['dice_mean']:.4f} | "
            f"train_iou={train_metrics['iou_mean']:.4f}"
        )
        logger.info(
            f"val_loss={val_loss:.4f} | "
            f"val_dice={val_metrics['dice_mean']:.4f} | "
            f"val_iou={val_metrics['iou_mean']:.4f} | "
            f"val_dice_ex={val_metrics['dice_ex']:.4f} | "
            f"val_dice_he={val_metrics['dice_he']:.4f}"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": cfg,
            },
            ckpt_dir / "last_model.pt",
        )

        if val_metrics["dice_mean"] > best_val_dice:
            best_val_dice = val_metrics["dice_mean"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "config": cfg,
                    "best_val_dice": best_val_dice,
                },
                ckpt_dir / "best_model.pt",
            )
            logger.info(f"Saved new best model with val_dice={best_val_dice:.4f}")

        with open(output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    logger.info("Training completed")


if __name__ == "__main__":
    main()