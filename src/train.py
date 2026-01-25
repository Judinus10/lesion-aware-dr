import argparse
from pathlib import Path

import yaml
from omegaconf import OmegaConf

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from tqdm import tqdm

from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.models import build_model
from src.data.dr_datamodule import DRDataModule

try:
    from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
    _HAS_TORCHMETRICS = True
except Exception:
    _HAS_TORCHMETRICS = False

from torch.cuda.amp import autocast, GradScaler


# -------------------------
# Config utils
# -------------------------
def load_config(cfg_path: str) -> OmegaConf:
    with open(cfg_path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    return OmegaConf.create(cfg_dict)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg_path",
        type=str,
        default="configs/base.yaml",
        help="Path to YAML config file",
    )
    return parser.parse_args()


# -------------------------
# Loss helpers
# -------------------------
def compute_class_weights(train_csv: str, label_col: str, num_classes: int) -> torch.Tensor:
    """
    Returns normalized inverse-frequency weights of shape [num_classes].
    """
    df = pd.read_csv(train_csv)
    counts = df[label_col].value_counts().sort_index()
    counts = counts.reindex(range(num_classes), fill_value=0).values.astype(np.float32)

    # avoid divide-by-zero
    counts = np.maximum(counts, 1.0)

    weights = 1.0 / counts
    weights = weights / weights.mean()  # normalize around 1.0
    return torch.tensor(weights, dtype=torch.float32)


class FocalLoss(nn.Module):
    """
    Focal loss for multiclass classification.
    alpha: tensor [C] or None
    gamma: float
    """
    def __init__(self, alpha=None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(
            logits, targets, reduction="none", weight=self.alpha
        )
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()


def build_criterion(cfg: OmegaConf, device: torch.device, logger=None) -> nn.Module:
    """
    Supports:
      loss.name: "ce" or "focal"
      loss.use_class_weights: true/false
      loss.gamma: (for focal, default 2.0)
    """
    loss_cfg = cfg.get("loss", {}) or {}
    loss_name = str(loss_cfg.get("name", "ce")).lower()
    use_w = bool(loss_cfg.get("use_class_weights", False))

    alpha = None
    if use_w:
        # Required config keys
        train_csv = cfg.data.train_csv
        label_col = cfg.data.label_col
        num_classes = int(cfg.model.num_classes)

        alpha = compute_class_weights(train_csv, label_col, num_classes).to(device)
        if logger is not None:
            logger.info(f"[LOSS] Using class weights: {alpha.detach().cpu().numpy()}")

    if loss_name == "focal":
        gamma = float(loss_cfg.get("gamma", 2.0))
        if logger is not None:
            logger.info(f"[LOSS] FocalLoss gamma={gamma}, class_weights={use_w}")
        return FocalLoss(alpha=alpha, gamma=gamma)

    # default: weighted CE or normal CE
    if logger is not None:
        logger.info(f"[LOSS] CrossEntropyLoss, class_weights={use_w}")
    return nn.CrossEntropyLoss(weight=alpha)


# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()
    cfg = load_config(args.cfg_path)

    # Cast numeric config values safely
    cfg.training.lr = float(cfg.training.lr)
    cfg.training.weight_decay = float(cfg.training.weight_decay)

    # dirs
    Path(cfg.paths.outputs_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.checkpoints_dir).mkdir(parents=True, exist_ok=True)

    # seed
    set_seed(cfg.training.seed)

    # logger
    logger = get_logger("train")
    logger.info("Starting training script")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(
            f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )
    else:
        logger.warning("CUDA not available — running on CPU")

    # speed
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # -------------------------
    # Data
    # -------------------------
    datamodule = DRDataModule(cfg)
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples:   {len(val_loader.dataset)}")
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches:   {len(val_loader)}")

    # -------------------------
    # Model / loss / optim
    # -------------------------
    model = build_model(
        backbone=cfg.model.backbone,
        num_classes=cfg.model.num_classes,
        pretrained=cfg.model.pretrained,
    ).to(device)

    # ✅ LOSS: weighted CE / focal controlled by YAML
    criterion = build_criterion(cfg, device, logger=logger)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    scheduler = (
        CosineAnnealingLR(optimizer, T_max=cfg.training.epochs)
        if cfg.training.scheduler == "cosine"
        else None
    )

    # AMP scaler
    scaler = GradScaler(enabled=(device.type == "cuda"))
    logger.info(f"AMP enabled: {scaler.is_enabled()}")

    # metrics
    if _HAS_TORCHMETRICS:
        acc_metric = MulticlassAccuracy(num_classes=cfg.model.num_classes).to(device)
        f1_metric = MulticlassF1Score(
            num_classes=cfg.model.num_classes, average="macro"
        ).to(device)
    else:
        acc_metric = f1_metric = None
        logger.warning("torchmetrics not installed — metrics disabled")

    grad_clip_norm = float(getattr(cfg.training, "grad_clip_norm", 0.0) or 0.0)

    # -------------------------
    # Training loop
    # -------------------------
    best_val_loss = float("inf")

    for epoch in range(1, cfg.training.epochs + 1):

        # -------- TRAIN --------
        model.train()
        running_loss = 0.0

        train_pbar = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"Epoch {epoch}/{cfg.training.epochs} [TRAIN]",
            dynamic_ncols=True,
        )

        for batch in train_pbar:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=(device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()

            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)

            train_pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        epoch_train_loss = running_loss / len(train_loader.dataset)

        # -------- VALIDATION --------
        model.eval()
        val_loss = 0.0

        if acc_metric is not None:
            acc_metric.reset()
            f1_metric.reset()

        val_pbar = tqdm(
            val_loader,
            total=len(val_loader),
            desc=f"Epoch {epoch}/{cfg.training.epochs} [VAL]",
            dynamic_ncols=True,
        )

        with torch.no_grad():
            for batch in val_pbar:
                images = batch["image"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)

                with autocast(enabled=(device.type == "cuda")):
                    logits = model(images)
                    loss = criterion(logits, labels)

                val_loss += loss.item() * images.size(0)

                if acc_metric is not None:
                    preds = torch.argmax(logits, dim=1)
                    acc_metric.update(preds, labels)
                    f1_metric.update(preds, labels)

                val_pbar.set_postfix(val_loss=f"{loss.item():.4f}")

        epoch_val_loss = val_loss / len(val_loader.dataset)

        if acc_metric is not None:
            val_acc = acc_metric.compute().item()
            val_f1 = f1_metric.compute().item()
        else:
            val_acc = val_f1 = 0.0

        if scheduler is not None:
            scheduler.step()

        logger.info(
            f"Epoch [{epoch}/{cfg.training.epochs}] "
            f"train_loss={epoch_train_loss:.4f} "
            f"val_loss={epoch_val_loss:.4f} "
            f"val_acc={val_acc:.4f} "
            f"val_f1={val_f1:.4f}"
        )

        # save best
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            ckpt_path = Path(cfg.paths.checkpoints_dir) / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                },
                ckpt_path,
            )
            logger.info(f"Saved best model → {ckpt_path}")

    logger.info("Training completed")


if __name__ == "__main__":
    main()
