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
# Class weights utils
# -------------------------
def compute_class_weights_from_csv(
    csv_path: str,
    label_col: str,
    num_classes: int,
    mode: str = "inverse",  # "inverse" or "effective"
    beta: float = 0.9999,
) -> np.ndarray:
    df = pd.read_csv(csv_path)
    labels = df[label_col].astype(int).values

    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)

    if mode == "effective":
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
    else:
        weights = 1.0 / counts

    weights = weights / weights.mean()
    return weights


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            at = self.alpha[targets]
            loss = at * loss
        return loss.mean()


def build_criterion(cfg: OmegaConf, device: torch.device, logger):
    loss_cfg = getattr(cfg, "loss", None)
    loss_name = getattr(loss_cfg, "name", "ce") if loss_cfg is not None else "ce"
    loss_name = str(loss_name).lower()

    use_class_weights = bool(getattr(loss_cfg, "use_class_weights", False)) if loss_cfg is not None else False
    weight_mode = str(getattr(loss_cfg, "weight_mode", "inverse")).lower() if loss_cfg is not None else "inverse"
    beta = float(getattr(loss_cfg, "beta", 0.9999)) if loss_cfg is not None else 0.9999

    class_weights_t = None
    if use_class_weights:
        w_np = compute_class_weights_from_csv(
            csv_path=cfg.data.train_csv,
            label_col=cfg.data.label_col,
            num_classes=int(cfg.model.num_classes),
            mode=weight_mode,
            beta=beta,
        )
        class_weights_t = torch.tensor(w_np, dtype=torch.float32, device=device)
        logger.info(f"[LOSS] Using class weights (mode={weight_mode}): {w_np.round(4).tolist()}")
    else:
        logger.info("[LOSS] Class weights: OFF")

    if loss_name == "ce":
        logger.info("[LOSS] Using CrossEntropyLoss")
        return nn.CrossEntropyLoss(weight=class_weights_t)

    if loss_name == "focal":
        gamma = float(getattr(loss_cfg, "gamma", 2.0)) if loss_cfg is not None else 2.0
        logger.info(f"[LOSS] Using FocalLoss(gamma={gamma}) alpha={'ON' if class_weights_t is not None else 'OFF'}")
        return FocalLoss(gamma=gamma, alpha=class_weights_t)

    raise ValueError(f"Unknown loss type: {loss_name}")


# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()
    cfg = load_config(args.cfg_path)

    cfg.training.lr = float(cfg.training.lr)
    cfg.training.weight_decay = float(cfg.training.weight_decay)

    Path(cfg.paths.outputs_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.checkpoints_dir).mkdir(parents=True, exist_ok=True)

    set_seed(cfg.training.seed)

    logger = get_logger("train")
    logger.info("Starting training script")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Data
    datamodule = DRDataModule(cfg)
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples:   {len(val_loader.dataset)}")
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches:   {len(val_loader)}")

    # Model
    model = build_model(
        backbone=cfg.model.backbone,
        num_classes=cfg.model.num_classes,
        pretrained=cfg.model.pretrained,
    ).to(device)

    criterion = build_criterion(cfg, device, logger)

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

    if _HAS_TORCHMETRICS:
        acc_metric = MulticlassAccuracy(num_classes=cfg.model.num_classes).to(device)
        f1_metric = MulticlassF1Score(num_classes=cfg.model.num_classes, average="macro").to(device)
    else:
        acc_metric = f1_metric = None
        logger.warning("torchmetrics not installed — metrics disabled")

    best_val_f1 = -1.0
    best_val_loss = float("inf")
    ckpt_dir = Path(cfg.paths.checkpoints_dir)

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
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            train_pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        epoch_train_loss = running_loss / len(train_loader.dataset)

        # -------- VALID --------
        model.eval()
        val_loss_sum = 0.0

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

                logits = model(images)
                loss = criterion(logits, labels)
                val_loss_sum += loss.item() * images.size(0)

                if acc_metric is not None:
                    preds = torch.argmax(logits, dim=1)
                    acc_metric.update(preds, labels)
                    f1_metric.update(preds, labels)

                val_pbar.set_postfix(val_loss=f"{loss.item():.4f}")

        epoch_val_loss = val_loss_sum / len(val_loader.dataset)

        if acc_metric is not None:
            val_acc = float(acc_metric.compute().item())
            val_f1 = float(f1_metric.compute().item())
        else:
            val_acc = 0.0
            val_f1 = 0.0

        if scheduler is not None:
            scheduler.step()

        logger.info(
            f"Epoch [{epoch}/{cfg.training.epochs}] "
            f"train_loss={epoch_train_loss:.4f} "
            f"val_loss={epoch_val_loss:.4f} "
            f"val_acc={val_acc:.4f} "
            f"val_f1={val_f1:.4f} "
            f"(best_f1={best_val_f1:.4f}, best_loss={best_val_loss:.4f})"
        )

        # Always save last (useful for debugging)
        torch.save({"epoch": epoch, "model_state": model.state_dict()}, ckpt_dir / "last_model.pt")

        # Save best by macro F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            ckpt_path = ckpt_dir / "best_macro_f1_model.pt"
            torch.save({"epoch": epoch, "model_state": model.state_dict()}, ckpt_path)
            logger.info(f"✅ Saved best macro-F1 model → {ckpt_path} (val_f1={best_val_f1:.4f})")

        # Save best by val loss
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            ckpt_path = ckpt_dir / "best_val_loss_model.pt"
            torch.save({"epoch": epoch, "model_state": model.state_dict()}, ckpt_path)
            logger.info(f"✅ Saved best val-loss model → {ckpt_path} (val_loss={best_val_loss:.4f})")

    logger.info("Training completed")


if __name__ == "__main__":
    main()
