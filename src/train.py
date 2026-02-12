import argparse
from pathlib import Path

import yaml
from omegaconf import OmegaConf

import torch
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from sklearn.metrics import accuracy_score, f1_score

from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.models import build_model
from src.data.dr_datamodule import DRDataModule
from src.losses.classifier import get_loss  # ✅ single source of truth


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
# Main
# -------------------------
def main():
    args = parse_args()
    cfg = load_config(args.cfg_path)

    cfg.training.lr = float(cfg.training.lr)
    cfg.training.weight_decay = float(cfg.training.weight_decay)

    # dirs
    Path(cfg.paths.outputs_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.checkpoints_dir).mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cfg.paths.checkpoints_dir)

    # seed
    set_seed(cfg.training.seed)

    # logger
    logger = get_logger("train")
    logger.info("Starting training script")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

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
    # Model
    # -------------------------
    model = build_model(
        backbone=cfg.model.backbone,
        num_classes=cfg.model.num_classes,
        pretrained=cfg.model.pretrained,
    ).to(device)

    # -------------------------
    # Loss (ce / focal / cb_focal handled here)
    # -------------------------
    criterion = get_loss(cfg, device=device)

    # -------------------------
    # Optimizer / scheduler
    # -------------------------
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    scheduler = (
        CosineAnnealingLR(optimizer, T_max=int(cfg.training.epochs))
        if str(cfg.training.scheduler).lower() == "cosine"
        else None
    )

    # -------------------------
    # Training loop
    # -------------------------
    best_val_f1 = -1.0
    best_val_loss = float("inf")

    for epoch in range(1, int(cfg.training.epochs) + 1):

        # -------- TRAIN --------
        model.train()
        train_loss_sum = 0.0

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

            train_loss_sum += loss.item() * images.size(0)

            train_pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        epoch_train_loss = train_loss_sum / len(train_loader.dataset)

        # -------- VALID --------
        model.eval()
        val_loss_sum = 0.0

        all_preds = []
        all_labels = []

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

                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.detach().cpu())
                all_labels.append(labels.detach().cpu())

                val_pbar.set_postfix(val_loss=f"{loss.item():.4f}")

        epoch_val_loss = val_loss_sum / len(val_loader.dataset)

        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_labels).numpy()

        val_acc = float(accuracy_score(y_true, y_pred))
        val_f1 = float(f1_score(y_true, y_pred, average="macro"))

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

        # Always save last
        torch.save(
            {"epoch": epoch, "model_state": model.state_dict()},
            ckpt_dir / "last_model.pt",
        )

        # Save best by macro F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict()},
                ckpt_dir / "best_macro_f1_model.pt",
            )
            logger.info(f"✅ Saved best macro-F1 model (val_f1={best_val_f1:.4f})")

        # Save best by val loss
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict()},
                ckpt_dir / "best_val_loss_model.pt",
            )
            logger.info(f"✅ Saved best val-loss model (val_loss={best_val_loss:.4f})")

    logger.info("Training completed")


if __name__ == "__main__":
    main()
