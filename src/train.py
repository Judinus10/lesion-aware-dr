import argparse
from pathlib import Path

from tqdm import tqdm
import time


import yaml
from omegaconf import OmegaConf

import torch
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.models import build_model
from src.data.dr_datamodule import DRDataModule
from src.losses.classifier import get_loss

try:
    from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
    _HAS_TORCHMETRICS = True
except Exception:
    _HAS_TORCHMETRICS = False


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
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1) Load config
    cfg = load_config(args.cfg_path)

    # Ensure lr & weight_decay are floats
    cfg.training.lr = float(cfg.training.lr)
    cfg.training.weight_decay = float(cfg.training.weight_decay)

    # 2) Create output directories
    Path(cfg.paths.outputs_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.checkpoints_dir).mkdir(parents=True, exist_ok=True)

    # 3) Seed
    set_seed(cfg.training.seed)

    # 4) Logger
    logger = get_logger("train")
    logger.info("Starting training script...")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 5) Optional: WandB
    if cfg.logging.use_wandb:
        import wandb

        wandb.init(
            project=cfg.logging.wandb_project,
            name=cfg.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        logger.info("WandB initialized.")
    else:
        wandb = None

    # 6) Data
    datamodule = DRDataModule(cfg)
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    # 7) Model, loss, optimizer, scheduler
    model = build_model(
        backbone=cfg.model.backbone,
        num_classes=cfg.model.num_classes,
        pretrained=cfg.model.pretrained,
    ).to(device)

    criterion = get_loss(cfg)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    if cfg.training.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.training.epochs)
    else:
        scheduler = None

    if _HAS_TORCHMETRICS:
        acc_metric = MulticlassAccuracy(num_classes=cfg.model.num_classes).to(device)
        f1_metric = MulticlassF1Score(
            num_classes=cfg.model.num_classes, average="macro"
        ).to(device)
    else:
        acc_metric = f1_metric = None
        logger.warning("torchmetrics not installed: val_acc/val_f1 will be 0.0")

    # 8) Training loop
    best_val_loss = float("inf")

    for epoch in range(1, cfg.training.epochs + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch}/{cfg.training.epochs} [TRAIN]")

        start_t = time.time()
        for batch in pbar:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            # update progress bar
            pbar.set_postfix(loss=f"{loss.item():.4f}", elapsed=f"{(time.time()-start_t)/60:.1f}m")


        epoch_train_loss = running_loss / len(train_loader.dataset)

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        if acc_metric is not None:
            acc_metric.reset()
            f1_metric.reset()

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)

                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item() * images.size(0)

                if acc_metric is not None:
                    preds = torch.argmax(logits, dim=1)
                    acc_metric.update(preds, labels)
                    f1_metric.update(preds, labels)

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
            f"train_loss={epoch_train_loss:.4f}  "
            f"val_loss={epoch_val_loss:.4f}  "
            f"val_acc={val_acc:.4f}  val_f1={val_f1:.4f}"
        )

        if wandb is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": epoch_train_loss,
                    "val_loss": epoch_val_loss,
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

        # Save best
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            ckpt_path = Path(cfg.paths.checkpoints_dir) / "best_model.pt"
            torch.save({"epoch": epoch, "model_state": model.state_dict()}, ckpt_path)
            logger.info(f"New best model saved to {ckpt_path}")

    if wandb is not None:
        wandb.finish()
        logger.info("WandB run finished.")


if __name__ == "__main__":
    main()
