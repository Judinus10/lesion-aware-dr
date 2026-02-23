from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

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
    parser.add_argument("--cfg_path", type=str, default="configs/base.yaml")

    # ✅ resume support
    parser.add_argument(
        "--resume_ckpt",
        type=str,
        default=None,
        help="Path to checkpoint (.pt) to resume from (e.g., outputs/2026-02-23/checkpoints/last_model.pt)",
    )

    # ✅ mid-epoch save to avoid losing progress on Colab timeout
    parser.add_argument(
        "--save_every_steps",
        type=int,
        default=0,
        help="If >0, save a checkpoint every N training steps (batches).",
    )

    # ✅ mixed precision (faster on T4)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable torch.cuda.amp mixed precision training.",
    )

    # ✅ gradient accumulation (simulate larger batch size)
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=1,
        help="Accumulate gradients for N steps before optimizer.step().",
    )

    # ✅ early stopping on macro-F1
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=3,
        help="Stop if macro-F1 does not improve for this many epochs. Set 0 to disable.",
    )
    parser.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=0.0,
        help="Minimum macro-F1 improvement to reset patience (e.g., 0.001).",
    )

    # ✅ optional run folder name override (keeps your date structure but allows custom)
    parser.add_argument(
        "--run_folder",
        type=str,
        default=None,
        help="Optional folder name under outputs/ (e.g., 2026-02-23_ce_run1). If not set, uses today's date YYYY-MM-DD.",
    )

    return parser.parse_args()


# -------------------------
# Checkpoint utils
# -------------------------
def save_checkpoint(
    path: Path,
    epoch: int,
    step_in_epoch: int,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    best_val_f1: float,
    best_val_loss: float,
    cfg: OmegaConf,
) -> None:
    state: Dict[str, Any] = {
        "epoch": int(epoch),
        "step_in_epoch": int(step_in_epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_f1": float(best_val_f1),
        "best_val_loss": float(best_val_loss),
        "cfg": OmegaConf.to_container(cfg, resolve=True),
    }
    if scheduler is not None:
        state["scheduler_state"] = scheduler.state_dict()

    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: Optional[torch.device] = None,
):
    ckpt = torch.load(path, map_location=device if device is not None else "cpu")
    model.load_state_dict(ckpt["model_state"], strict=True)

    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])

    if scheduler is not None and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    step_in_epoch = int(ckpt.get("step_in_epoch", 0))
    best_val_f1 = float(ckpt.get("best_val_f1", -1.0))
    best_val_loss = float(ckpt.get("best_val_loss", float("inf")))

    return start_epoch, step_in_epoch, best_val_f1, best_val_loss


def save_cfg_snapshot(cfg: OmegaConf, run_dir: Path) -> None:
    """Save the resolved config used for this run (so you can compare later)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    snap_path = run_dir / "config_used.yaml"
    with open(snap_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))


# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()
    cfg = load_config(args.cfg_path)

    cfg.training.lr = float(cfg.training.lr)
    cfg.training.weight_decay = float(cfg.training.weight_decay)

    # logger (create early)
    logger = get_logger("train")
    logger.info("Starting training script")

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # -------------------------
    # Output folders (DATED RUN)
    # -------------------------
    # If resuming: keep the folder of the checkpoint
    # If not resuming: create outputs/<YYYY-MM-DD>/checkpoints/
    base_outputs = Path(cfg.paths.outputs_dir)

    if args.resume_ckpt:
        resume_path = Path(args.resume_ckpt)
        ckpt_dir = resume_path.parent
        run_dir = ckpt_dir.parent
        logger.info(f"[RUN DIR] Resuming inside existing run folder: {run_dir}")
    else:
        date_folder = args.run_folder or datetime.now().strftime("%Y-%m-%d")
        # auto-increment if folder already exists: 2026-02-23, 2026-02-23_2, 2026-02-23_3...
        run_dir = base_outputs / date_folder
        if run_dir.exists():
            i = 2
            while (base_outputs / f"{date_folder}_{i}").exists():
                i += 1
            run_dir = base_outputs / f"{date_folder}_{i}"

        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[RUN DIR] New run folder: {run_dir}")
        logger.info(f"[RUN DIR] Checkpoints folder: {ckpt_dir}")

    # overwrite cfg paths so anything else that reads cfg uses THIS run folder
    cfg.paths.outputs_dir = str(run_dir)
    cfg.paths.checkpoints_dir = str(ckpt_dir)

    # save config snapshot for later comparison
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    save_cfg_snapshot(cfg, run_dir)

    # seed
    set_seed(cfg.training.seed)

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

    # Loss (ce / focal / cb_focal)
    criterion = get_loss(cfg, device=device)

    # Optimizer / scheduler
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

    # AMP
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # -------------------------
    # Resume state (NO global-best comparison)
    # -------------------------
    best_val_f1 = -1.0
    best_val_loss = float("inf")
    start_epoch = 1
    resume_step_in_epoch = 0

    if args.resume_ckpt:
        logger.info(f"[RESUME] Loading checkpoint: {args.resume_ckpt}")
        start_epoch, resume_step_in_epoch, best_val_f1, best_val_loss = load_checkpoint(
            args.resume_ckpt, model, optimizer=optimizer, scheduler=scheduler, device=device
        )
        logger.info(
            f"[RESUME] start_epoch={start_epoch}, resume_step_in_epoch={resume_step_in_epoch}, "
            f"best_val_f1={best_val_f1:.4f}, best_val_loss={best_val_loss:.4f}"
        )

    # -------------------------
    # Training loop
    # -------------------------
    total_epochs = int(cfg.training.epochs)
    grad_accum = max(1, int(args.grad_accum))
    save_every_steps = max(0, int(args.save_every_steps))

    # early stopping state (you can disable by setting patience=0 or commenting the block below)
    patience = max(0, int(args.early_stop_patience))
    min_delta = float(args.early_stop_min_delta)
    bad_epochs = 0

    for epoch in range(start_epoch, total_epochs + 1):

        # -------- TRAIN --------
        model.train()
        train_loss_sum = 0.0

        train_pbar = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"Epoch {epoch}/{total_epochs} [TRAIN]",
            dynamic_ncols=True,
        )

        optimizer.zero_grad(set_to_none=True)

        for step_idx, batch in enumerate(train_pbar, start=1):
            # if resuming mid-epoch, skip steps already done
            if epoch == start_epoch and resume_step_in_epoch > 0 and step_idx <= resume_step_in_epoch:
                continue

            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
                loss_to_backprop = loss / grad_accum

            scaler.scale(loss_to_backprop).backward()

            # step optimizer every grad_accum steps
            if step_idx % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            train_loss_sum += loss.item() * images.size(0)

            train_pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

            # ✅ mid-epoch checkpoint
            if save_every_steps > 0 and (step_idx % save_every_steps == 0):
                tmp_path = ckpt_dir / "last_model.pt"
                save_checkpoint(
                    tmp_path,
                    epoch=epoch,
                    step_in_epoch=step_idx,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    best_val_f1=best_val_f1,
                    best_val_loss=best_val_loss,
                    cfg=cfg,
                )
                logger.info(f"[CKPT] Saved mid-epoch checkpoint at epoch={epoch}, step={step_idx}")

        # flush if last steps not divisible by grad_accum
        if (len(train_loader) % grad_accum) != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        epoch_train_loss = train_loss_sum / len(train_loader.dataset)

        # after first resumed epoch, stop skipping
        resume_step_in_epoch = 0

        # -------- VALID --------
        model.eval()
        val_loss_sum = 0.0
        all_preds = []
        all_labels = []

        val_pbar = tqdm(
            val_loader,
            total=len(val_loader),
            desc=f"Epoch {epoch}/{total_epochs} [VAL]",
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
            f"Epoch [{epoch}/{total_epochs}] "
            f"train_loss={epoch_train_loss:.4f} "
            f"val_loss={epoch_val_loss:.4f} "
            f"val_acc={val_acc:.4f} "
            f"val_f1={val_f1:.4f} "
            f"(best_f1={best_val_f1:.4f}, best_loss={best_val_loss:.4f})"
        )

        # Always save last (FULL STATE ✅) into THIS run folder
        save_checkpoint(
            ckpt_dir / "last_model.pt",
            epoch=epoch,
            step_in_epoch=0,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_val_f1=best_val_f1,
            best_val_loss=best_val_loss,
            cfg=cfg,
        )

        # Save best by macro F1 (within this run only ✅)
        improved_f1 = val_f1 > (best_val_f1 + min_delta)
        if improved_f1:
            best_val_f1 = val_f1
            save_checkpoint(
                ckpt_dir / "best_macro_f1_model.pt",
                epoch=epoch,
                step_in_epoch=0,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_val_f1=best_val_f1,
                best_val_loss=best_val_loss,
                cfg=cfg,
            )
            logger.info(f"✅ Saved best macro-F1 model for THIS RUN (val_f1={best_val_f1:.4f})")
            bad_epochs = 0
        else:
            bad_epochs += 1

        # Save best by val loss (within this run only ✅)
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            save_checkpoint(
                ckpt_dir / "best_val_loss_model.pt",
                epoch=epoch,
                step_in_epoch=0,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_val_f1=best_val_f1,
                best_val_loss=best_val_loss,
                cfg=cfg,
            )
            logger.info(f"✅ Saved best val-loss model for THIS RUN (val_loss={best_val_loss:.4f})")

        # ✅ EARLY STOPPING (macro-F1)  -> you already commented it, keep it commented if you want
        # if patience > 0 and bad_epochs >= patience:
        #     logger.info(
        #         f"🛑 Early stopping triggered: no macro-F1 improvement for {bad_epochs} epoch(s). "
        #         f"Best macro-F1={best_val_f1:.4f}"
        #     )
        #     break

    logger.info(f"Training completed. Run saved at: {run_dir}")


if __name__ == "__main__":
    main()