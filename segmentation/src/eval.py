from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import SegmentationDataset
from src.losses.segmentation import BCEDiceLoss
from src.models.unet import UNet
from src.utils.logger import get_logger
from src.utils.metrics import compute_batch_metrics


def load_config(cfg_path: str) -> dict:
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, default="configs/base.yaml")
    parser.add_argument("--ckpt_path", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.cfg_path)
    device_name = cfg.get("device", "cuda")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")

    output_dir = Path(cfg["paths"]["outputs_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("seg_eval", str(output_dir / "eval.log"))

    ckpt_path = args.ckpt_path or str(output_dir / "checkpoints" / "best_model.pt")
    logger.info(f"Using checkpoint: {ckpt_path}")

    dataset = SegmentationDataset(
        csv_path=cfg["paths"]["test_csv"],
        image_size=int(cfg["data"]["image_size"]),
        is_train=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["data"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=bool(cfg["data"]["pin_memory"]),
    )

    model = UNet(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        base_channels=cfg["model"]["base_channels"],
    ).to(device)

    criterion = BCEDiceLoss(
        bce_weight=float(cfg["loss"]["bce_weight"]),
        dice_weight=float(cfg["loss"]["dice_weight"]),
        smooth=float(cfg["loss"].get("smooth", 1.0)),
    )

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
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
    count = 0

    threshold = float(cfg["data"]["threshold"])

    with torch.no_grad():
        for images, targets, _ in tqdm(loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, targets)
            metrics = compute_batch_metrics(logits, targets, threshold=threshold)

            total_loss += float(loss.item())
            for k in total_metrics:
                total_metrics[k] += metrics[k]
            count += 1

    results = {
        "test_loss": total_loss / max(1, count),
        **{f"test_{k}": v / max(1, count) for k, v in total_metrics.items()},
    }

    logger.info(json.dumps(results, indent=2))

    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()