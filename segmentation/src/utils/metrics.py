from __future__ import annotations

from typing import Dict

import torch


CHANNEL_NAMES = ["ex", "he", "ma", "od"]


def compute_batch_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1.0,
) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.view(preds.size(0), preds.size(1), -1)
    targets = targets.view(targets.size(0), targets.size(1), -1)

    intersection = (preds * targets).sum(dim=2)
    pred_sum = preds.sum(dim=2)
    target_sum = targets.sum(dim=2)
    union = pred_sum + target_sum - intersection

    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
    iou = (intersection + smooth) / (union + smooth)

    dice_mean_per_channel = dice.mean(dim=0)
    iou_mean_per_channel = iou.mean(dim=0)

    results: Dict[str, float] = {}
    for i, name in enumerate(CHANNEL_NAMES):
        results[f"dice_{name}"] = float(dice_mean_per_channel[i].item())
        results[f"iou_{name}"] = float(iou_mean_per_channel[i].item())

    results["dice_mean"] = float(dice.mean().item())
    results["iou_mean"] = float(iou.mean().item())

    return results