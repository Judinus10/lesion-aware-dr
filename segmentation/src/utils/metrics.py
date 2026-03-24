from __future__ import annotations

from typing import Dict

import torch


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

    dice_mean = dice.mean(dim=0)
    iou_mean = iou.mean(dim=0)

    return {
        "dice_ex": float(dice_mean[0].item()),
        "dice_he": float(dice_mean[1].item()),
        "dice_mean": float(dice.mean().item()),
        "iou_ex": float(iou_mean[0].item()),
        "iou_he": float(iou_mean[1].item()),
        "iou_mean": float(iou.mean().item()),
    }