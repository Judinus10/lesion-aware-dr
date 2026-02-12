from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn


def _counts_from_csv(csv_path: str, label_col: str, num_classes: int) -> np.ndarray:
    df = pd.read_csv(csv_path)
    labels = df[label_col].astype(int).values
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)  # avoid zeros
    return counts


def _alpha_from_counts(
    counts: np.ndarray,
    mode: str = "effective",   # "effective" | "inverse" | "none"
    beta: float = 0.9999,
) -> np.ndarray:
    mode = (mode or "effective").lower()

    if mode == "none":
        return np.ones_like(counts, dtype=np.float64)

    if mode == "inverse":
        alpha = 1.0 / counts

    else:
        # "effective" (Cui et al.) with effective number of samples
        # alpha_c = (1 - beta) / (1 - beta^n_c)
        effective_num = 1.0 - np.power(beta, counts)
        alpha = (1.0 - beta) / np.maximum(effective_num, 1e-12)

    # normalize alpha to mean=1 for stability
    alpha = alpha / np.mean(alpha)
    return alpha.astype(np.float32)


class CBFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss (multi-class).

    Uses:
      - alpha (class weights) derived from label counts (inverse or effective number)
      - gamma focal focusing parameter

    Forward:
      - compute CE per sample
      - apply focal term (1-pt)^gamma
      - multiply by alpha[target_class]
    """

    def __init__(
        self,
        num_classes: int,
        alpha: Optional[torch.Tensor] = None,   # Tensor [C] on device, or None
        gamma: float = 2.0,
        reduction: str = "mean",               # "mean" | "sum" | "none"
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.gamma = float(gamma)
        self.reduction = str(reduction).lower()

        if alpha is not None:
            if alpha.ndim != 1 or alpha.shape[0] != self.num_classes:
                raise ValueError(f"alpha must be shape [C={self.num_classes}], got {tuple(alpha.shape)}")
        self.register_buffer("alpha", alpha if alpha is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [N, C], targets: [N]
        ce = nn.functional.cross_entropy(logits, targets, reduction="none")  # [N]
        pt = torch.exp(-ce)  # [N]
        focal = (1.0 - pt) ** self.gamma  # [N]
        loss = focal * ce  # [N]

        if self.alpha is not None:
            at = self.alpha[targets]  # [N]
            loss = at * loss

        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()
        # default mean
        return loss.mean()


def build_cb_focal_from_cfg(cfg, device: torch.device) -> CBFocalLoss:
    """
    Build CBFocalLoss from your OmegaConf cfg.

    Expected cfg.loss:
      name: "cb_focal"
      beta: 0.9999
      gamma: 2.0
      alpha_mode: "effective" | "inverse" | "none"
      reduction: "mean" | "sum" | "none"
    """
    num_classes = int(cfg.model.num_classes)

    loss_cfg = getattr(cfg, "loss", None)
    beta = float(getattr(loss_cfg, "beta", 0.9999)) if loss_cfg is not None else 0.9999
    gamma = float(getattr(loss_cfg, "gamma", 2.0)) if loss_cfg is not None else 2.0
    alpha_mode = str(getattr(loss_cfg, "alpha_mode", "effective")).lower() if loss_cfg is not None else "effective"
    reduction = str(getattr(loss_cfg, "reduction", "mean")).lower() if loss_cfg is not None else "mean"

    # compute counts from training csv
    counts = _counts_from_csv(
        csv_path=str(cfg.data.train_csv),
        label_col=str(cfg.data.label_col),
        num_classes=num_classes,
    )

    alpha_np = _alpha_from_counts(counts, mode=alpha_mode, beta=beta)
    alpha_t = torch.tensor(alpha_np, dtype=torch.float32, device=device)

    return CBFocalLoss(
        num_classes=num_classes,
        alpha=alpha_t if alpha_mode != "none" else None,
        gamma=gamma,
        reduction=reduction,
    )
