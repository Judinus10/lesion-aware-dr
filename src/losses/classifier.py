from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

import torch
from torch import nn

from src.models.cb_focal_loss import build_cb_focal_from_cfg


def _compute_class_weights_from_csv(
    csv_path: str,
    label_col: str,
    num_classes: int,
    mode: str = "inverse",      # "inverse" | "effective"
    beta: float = 0.9999,
) -> np.ndarray:
    df = pd.read_csv(csv_path)
    labels = df[label_col].astype(int).values

    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)

    mode = (mode or "inverse").lower()
    if mode == "effective":
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / np.maximum(effective_num, 1e-12)
    else:
        weights = 1.0 / counts

    weights = weights / np.mean(weights)  # normalize mean=1
    return weights.astype(np.float32)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = alpha
        self.reduction = str(reduction).lower()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce

        if self.alpha is not None:
            loss = self.alpha[targets] * loss

        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()
        return loss.mean()


def get_loss(cfg, device: Optional[torch.device] = None) -> Callable:
    """
    Central loss factory.

    Supports:
      - ce       : CrossEntropyLoss (optionally weighted)
      - focal    : FocalLoss (optionally weighted)
      - cb_focal : Class-Balanced Focal Loss (alpha from counts, focal gamma)

    Best practice: pass device so weights live on GPU.
    """
    if device is None:
        device = torch.device("cpu")

    loss_cfg = getattr(cfg, "loss", None)
    loss_name = str(getattr(loss_cfg, "name", "ce")).lower() if loss_cfg is not None else "ce"

    # shared fields (only used for ce/focal)
    use_class_weights = bool(getattr(loss_cfg, "use_class_weights", False)) if loss_cfg is not None else False
    weight_mode = str(getattr(loss_cfg, "weight_mode", "effective")).lower() if loss_cfg is not None else "effective"
    beta = float(getattr(loss_cfg, "beta", 0.9999)) if loss_cfg is not None else 0.9999
    reduction = str(getattr(loss_cfg, "reduction", "mean")).lower() if loss_cfg is not None else "mean"

    # ---- cb_focal ----
    if loss_name == "cb_focal":
        return build_cb_focal_from_cfg(cfg, device=device)

    # ---- optional weights for ce/focal ----
    class_weights_t: Optional[torch.Tensor] = None
    if use_class_weights:
        w_np = _compute_class_weights_from_csv(
            csv_path=str(cfg.data.train_csv),
            label_col=str(cfg.data.label_col),
            num_classes=int(cfg.model.num_classes),
            mode=weight_mode,
            beta=beta,
        )
        class_weights_t = torch.tensor(w_np, dtype=torch.float32, device=device)

    if loss_name == "ce":
        # NOTE: torch CE supports weight but not reduction="none"/"sum" config in same way.
        # We keep default mean unless you want more control.
        return nn.CrossEntropyLoss(weight=class_weights_t)

    if loss_name == "focal":
        gamma = float(getattr(loss_cfg, "gamma", 2.0)) if loss_cfg is not None else 2.0
        return FocalLoss(gamma=gamma, alpha=class_weights_t, reduction=reduction)

    raise ValueError(f"Unknown loss type: {loss_name}")
