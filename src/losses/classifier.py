from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

import torch
from torch import nn

try:
    from .cb_focal_loss import CBFocalLoss  # optional (implement later)
    _HAS_CB = True
except Exception:
    _HAS_CB = False


def _compute_class_weights_from_csv(
    csv_path: str,
    label_col: str,
    num_classes: int,
    mode: str = "inverse",      # "inverse" or "effective"
    beta: float = 0.9999,
) -> np.ndarray:
    """
    Compute class weights based on training label distribution.

    mode:
      - "inverse": w_c = 1 / n_c
      - "effective": Class-Balanced weights via effective number of samples:
          w_c = (1 - beta) / (1 - beta^n_c)

    Returns:
      weights: np.ndarray of shape [num_classes], normalized to mean=1
    """
    df = pd.read_csv(csv_path)
    labels = df[label_col].astype(int).values

    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)  # avoid divide-by-zero

    mode = (mode or "inverse").lower()
    if mode == "effective":
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
    else:
        weights = 1.0 / counts

    # normalize to mean=1 for stable optimization
    weights = weights / np.mean(weights)
    return weights


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss.

    alpha: Optional Tensor [C] class weights (like weighted CE)
    gamma: focusing parameter
    """
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = alpha  # Tensor [C] on correct device, or None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce

        if self.alpha is not None:
            # alpha indexed by class id
            at = self.alpha[targets]
            loss = at * loss

        return loss.mean()


def get_loss(cfg, device: Optional[torch.device] = None) -> Callable:
    """
    Return a loss function based on cfg.loss.name.
    Supports:
      - 'ce'        : CrossEntropy (optionally class-weighted)
      - 'focal'     : FocalLoss (optionally class-weighted via alpha)
      - 'cb_focal'  : Class-Balanced Focal (if cb_focal_loss.py exists)

    IMPORTANT:
    - If cfg.loss.use_class_weights is true, weights are computed from cfg.data.train_csv.
    - device is optional; if not passed, weights will be on CPU (still works, but slower / may error if indexing on GPU).
      Best practice: pass the same device used for the model.
    """
    loss_cfg = getattr(cfg, "loss", None)
    loss_name = getattr(loss_cfg, "name", "ce") if loss_cfg is not None else "ce"
    loss_name = str(loss_name).lower()

    # Optional class weighting
    use_class_weights = bool(getattr(loss_cfg, "use_class_weights", False)) if loss_cfg is not None else False
    weight_mode = str(getattr(loss_cfg, "weight_mode", "inverse")).lower() if loss_cfg is not None else "inverse"
    beta = float(getattr(loss_cfg, "beta", 0.9999)) if loss_cfg is not None else 0.9999

    class_weights_t: Optional[torch.Tensor] = None
    if use_class_weights:
        w_np = _compute_class_weights_from_csv(
            csv_path=cfg.data.train_csv,
            label_col=cfg.data.label_col,
            num_classes=int(cfg.model.num_classes),
            mode=weight_mode,
            beta=beta,
        )
        class_weights_t = torch.tensor(w_np, dtype=torch.float32)
        if device is not None:
            class_weights_t = class_weights_t.to(device)

    # ---- Loss selection ----
    if loss_name == "ce":
        # weighted CE if class_weights_t provided
        return nn.CrossEntropyLoss(weight=class_weights_t)

    elif loss_name == "focal":
        gamma = float(getattr(loss_cfg, "gamma", 2.0)) if loss_cfg is not None else 2.0
        # If user provided explicit alpha in cfg, allow it; otherwise use computed weights
        alpha_cfg = getattr(loss_cfg, "alpha", None) if loss_cfg is not None else None

        if alpha_cfg is not None:
            # allow list or tensor in yaml
            alpha_t = torch.tensor(alpha_cfg, dtype=torch.float32)
            if device is not None:
                alpha_t = alpha_t.to(device)
            return FocalLoss(gamma=gamma, alpha=alpha_t)

        return FocalLoss(gamma=gamma, alpha=class_weights_t)

    elif loss_name == "cb_focal":
        if not _HAS_CB:
            raise ImportError(
                "CBFocalLoss requested but cb_focal_loss.py is not implemented/importable."
            )
        return CBFocalLoss(cfg)

    else:
        raise ValueError(f"Unknown loss type: {loss_name}")
