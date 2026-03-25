from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

import torch
from torch import nn

from src.models.cb_focal_loss import build_cb_focal_from_cfg


# -------------------------
# Helpers
# -------------------------
def _read_labels_from_csv(csv_path: str, label_col: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    if label_col not in df.columns:
        raise ValueError(f"label_col='{label_col}' not found in CSV columns: {list(df.columns)}")
    labels = df[label_col].astype(int).values
    return labels


def _compute_class_counts_from_csv(
    csv_path: str,
    label_col: str,
    num_classes: int,
) -> np.ndarray:
    labels = _read_labels_from_csv(csv_path, label_col)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)  # avoid zeros (safety)
    return counts


def _compute_log_priors_from_counts(counts: np.ndarray) -> np.ndarray:
    priors = counts / np.sum(counts)
    log_priors = np.log(priors + 1e-12)
    return log_priors.astype(np.float32)


def _compute_class_weights_from_csv(
    csv_path: str,
    label_col: str,
    num_classes: int,
    mode: str = "inverse",      # "inverse" | "effective"
    beta: float = 0.9999,
) -> np.ndarray:
    counts = _compute_class_counts_from_csv(csv_path, label_col, num_classes)

    mode = (mode or "inverse").lower()
    if mode == "effective":
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / np.maximum(effective_num, 1e-12)
    else:
        weights = 1.0 / counts

    weights = weights / np.mean(weights)  # normalize mean=1
    return weights.astype(np.float32)


# -------------------------
# Losses
# -------------------------
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


class LogitAdjustedCrossEntropy(nn.Module):
    """
    Logit Adjustment:
      logits' = logits + tau * log(p_class)
      loss = CE(logits', y)

    Notes:
      - Use with normal sampling (no WeightedRandomSampler).
      - Do NOT also apply class weights (unless you know exactly why).
    """
    def __init__(self, log_priors: torch.Tensor, tau: float = 1.0):
        super().__init__()
        if log_priors.dim() != 1:
            raise ValueError("log_priors must be a 1D tensor of shape [num_classes]")
        self.register_buffer("log_priors", log_priors)
        self.tau = float(tau)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits_adj = logits + self.tau * self.log_priors
        return nn.functional.cross_entropy(logits_adj, targets)


class BalancedSoftmaxLoss(nn.Module):
    """
    Balanced Softmax:
      logits' = logits + log(n_class)
      loss = CE(logits', y)

    Notes:
      - Use with normal sampling (no WeightedRandomSampler).
      - Do NOT also apply class weights (over-correction).
    """
    def __init__(self, class_counts: torch.Tensor):
        super().__init__()
        if class_counts.dim() != 1:
            raise ValueError("class_counts must be a 1D tensor of shape [num_classes]")
        self.register_buffer("class_counts", class_counts)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits_bal = logits + torch.log(self.class_counts + 1e-12)
        return nn.functional.cross_entropy(logits_bal, targets)


# -------------------------
# Factory
# -------------------------
def get_loss(cfg, device: Optional[torch.device] = None) -> Callable:
    """
    Central loss factory.

    Supports:
      - ce               : CrossEntropyLoss (optionally weighted)
      - focal            : FocalLoss (optionally weighted)
      - cb_focal         : Class-Balanced Focal Loss
      - logit_adjusted_ce: CE with logit adjustment using class priors
      - balanced_softmax : CE with balanced softmax using class counts

    Best practice: pass device so any buffers live on GPU.
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

    # ---- logit_adjusted_ce / balanced_softmax (compute counts/priors from train csv) ----
    if loss_name in {"logit_adjusted_ce", "balanced_softmax"}:
        # Hard rule: don't stack class weights with these unless you really know what you're doing.
        if use_class_weights:
            raise ValueError(
                f"loss.name='{loss_name}' should NOT be combined with use_class_weights=true. "
                f"Disable class weights (and disable WeightedRandomSampler too)."
            )

        num_classes = int(cfg.model.num_classes)
        counts_np = _compute_class_counts_from_csv(
            csv_path=str(cfg.data.train_csv),
            label_col=str(cfg.data.label_col),
            num_classes=num_classes,
        )

        if loss_name == "logit_adjusted_ce":
            tau = float(getattr(loss_cfg, "tau", 1.0)) if loss_cfg is not None else 1.0
            log_priors_np = _compute_log_priors_from_counts(counts_np)
            log_priors_t = torch.tensor(log_priors_np, dtype=torch.float32, device=device)
            return LogitAdjustedCrossEntropy(log_priors=log_priors_t, tau=tau)

        # balanced_softmax
        class_counts_t = torch.tensor(counts_np.astype(np.float32), dtype=torch.float32, device=device)
        return BalancedSoftmaxLoss(class_counts=class_counts_t)

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
        return nn.CrossEntropyLoss(weight=class_weights_t)

    if loss_name == "focal":
        gamma = float(getattr(loss_cfg, "gamma", 2.0)) if loss_cfg is not None else 2.0
        return FocalLoss(gamma=gamma, alpha=class_weights_t, reduction=reduction)

    raise ValueError(f"Unknown loss type: {loss_name}")