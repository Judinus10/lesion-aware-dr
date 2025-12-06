from typing import Callable
import torch
from torch import nn

try:
    from .cb_focal_loss import CBFocalLoss  # you can implement later
    _HAS_CB = True
except Exception:
    _HAS_CB = False


def get_loss(cfg) -> Callable:
    """
    Return a loss function based on cfg.loss.name (default: 'ce').
    Supports: 'ce', 'focal', 'cb_focal' (when implemented).
    """
    loss_cfg = getattr(cfg, "loss", None)
    loss_name = getattr(loss_cfg, "name", "ce") if loss_cfg is not None else "ce"
    loss_name = loss_name.lower()

    if loss_name == "ce":
        return nn.CrossEntropyLoss()

    elif loss_name == "focal":
        gamma = getattr(loss_cfg, "gamma", 2.0) if loss_cfg is not None else 2.0
        alpha = getattr(loss_cfg, "alpha", None) if loss_cfg is not None else None

        class FocalLoss(nn.Module):
            def __init__(self, gamma: float, alpha=None):
                super().__init__()
                self.gamma = gamma
                self.alpha = alpha

            def forward(self, logits, targets):
                ce = nn.functional.cross_entropy(logits, targets, reduction="none")
                pt = torch.exp(-ce)
                loss = ((1 - pt) ** self.gamma) * ce
                if self.alpha is not None:
                    at = self.alpha[targets]
                    loss = at * loss
                return loss.mean()

        return FocalLoss(gamma, alpha)

    elif loss_name == "cb_focal":
        if not _HAS_CB:
            raise ImportError(
                "CBFocalLoss requested but cb_focal_loss.py is not implemented/importable."
            )
        return CBFocalLoss(cfg)

    else:
        raise ValueError(f"Unknown loss type: {loss_name}")
