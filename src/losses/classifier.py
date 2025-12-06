# src/losses/classifier.py
from typing import Callable
import torch
from torch import nn

from .cb_focal_loss import CBFocalLoss  # you’ll implement this


def get_loss(cfg) -> Callable:
    """Return a loss function based on cfg.loss.name (default: ce)."""
    loss_name = getattr(cfg, "loss", {}).get("name", "ce") if hasattr(cfg, "loss") else "ce"
    loss_name = loss_name.lower()

    if loss_name == "ce":
        return nn.CrossEntropyLoss()
    elif loss_name == "focal":
        # basic focal loss example (you can customise)
        gamma = getattr(cfg.loss, "gamma", 2.0)
        alpha = getattr(cfg.loss, "alpha", None)

        class FocalLoss(nn.Module):
            def __init__(self, gamma, alpha=None):
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
        # your class-balanced focal loss
        return CBFocalLoss(cfg)
    else:
        raise ValueError(f"Unknown loss type: {loss_name}")
