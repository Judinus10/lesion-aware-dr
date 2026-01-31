import torch
import torch.nn as nn
import torch.nn.functional as F


class CBFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss
    Paper: Class-Balanced Loss Based on Effective Number of Samples (CVPR 2019)

    Args:
        samples_per_class (list or tensor): number of samples per class
        num_classes (int)
        beta (float): typically 0.99 or 0.999
        gamma (float): focal loss gamma, typically 2.0
    """

    def __init__(
        self,
        samples_per_class,
        num_classes: int,
        beta: float = 0.999,
        gamma: float = 2.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.beta = beta
        self.gamma = gamma

        samples_per_class = torch.tensor(samples_per_class, dtype=torch.float32)

        # Effective number of samples
        effective_num = 1.0 - torch.pow(self.beta, samples_per_class)
        weights = (1.0 - self.beta) / effective_num
        weights = weights / weights.sum() * num_classes

        self.register_buffer("class_weights", weights)

    def forward(self, logits, targets):
        """
        logits: (B, C)
        targets: (B,)
        """
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            reduction="none",
        )

        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        return focal_loss.mean()
