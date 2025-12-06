import timm
import torch.nn as nn


def build_model(backbone: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    model = timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model
