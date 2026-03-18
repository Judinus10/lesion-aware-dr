from __future__ import annotations

from segmentation.src.models.unet import UNet


def build_model(cfg):
    model_name = str(cfg.model.name).lower()

    if model_name == "unet":
        return UNet(
            in_channels=int(cfg.model.in_channels),
            out_channels=int(cfg.model.out_channels),
            base_channels=int(cfg.model.base_channels),
        )

    raise ValueError(f"Unsupported model name: {cfg.model.name}")