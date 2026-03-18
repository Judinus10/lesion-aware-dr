from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2


def _read_rgb_image(image_path: str, image_size: int) -> np.ndarray:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    return image


def _read_binary_mask(mask_path: str | float | None, image_size: int) -> np.ndarray:
    if mask_path is None or (isinstance(mask_path, float) and np.isnan(mask_path)) or str(mask_path).strip() == "":
        return np.zeros((image_size, image_size), dtype=np.uint8)

    mask_path = str(mask_path)
    if not Path(mask_path).exists():
        return np.zeros((image_size, image_size), dtype=np.uint8)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.zeros((image_size, image_size), dtype=np.uint8)

    mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 127).astype(np.uint8)
    return mask


class DRSegmentationDataset(Dataset):
    def __init__(self, csv_path: str, image_size: int, is_train: bool = True):
        self.df = pd.read_csv(csv_path)
        self.image_size = int(image_size)
        self.is_train = bool(is_train)

        if self.is_train:
            self.transforms = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.03,
                    scale_limit=0.08,
                    rotate_limit=15,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.5,
                ),
                A.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.15,
                    hue=0.05,
                    p=0.4,
                ),
                A.Normalize(),
                ToTensorV2(),
            ])
        else:
            self.transforms = A.Compose([
                A.Normalize(),
                ToTensorV2(),
            ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        image_path = str(row["image_path"])
        exudate_mask_path = row.get("exudate_mask_path", "")
        hemorrhage_mask_path = row.get("hemorrhage_mask_path", "")

        image = _read_rgb_image(image_path, self.image_size)
        ex_mask = _read_binary_mask(exudate_mask_path, self.image_size)
        he_mask = _read_binary_mask(hemorrhage_mask_path, self.image_size)

        mask = np.stack([ex_mask, he_mask], axis=-1).astype(np.float32)

        transformed = self.transforms(image=image, mask=mask)
        image_t = transformed["image"]
        mask_t = transformed["mask"].permute(2, 0, 1).float()

        return {
            "image": image_t,
            "mask": mask_t,
            "image_path": image_path,
        }


class DRDataModule:
    def __init__(self, cfg):
        self.cfg = cfg

        self.train_dataset = DRSegmentationDataset(
            csv_path=str(cfg.paths.train_csv),
            image_size=int(cfg.data.image_size),
            is_train=True,
        )
        self.val_dataset = DRSegmentationDataset(
            csv_path=str(cfg.paths.val_csv),
            image_size=int(cfg.data.image_size),
            is_train=False,
        )
        self.test_dataset = DRSegmentationDataset(
            csv_path=str(cfg.paths.test_csv),
            image_size=int(cfg.data.image_size),
            is_train=False,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=int(self.cfg.data.batch_size),
            shuffle=True,
            num_workers=int(self.cfg.data.num_workers),
            pin_memory=bool(self.cfg.data.pin_memory),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=int(self.cfg.data.batch_size),
            shuffle=False,
            num_workers=int(self.cfg.data.num_workers),
            pin_memory=bool(self.cfg.data.pin_memory),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=int(self.cfg.data.batch_size),
            shuffle=False,
            num_workers=int(self.cfg.data.num_workers),
            pin_memory=bool(self.cfg.data.pin_memory),
        )