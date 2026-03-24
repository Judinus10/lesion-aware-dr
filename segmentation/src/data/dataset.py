from __future__ import annotations

from pathlib import Path
from typing import Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def build_transforms(image_size: int, is_train: bool):
    if is_train:
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Rotate(limit=15, p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


class SegmentationDataset(Dataset):
    def __init__(self, csv_path: str, image_size: int = 512, is_train: bool = True):
        self.df = pd.read_csv(csv_path)
        self.transforms = build_transforms(image_size=image_size, is_train=is_train)

        required_cols = ["image_path", "ex_mask_path", "he_mask_path"]
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns in CSV: {missing}")

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def _read_image(path: str) -> np.ndarray:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    @staticmethod
    def _read_mask(path: str) -> np.ndarray:
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {path}")
        mask = (mask > 0).astype(np.float32)
        return mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        row = self.df.iloc[idx]

        image = self._read_image(row["image_path"])
        ex_mask = self._read_mask(row["ex_mask_path"])
        he_mask = self._read_mask(row["he_mask_path"])

        transformed = self.transforms(image=image, masks=[ex_mask, he_mask])
        image = transformed["image"]
        ex_mask, he_mask = transformed["masks"]

        target = np.stack([ex_mask, he_mask], axis=0).astype(np.float32)
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        image_tensor = torch.from_numpy(image)
        target_tensor = torch.from_numpy(target)
        image_id = str(row.get("image_id", Path(row["image_path"]).stem))

        return image_tensor, target_tensor, image_id