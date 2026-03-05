from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import random

import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ---------- Dummy dataset for quick testing ----------

class DummyDRDataset(Dataset):
    """
    Returns random images & labels.
    Use only when cfg.data.use_dummy = true.
    """
    def __init__(self, num_samples: int, num_classes: int, image_size: int):
        self.num_samples = int(num_samples)
        self.num_classes = int(num_classes)
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image = torch.rand(3, self.image_size, self.image_size)
        label = torch.randint(0, self.num_classes, (1,)).item()
        return {"image": image, "label": torch.tensor(label, dtype=torch.long)}


# ---------- Real dataset for APTOS / EyePACS / etc. ----------

class DRDataset(Dataset):
    """
    Generic DR dataset reading from CSV:
      - image filename column (e.g., '16028_left.png')
      - integer label column (0..num_classes-1)

    Robust to corrupted images: retries a few times and then errors clearly.
    """

    def __init__(
        self,
        csv_path: str,
        images_dir: str,
        image_col: str,
        label_col: str,
        image_size: int = 224,
        augment: bool = False,
        max_decode_retries: int = 20,
        log_bad_every: int = 50,
    ):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.image_col = str(image_col)
        self.label_col = str(label_col)
        self.image_size = int(image_size)

        self.max_decode_retries = int(max_decode_retries)
        self.log_bad_every = int(log_bad_every)
        self.bad_count = 0

        # ✅ EfficientNet pretrained expects ImageNet normalization
        imagenet_mean = (0.485, 0.456, 0.406)
        imagenet_std = (0.229, 0.224, 0.225)

        if augment:
            self.transform = A.Compose(
                [
                    A.Resize(image_size, image_size),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.4),
                    # Replace ShiftScaleRotate with Affine (newer albumentations recommendation)
                    A.Affine(
                        scale=(0.95, 1.05),
                        translate_percent=(0.0, 0.05),
                        rotate=(-15, 15),
                        p=0.5,
                        mode=cv2.BORDER_REFLECT_101,
                    ),
                    A.Normalize(mean=imagenet_mean, std=imagenet_std),
                    ToTensorV2(),
                ]
            )
        else:
            self.transform = A.Compose(
                [
                    A.Resize(image_size, image_size),
                    A.Normalize(mean=imagenet_mean, std=imagenet_std),
                    ToTensorV2(),
                ]
            )

    def __len__(self) -> int:
        return len(self.df)

    def _read_image(self, img_path: Path):
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        for _ in range(self.max_decode_retries):
            row = self.df.iloc[idx]
            fname = str(row[self.image_col])
            img_path = self.images_dir / fname

            image = self._read_image(img_path)

            if image is not None:
                label = int(row[self.label_col])
                augmented = self.transform(image=image)
                return {
                    "image": augmented["image"],
                    "label": torch.tensor(label, dtype=torch.long),
                }

            # Bad image encountered
            self.bad_count += 1
            if self.log_bad_every > 0 and self.bad_count % self.log_bad_every == 0:
                print(f"[WARN] Skipped {self.bad_count} corrupted images so far. Latest: {img_path}")

            idx = random.randint(0, len(self.df) - 1)

        raise RuntimeError(
            f"Too many unreadable images encountered (>{self.max_decode_retries} retries). "
            f"Check dataset integrity and image_dir. Last attempted: {img_path}"
        )


# ---------- DataModule wrapper ----------

class DRDataModule:
    def __init__(self, cfg):
        self.cfg = cfg
        self.batch_size = int(cfg.training.batch_size)
        self.num_workers = int(cfg.training.num_workers)
        self.num_classes = int(cfg.model.num_classes)
        self.image_size = int(cfg.data.image_size)

        self.use_dummy = bool(cfg.data.get("use_dummy", False))

        # Optional sampler (only if you intentionally enable it in YAML)
        self.use_weighted_sampler = bool(cfg.data.get("use_weighted_sampler", False))
        self.sampler_mode = str(cfg.data.get("sampler_mode", "inverse")).lower()

    def _pin_memory(self) -> bool:
        return torch.cuda.is_available()

    def _build_weighted_sampler_from_csv(self, csv_path: str, label_col: str) -> WeightedRandomSampler:
        df = pd.read_csv(csv_path)
        labels = df[label_col].astype(int).values

        class_counts = np.bincount(labels, minlength=self.num_classes).astype(np.float32)
        class_counts = np.maximum(class_counts, 1.0)

        # inverse-frequency sampling
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels]
        sample_weights = torch.tensor(sample_weights, dtype=torch.double)

        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    # ---- Dataloaders ----
    def train_dataloader(self) -> DataLoader:
        if self.use_dummy:
            ds = DummyDRDataset(
                num_samples=256,
                num_classes=self.num_classes,
                image_size=self.image_size,
            )
            return DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self._pin_memory(),
            )

        ds = DRDataset(
            csv_path=self.cfg.data.train_csv,
            images_dir=self.cfg.data.image_dir,
            image_col=self.cfg.data.image_col,
            label_col=self.cfg.data.label_col,
            image_size=self.image_size,
            augment=True,
            max_decode_retries=30,
            log_bad_every=50,
        )

        if self.use_weighted_sampler:
            sampler = self._build_weighted_sampler_from_csv(
                csv_path=self.cfg.data.train_csv,
                label_col=self.cfg.data.label_col,
            )
            return DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=False,  # sampler + shuffle cannot both be True
                sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=self._pin_memory(),
            )

        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self._pin_memory(),
        )

    def val_dataloader(self) -> DataLoader:
        if self.use_dummy:
            ds = DummyDRDataset(
                num_samples=64,
                num_classes=self.num_classes,
                image_size=self.image_size,
            )
        else:
            ds = DRDataset(
                csv_path=self.cfg.data.val_csv,
                images_dir=self.cfg.data.image_dir,
                image_col=self.cfg.data.image_col,
                label_col=self.cfg.data.label_col,
                image_size=self.image_size,
                augment=False,
                max_decode_retries=30,
                log_bad_every=50,
            )

        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self._pin_memory(),
        )
