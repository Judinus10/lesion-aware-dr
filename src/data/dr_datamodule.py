from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional
import random

import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import albumentations as A
from albumentations.pytorch import ToTensorV2


# =========================================================
# Dummy dataset for quick testing
# =========================================================

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
        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
        }


# =========================================================
# Real DR dataset
# =========================================================

class DRDataset(Dataset):
    """
    Generic DR dataset reading from CSV.

    Supported CSV modes:

    1) Old mode:
       image_id,label
       10_left.png,0

       -> requires images_dir + image_col=image_id

    2) New mode:
       image_path,label,dataset
       /full/path/to/file.png,0,eyepacs

       -> set image_col=image_path
       -> images_dir can be empty

    Robust to corrupted images:
    retries a few times and then errors clearly.
    """

    def __init__(
        self,
        csv_path: str,
        images_dir: str = "",
        image_col: str = "image_path",
        label_col: str = "label",
        dataset_col: Optional[str] = None,
        image_size: int = 224,
        augment: bool = False,
        max_decode_retries: int = 20,
        log_bad_every: int = 50,
    ):
        self.csv_path = str(csv_path)
        self.df = pd.read_csv(csv_path).reset_index(drop=True)

        self.images_dir = Path(images_dir) if str(images_dir).strip() else None
        self.image_col = str(image_col)
        self.label_col = str(label_col)
        self.dataset_col = str(dataset_col) if dataset_col else None
        self.image_size = int(image_size)

        self.max_decode_retries = int(max_decode_retries)
        self.log_bad_every = int(log_bad_every)
        self.bad_count = 0

        self._validate_columns()

        imagenet_mean = (0.485, 0.456, 0.406)
        imagenet_std = (0.229, 0.224, 0.225)

        if augment:
            self.transform = A.Compose(
                [
                    A.Resize(image_size, image_size),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.4),
                    A.Affine(
                        scale=(0.95, 1.05),
                        translate_percent=(0.0, 0.05),
                        rotate=(-15, 15),
                        p=0.5,
                        border_mode=cv2.BORDER_REFLECT_101,
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

    def _validate_columns(self) -> None:
        missing_cols = []

        if self.image_col not in self.df.columns:
            missing_cols.append(self.image_col)

        if self.label_col not in self.df.columns:
            missing_cols.append(self.label_col)

        if missing_cols:
            raise ValueError(
                f"Missing required columns in CSV {self.csv_path}. "
                f"Missing: {missing_cols}. Found: {list(self.df.columns)}"
            )

        if self.image_col != "image_path" and self.images_dir is None:
            raise ValueError(
                f"CSV uses image_col='{self.image_col}', so images_dir must be provided. "
                f"csv_path={self.csv_path}"
            )

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_img_path(self, row: pd.Series) -> Path:
        raw_value = str(row[self.image_col])

        if self.image_col == "image_path":
            return Path(raw_value)

        return self.images_dir / raw_value  # type: ignore[operator]

    def _read_image(self, img_path: Path):
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        last_img_path = None

        for _ in range(self.max_decode_retries):
            row = self.df.iloc[idx]
            img_path = self._resolve_img_path(row)
            last_img_path = img_path

            image = self._read_image(img_path)

            if image is not None:
                label = int(row[self.label_col])
                augmented = self.transform(image=image)

                sample = {
                    "image": augmented["image"],
                    "label": torch.tensor(label, dtype=torch.long),
                }

                if self.dataset_col and self.dataset_col in row.index:
                    sample["dataset"] = str(row[self.dataset_col])

                return sample

            self.bad_count += 1
            if self.log_bad_every > 0 and self.bad_count % self.log_bad_every == 0:
                print(f"[WARN] Skipped {self.bad_count} unreadable images so far. Latest: {img_path}")

            idx = random.randint(0, len(self.df) - 1)

        raise RuntimeError(
            f"Too many unreadable images encountered (>{self.max_decode_retries} retries). "
            f"Check CSV paths / image_dir / dataset integrity. Last attempted: {last_img_path}"
        )


# =========================================================
# DataModule wrapper
# =========================================================

class DRDataModule:
    def __init__(self, cfg):
        self.cfg = cfg
        self.batch_size = int(cfg.training.batch_size)
        self.num_workers = int(cfg.training.num_workers)
        self.num_classes = int(cfg.model.num_classes)
        self.image_size = int(cfg.data.image_size)

        self.use_dummy = bool(cfg.data.get("use_dummy", False))
        self.use_weighted_sampler = bool(cfg.data.get("use_weighted_sampler", False))
        self.sampler_mode = str(cfg.data.get("sampler_mode", "inverse")).lower()

        self.train_csv = str(cfg.data.train_csv)
        self.val_csv = str(cfg.data.val_csv)
        self.image_dir = str(cfg.data.get("image_dir", ""))
        self.image_col = str(cfg.data.get("image_col", "image_path"))
        self.label_col = str(cfg.data.get("label_col", "label"))
        self.dataset_col = str(cfg.data.get("dataset_col", "dataset"))

    def _pin_memory(self) -> bool:
        return torch.cuda.is_available()

    def _build_weighted_sampler_from_csv(self, csv_path: str, label_col: str) -> WeightedRandomSampler:
        df = pd.read_csv(csv_path)
        labels = df[label_col].astype(int).values

        class_counts = np.bincount(labels, minlength=self.num_classes).astype(np.float32)
        class_counts = np.maximum(class_counts, 1.0)

        if self.sampler_mode != "inverse":
            print(f"[WARN] sampler_mode='{self.sampler_mode}' not implemented. Falling back to inverse.")

        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels]
        sample_weights = torch.tensor(sample_weights, dtype=torch.double)

        print("[INFO] Weighted sampler class counts:", class_counts.tolist())
        print("[INFO] Weighted sampler class weights:", class_weights.tolist())

        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    def _build_dataset(self, csv_path: str, augment: bool) -> Dataset:
        return DRDataset(
            csv_path=csv_path,
            images_dir=self.image_dir,
            image_col=self.image_col,
            label_col=self.label_col,
            dataset_col=self.dataset_col,
            image_size=self.image_size,
            augment=augment,
            max_decode_retries=30,
            log_bad_every=50,
        )

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

        ds = self._build_dataset(
            csv_path=self.train_csv,
            augment=True,
        )

        if self.use_weighted_sampler:
            sampler = self._build_weighted_sampler_from_csv(
                csv_path=self.train_csv,
                label_col=self.label_col,
            )
            return DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=False,
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
            ds = self._build_dataset(
                csv_path=self.val_csv,
                augment=False,
            )

        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self._pin_memory(),
        )