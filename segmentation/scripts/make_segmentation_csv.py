from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]


def load_config(cfg_path: str) -> dict:
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def extract_id(path: Path) -> Optional[str]:
    """
    Extract numeric id from names like:
    IDRiD_01.jpg
    IDRiD_01_EX.tif
    IDRiD_01_HE.tif
    """
    m = re.search(r"IDRiD_(\d+)", path.stem)
    if not m:
        return None
    return m.group(1).zfill(2)


def index_files(folder: Path) -> Dict[str, Path]:
    files = {}
    for p in folder.iterdir():
        if p.is_file():
            idx = extract_id(p)
            if idx is not None:
                files[idx] = p
    return files


def build_rows(
    image_dir: Path,
    ex_dir: Path,
    he_dir: Path,
    split_name: str,
) -> List[dict]:
    image_index = index_files(image_dir)
    ex_index = index_files(ex_dir)
    he_index = index_files(he_dir)

    common_ids = sorted(set(image_index.keys()) & set(ex_index.keys()) & set(he_index.keys()))
    rows = []

    for idx in common_ids:
        rows.append(
            {
                "image_id": idx,
                "split": split_name,
                "image_path": str(image_index[idx]),
                "ex_mask_path": str(ex_index[idx]),
                "he_mask_path": str(he_index[idx]),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, default="configs/base.yaml")
    args = parser.parse_args()

    cfg = load_config(args.cfg_path)
    raw_root = Path(cfg["paths"]["raw_root"])
    out_train = Path(cfg["paths"]["train_csv"])
    out_val = Path(cfg["paths"]["val_csv"])
    out_test = Path(cfg["paths"]["test_csv"])

    random.seed(cfg.get("seed", 42))

    # dataset folders
    original_images = raw_root / "Original_Images"
    gt_root = raw_root / "Segmentation_Groundtruths"

    train_img_dir = original_images / "Training Set"
    test_img_dir = original_images / "Testing Set"

    train_ex_dir = gt_root / "Training Set" / "Hard Exudates"
    train_he_dir = gt_root / "Training Set" / "Haemorrhages"

    test_ex_dir = gt_root / "Testing Set" / "Hard Exudates"
    test_he_dir = gt_root / "Testing Set" / "Haemorrhages"

    train_rows_full = build_rows(train_img_dir, train_ex_dir, train_he_dir, "train")
    test_rows = build_rows(test_img_dir, test_ex_dir, test_he_dir, "test")

    if len(train_rows_full) == 0:
        raise RuntimeError("No training rows found. Check folder names and file names.")
    if len(test_rows) == 0:
        raise RuntimeError("No test rows found. Check folder names and file names.")

    df_train_full = pd.DataFrame(train_rows_full)
    df_test = pd.DataFrame(test_rows)

    val_ratio = float(cfg["data"].get("train_val_split", 0.2))
    shuffled_idx = list(df_train_full.index)
    random.shuffle(shuffled_idx)

    val_size = max(1, int(len(shuffled_idx) * val_ratio))
    val_idx = set(shuffled_idx[:val_size])

    df_train = df_train_full[~df_train_full.index.isin(val_idx)].copy().reset_index(drop=True)
    df_val = df_train_full[df_train_full.index.isin(val_idx)].copy().reset_index(drop=True)
    df_val["split"] = "val"

    out_train.parent.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(out_train, index=False)
    df_val.to_csv(out_val, index=False)
    df_test.to_csv(out_test, index=False)

    print(f"Saved train CSV: {out_train} ({len(df_train)} rows)")
    print(f"Saved val CSV:   {out_val} ({len(df_val)} rows)")
    print(f"Saved test CSV:  {out_test} ({len(df_test)} rows)")


if __name__ == "__main__":
    main()