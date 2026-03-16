from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


# =========================================================
# CONFIG LOADING
# =========================================================

def load_config(cfg_path: str) -> dict:
    cfg_path = Path(cfg_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# =========================================================
# HELPERS
# =========================================================

def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def verify_files_exist(df: pd.DataFrame, image_path_col: str = "image_path", sample_n: int = 5) -> None:
    missing = df[~df[image_path_col].apply(lambda x: Path(x).exists())]

    if len(missing) > 0:
        print(f"❌ Missing image files detected: {len(missing)}")
        print(missing.head(sample_n))
        raise RuntimeError("Some image files listed in CSV do not exist.")

    dataset_name = df["dataset"].iloc[0] if "dataset" in df.columns and len(df) > 0 else "unknown"
    print(f"✅ All image files found for dataset={dataset_name}, rows={len(df)}")


def save_df(df: pd.DataFrame, out_path: Path, name: str) -> None:
    ensure_parent_dir(out_path)
    df.to_csv(out_path, index=False)
    print(f"✅ Saved {name}: {out_path} | shape={df.shape}")


def print_distribution(df: pd.DataFrame, title: str) -> None:
    print(f"\n{title}")
    print(df["label"].value_counts().sort_index())

    if "dataset" in df.columns:
        print("\nDataset distribution:")
        print(df["dataset"].value_counts())


# =========================================================
# STANDARDIZATION
# =========================================================

def build_eyepacs_df(cfg: dict) -> pd.DataFrame:
    eyepacs_cfg = cfg["data"]["sources"]["eyepacs"]

    labels_csv = Path(eyepacs_cfg["labels_csv"])
    images_dir = Path(eyepacs_cfg["images_dir"])
    image_ext = str(eyepacs_cfg.get("image_ext", ".png"))
    image_col_raw = str(eyepacs_cfg["image_col_raw"])
    label_col_raw = str(eyepacs_cfg["label_col_raw"])

    df = pd.read_csv(labels_csv)
    print("EyePACS original columns:", list(df.columns))

    required_cols = {image_col_raw, label_col_raw}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"EyePACS CSV missing required columns. "
            f"Required={required_cols}, found={list(df.columns)}"
        )

    df = df.rename(columns={
        image_col_raw: "image_id",
        label_col_raw: "label",
    })

    df["label"] = df["label"].astype(int)
    df["image_id"] = df["image_id"].astype(str) + image_ext
    df["image_path"] = df["image_id"].apply(lambda x: str(images_dir / x))
    df["dataset"] = "eyepacs"

    df = df[["image_path", "label", "dataset"]].copy()

    if cfg["data"]["split"].get("verify_files", True):
        verify_files_exist(df)

    print_distribution(df, "EyePACS label distribution:")
    return df


def build_aptos_df(csv_path: Path, images_dir: Path, cfg: dict, split_name: str) -> pd.DataFrame:
    aptos_cfg = cfg["data"]["sources"]["aptos"]

    image_ext = str(aptos_cfg.get("image_ext", ".png"))
    image_col_raw = str(aptos_cfg["image_col_raw"])
    label_col_raw = str(aptos_cfg["label_col_raw"])

    df = pd.read_csv(csv_path)
    print(f"\nAPTOS {split_name} original columns:", list(df.columns))

    required_cols = {image_col_raw, label_col_raw}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"APTOS {split_name} CSV missing required columns. "
            f"Required={required_cols}, found={list(df.columns)}"
        )

    df = df.rename(columns={
        image_col_raw: "image_id",
        label_col_raw: "label",
    })

    df["label"] = df["label"].astype(int)
    df["image_id"] = df["image_id"].astype(str) + image_ext
    df["image_path"] = df["image_id"].apply(lambda x: str(images_dir / x))
    df["dataset"] = "aptos"

    df = df[["image_path", "label", "dataset"]].copy()

    if cfg["data"]["split"].get("verify_files", True):
        verify_files_exist(df)

    print_distribution(df, f"APTOS {split_name} label distribution:")
    return df


# =========================================================
# SPLIT LOGIC
# =========================================================

def split_eyepacs(df: pd.DataFrame, cfg: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_cfg = cfg["data"]["split"]
    val_size = float(split_cfg.get("eyepacs_val_size", 0.2))
    random_state = int(split_cfg.get("random_state", 42))

    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        stratify=df["label"],
        random_state=random_state,
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


# =========================================================
# MAIN
# =========================================================

def main(cfg_path: str = "configs/base.yaml") -> None:
    cfg = load_config(cfg_path)

    split_outputs = cfg["data"]["split"]["outputs"]
    aptos_cfg = cfg["data"]["sources"]["aptos"]

    aptos_train_csv = Path(aptos_cfg["train_csv"])
    aptos_val_csv = Path(aptos_cfg["val_csv"])
    aptos_train_images_dir = Path(aptos_cfg["train_images_dir"])
    aptos_val_images_dir = Path(aptos_cfg["val_images_dir"])

    out_eyepacs_train = Path(split_outputs["eyepacs_train_csv"])
    out_eyepacs_val = Path(split_outputs["eyepacs_val_csv"])
    out_aptos_train = Path(split_outputs["aptos_train_csv"])
    out_aptos_val = Path(split_outputs["aptos_val_csv"])
    out_combined_train = Path(split_outputs["combined_train_csv"])
    out_combined_val = Path(split_outputs["combined_val_csv"])

    print("===== Building multi-dataset splits =====\n")

    # --------------------------------------------------
    # EyePACS
    # --------------------------------------------------
    eyepacs_df = build_eyepacs_df(cfg)
    eyepacs_train_df, eyepacs_val_df = split_eyepacs(eyepacs_df, cfg)

    # --------------------------------------------------
    # APTOS
    # --------------------------------------------------
    aptos_train_df = build_aptos_df(
        csv_path=aptos_train_csv,
        images_dir=aptos_train_images_dir,
        cfg=cfg,
        split_name="train",
    ).reset_index(drop=True)

    aptos_val_df = build_aptos_df(
        csv_path=aptos_val_csv,
        images_dir=aptos_val_images_dir,
        cfg=cfg,
        split_name="val",
    ).reset_index(drop=True)

    # --------------------------------------------------
    # Combined
    # --------------------------------------------------
    random_state = int(cfg["data"]["split"].get("random_state", 42))

    combined_train_df = pd.concat(
        [eyepacs_train_df, aptos_train_df],
        axis=0
    ).sample(frac=1, random_state=random_state).reset_index(drop=True)

    combined_val_df = pd.concat(
        [eyepacs_val_df, aptos_val_df],
        axis=0
    ).sample(frac=1, random_state=random_state).reset_index(drop=True)

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    save_df(eyepacs_train_df, out_eyepacs_train, "EyePACS train")
    save_df(eyepacs_val_df, out_eyepacs_val, "EyePACS val")
    save_df(aptos_train_df, out_aptos_train, "APTOS train")
    save_df(aptos_val_df, out_aptos_val, "APTOS val")
    save_df(combined_train_df, out_combined_train, "Combined train")
    save_df(combined_val_df, out_combined_val, "Combined val")

    # --------------------------------------------------
    # Final report
    # --------------------------------------------------
    print("\n===== FINAL SUMMARY =====")
    print(f"EyePACS train:   {eyepacs_train_df.shape}")
    print(f"EyePACS val:     {eyepacs_val_df.shape}")
    print(f"APTOS train:     {aptos_train_df.shape}")
    print(f"APTOS val:       {aptos_val_df.shape}")
    print(f"Combined train:  {combined_train_df.shape}")
    print(f"Combined val:    {combined_val_df.shape}")

    print_distribution(combined_train_df, "Combined train distribution:")
    print_distribution(combined_val_df, "Combined val distribution:")

    print("\nDone.")


if __name__ == "__main__":
    main()