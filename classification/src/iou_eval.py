import argparse
from pathlib import Path

import numpy as np
import cv2
import pandas as pd


def compute_iou(mask_bin: np.ndarray, heat_bin: np.ndarray) -> float:
    intersection = np.logical_and(mask_bin, heat_bin)
    union = np.logical_or(mask_bin, heat_bin)
    if union.sum() == 0:
        return 0.0
    return float(intersection.sum() / union.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam_npy_dir", type=str, required=True, help="Folder containing *_cam.npy files")
    ap.add_argument("--mask_dir", type=str, required=True, help="Folder containing lesion masks (png/jpg)")
    ap.add_argument("--mask_ext", type=str, default="png", help="Mask extension (png/jpg)")
    ap.add_argument("--threshold", type=float, default=0.5, help="CAM threshold for binarization")
    ap.add_argument("--out_csv", type=str, default="outputs/iou/iou_scores.csv")
    args = ap.parse_args()

    cam_dir = Path(args.cam_npy_dir)
    mask_dir = Path(args.mask_dir)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    npy_files = sorted(cam_dir.glob("*_cam.npy"))

    if len(npy_files) == 0:
        raise FileNotFoundError(f"No *_cam.npy files found in: {cam_dir}")

    for cam_path in npy_files:
        stem = cam_path.stem.replace("_cam", "")
        mask_path = mask_dir / f"{stem}.{args.mask_ext}"

        if not mask_path.exists():
            # skip if no matching mask
            continue

        cam = np.load(cam_path)  # HxW in [0..1]
        if cam.ndim != 2:
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        # resize mask to match CAM
        mask_rs = cv2.resize(mask, (cam.shape[1], cam.shape[0]), interpolation=cv2.INTER_NEAREST)

        cam_bin = cam >= args.threshold
        mask_bin = mask_rs > 0

        iou = compute_iou(mask_bin, cam_bin)

        rows.append({
            "image_id": stem,
            "iou": iou,
            "cam_file": str(cam_path),
            "mask_file": str(mask_path),
            "threshold": args.threshold,
        })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise RuntimeError(
            "No IoU computed. Most likely mask filenames don’t match CAM filenames.\n"
            "Fix: ensure mask files are named exactly like the image stem used in CAM."
        )

    df.to_csv(out_csv, index=False)
    print(f"[IoU] Saved: {out_csv}")
    print(f"[IoU] Mean IoU: {df['iou'].mean():.4f}  |  N={len(df)}")


if __name__ == "__main__":
    main()