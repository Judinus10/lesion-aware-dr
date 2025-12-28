import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# --------------------
# Paths
# --------------------
DATA_DIR = Path(r"E:\DR_related\eyepacs")
LABELS_CSV = DATA_DIR / "all_labels.csv"
IMAGES_DIR = DATA_DIR / "Images"

OUT_TRAIN = DATA_DIR / "train.csv"
OUT_VAL = DATA_DIR / "val.csv"

# --------------------
# Load labels
# --------------------
df = pd.read_csv(LABELS_CSV)
print("Original columns:", list(df.columns))

# --------------------
# Normalize column names
# --------------------
rename_map = {}
if "image" in df.columns:
    rename_map["image"] = "image_id"
if "level" in df.columns:
    rename_map["level"] = "label"

df = df.rename(columns=rename_map)

if "image_id" not in df.columns or "label" not in df.columns:
    raise ValueError(f"Required columns missing. Found: {list(df.columns)}")

# --------------------
# Clean labels
# --------------------
df["label"] = df["label"].astype(int)

# --------------------
# Add .png extension (EyePACS resized)
# --------------------
df["image_id"] = df["image_id"].astype(str) + ".png"

# --------------------
# Drop unused columns
# --------------------
keep_cols = ["image_id", "label"]
df = df[keep_cols]

# --------------------
# Verify image files exist (VERY IMPORTANT)
# --------------------
missing = df[~df["image_id"].apply(lambda x: (IMAGES_DIR / x).exists())]

if len(missing) > 0:
    print("❌ Missing image files detected:")
    print(missing.head())
    raise RuntimeError("Some image files listed in CSV do not exist in Images/")

print("✅ All image files found.")

# --------------------
# Shuffle for safety
# --------------------
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# --------------------
# Stratified 80/20 split
# --------------------
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

# --------------------
# Save outputs
# --------------------
train_df.to_csv(OUT_TRAIN, index=False)
val_df.to_csv(OUT_VAL, index=False)

# --------------------
# Report
# --------------------
print("Saved:", OUT_TRAIN)
print("Saved:", OUT_VAL)
print("Train size:", len(train_df))
print("Val size:", len(val_df))
print("\nTrain label distribution:")
print(train_df["label"].value_counts().sort_index())
print("\nVal label distribution:")
print(val_df["label"].value_counts().sort_index())
