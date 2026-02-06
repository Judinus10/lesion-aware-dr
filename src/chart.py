import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = r"E:\DR_related\eyepacs\all_labels.csv"
LABEL_COL = "level"   # <-- FIXED

df = pd.read_csv(CSV_PATH)

print("Loaded file:", CSV_PATH)
print("Columns:", list(df.columns))

# sanity check
assert LABEL_COL in df.columns, f"{LABEL_COL} not found in CSV!"

counts = df[LABEL_COL].value_counts().sort_index()
print("Counts:\n", counts)

plt.figure(figsize=(7,5))
plt.bar(counts.index, counts.values)
plt.xlabel("DR Grade")
plt.ylabel("Image Count")
plt.title("EyePACS Class Distribution (Used Dataset)")
plt.tight_layout()
plt.savefig("eyepacs_full_distribution.png", dpi=300)
plt.show()
