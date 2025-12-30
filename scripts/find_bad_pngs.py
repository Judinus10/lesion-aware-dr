import cv2
from pathlib import Path
import time

img_dir = Path(r"E:\DR_related\eyepacs\Images")
out_file = Path("bad_images.txt")

bad = []
total = 0
t0 = time.time()

files = list(img_dir.glob("*.png"))
n = len(files)

print(f"Scanning {n} PNG files in: {img_dir}")

for i, p in enumerate(files, start=1):
    total += 1
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)

    if img is None:
        bad.append(p)

    # progress every 500 files
    if i % 500 == 0:
        elapsed = time.time() - t0
        print(f"[{i}/{n}] bad={len(bad)} elapsed={elapsed:.1f}s")

print("\nDONE.")
print("Total files:", n)
print("Bad images:", len(bad))

if bad:
    out_file.write_text("\n".join(str(x) for x in bad), encoding="utf-8")
    print("Saved list to:", out_file.resolve())
else:
    print("No bad images found.")
