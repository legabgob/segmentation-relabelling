# scripts/gray_to_rgb_dir_smk.py
#
# Snakemake "directory → directory" batch conversion:
# Convert grayscale label PNGs with values {0,1,2,3} to RGB A/V/BV mapping.
#
# Expected Snakemake interface:
#   input:
#     in_dir = "path/to/grayscale_dir"
#   output:
#     out_dir = directory("path/to/rgb_dir")
#   params (optional):
#     ext = ".png"   # default ".png"
#
# Notes:
# - This script processes ALL files matching *ext in in_dir (non-recursive).
# - Output filenames are preserved (same basename + same extension).

from pathlib import Path
import numpy as np
from PIL import Image


def gray_labels_to_rgb(arr: np.ndarray) -> np.ndarray:
    """
    Convert grayscale labels to RGB colors.

    Labels:
      0 = background (black)
      1 = arteries   -> magenta (R+B)
      2 = veins      -> cyan    (G+B)
      3 = crossings  -> white   (R+G+B)
    """
    if arr.ndim != 2:
        raise ValueError("Input array must be 2-dimensional (grayscale).")

    arr = arr.astype(np.int32, copy=False)

    h, w = arr.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    arteries = (arr == 1)
    veins = (arr == 2)
    crossings = (arr == 3)

    # arteries: magenta
    rgb[arteries, 0] = 255
    rgb[arteries, 2] = 255

    # veins: cyan
    rgb[veins, 1] = 255
    rgb[veins, 2] = 255

    # crossings: white
    rgb[crossings] = (255, 255, 255)

    return rgb


def convert_one(src: Path, dst: Path) -> None:
    img = Image.open(src).convert("L")
    arr = np.array(img)
    rgb = gray_labels_to_rgb(arr)
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(dst)


# ---------------- Snakemake entrypoint ----------------

in_dir = Path(str(snakemake.input.in_dir))
out_dir = Path(str(snakemake.output.out_dir))
ext = getattr(snakemake.params, "ext", ".png")

out_dir.mkdir(parents=True, exist_ok=True)

files = sorted(p for p in in_dir.glob(f"*{ext}") if p.is_file())
if not files:
    raise SystemExit(f"No {ext} files found in {in_dir}")

for i, src in enumerate(files, 1):
    dst = out_dir / src.name  # keep filename
    convert_one(src, dst)
    if i % 100 == 0:
        print(f"[gray_to_rgb] {i}/{len(files)} done...")

print(f"[gray_to_rgb] Done. Wrote {len(files)} file(s) to {out_dir}")

