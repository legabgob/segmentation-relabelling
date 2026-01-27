# scripts/gray_to_rgb_dir_smk.py
#
# Snakemake "script:" entrypoint
#
# Expected:
#   snakemake.input.in_dir   -> input directory containing grayscale label images
#   snakemake.output.out_dir -> output directory for RGB images
#   snakemake.params.ext     -> extension to match (default ".png")
#   snakemake.params.recursive -> bool, recurse into subdirectories (default False)
#   snakemake.params.overwrite -> bool, overwrite existing outputs (default False)
#
# Notes:
# - This script is intended to be run by Snakemake, which injects a global `snakemake` object.
# - Linters may complain "snakemake is undefined" when editing outside Snakemake; that's normal.

from pathlib import Path
import numpy as np
from PIL import Image


def gray_labels_to_rgb(arr: np.ndarray) -> np.ndarray:
    """
    Convert 2D grayscale label array {0,1,2,3} into RGB A/V/crossings:
      - 1 (arteries): magenta (R=255, B=255)
      - 2 (veins):    cyan    (G=255, B=255)
      - 3 (crossings): white  (255,255,255)
      - 0: background (0,0,0)
    """
    if arr.ndim != 2:
        raise ValueError("Input array must be 2-dimensional")

    arr = arr.astype(np.int32, copy=False)
    H, W = arr.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)

    arteries = (arr == 1)
    veins = (arr == 2)
    crossings = (arr == 3)

    rgb[arteries, 0] = 255
    rgb[arteries, 2] = 255

    rgb[veins, 1] = 255
    rgb[veins, 2] = 255

    rgb[crossings] = [255, 255, 255]

    return rgb


def convert_one(src: Path, dst: Path) -> None:
    img = Image.open(src).convert("L")
    arr = np.array(img)
    rgb = gray_labels_to_rgb(arr)

    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(dst)


def iter_inputs(in_dir: Path, ext: str, recursive: bool):
    if recursive:
        yield from sorted(p for p in in_dir.rglob(f"*{ext}") if p.is_file())
    else:
        yield from sorted(p for p in in_dir.glob(f"*{ext}") if p.is_file())


# ---------------- Snakemake entrypoint ----------------

in_dir = Path(str(snakemake.input.in_dir))
out_dir = Path(str(snakemake.output.out_dir))

ext = str(getattr(snakemake.params, "ext", ".png"))
recursive = bool(getattr(snakemake.params, "recursive", False))
overwrite = bool(getattr(snakemake.params, "overwrite", False))

if not in_dir.exists():
    raise SystemExit(f"Input directory does not exist: {in_dir}")

out_dir.mkdir(parents=True, exist_ok=True)

files = list(iter_inputs(in_dir, ext, recursive))
if not files:
    raise SystemExit(f"No {ext} files found in {in_dir} (recursive={recursive})")

n_done = 0
n_skipped = 0

for i, src in enumerate(files, 1):
    # preserve relative subdir structure if recursive
    rel = src.relative_to(in_dir) if recursive else Path(src.name)
    dst = out_dir / rel

    if dst.exists() and not overwrite:
        n_skipped += 1
        continue

    convert_one(src, dst)
    n_done += 1

    if i % 100 == 0:
        print(f"[gray_to_rgb_dir] processed {i}/{len(files)}...")

print(
    f"[gray_to_rgb_dir] done. converted={n_done}, skipped={n_skipped}, "
    f"inputs={len(files)}, in={in_dir}, out={out_dir}"
)

