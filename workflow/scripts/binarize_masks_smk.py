from pathlib import Path
import numpy as np
from PIL import Image

# Snakemake provides a global `snakemake` object when using `script:`

in_dir = Path(str(snakemake.input.in_dir))
out_dir = Path(str(snakemake.output.out_dir))
ext = getattr(snakemake.params, "ext", ".png")  # default to .png

out_dir.mkdir(parents=True, exist_ok=True)

files = sorted(in_dir.glob(f"*{ext}"))
if not files:
    raise SystemExit(f"No {ext} files found in {in_dir}")

count = 0
for src in files:
    arr = np.array(Image.open(src).convert("L"), dtype=np.uint8)
    arr[arr == 1] = 255
    dst = out_dir / src.name
    Image.fromarray(arr, mode="L").save(dst)
    count += 1

print(f"Done. Processed {count} file(s) from {in_dir} to {out_dir}.")
