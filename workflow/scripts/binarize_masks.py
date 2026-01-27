import argparse
from pathlib import Path
import numpy as np
from PIL import Image

def main():
    ap = argparse.ArgumentParser(description="Batch replace pixel value 1 -> 255 (grayscale PNGs).")
    ap.add_argument("--in_dir", required=True, help="Input directory with grayscale .png images")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.png"))
    if not files:
        raise SystemExit(f"No .png files found in {in_dir}")

    for src in files:
        arr = np.array(Image.open(src).convert("L"), dtype=np.uint8)
        arr[arr == 1] = 255
        Image.fromarray(arr, mode="L").save(out_dir / src.name)

    print(f"Done. Processed {len(files)} file(s).")

if __name__ == "__main__":
    main()