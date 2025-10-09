from pathlib import Path
import argparse
from PIL import Image, ImageOps

def downsample_one(src: Path, dst: Path, kind: str, target_w: int):
    im = Image.open(src)

    if kind == "predictions":
        # Keep as RGB discrete labels
        im = im.convert("RGB")
        resample = Image.NEAREST
    elif kind == "masks":
        # Keep as grayscale discrete labels
        im = im.convert("L")
        resample = Image.NEAREST
    else:
        raise ValueError("--kind must be 'predictions' or 'masks'")

    w, h = im.size
    # Preserve aspect ratio
    target_h = int(round(h * (target_w / float(w))))

    im_small = im.resize((target_w, target_h), resample=resample)

    dst.parent.mkdir(parents=True, exist_ok=True)
    im_small.save(dst)

def main():
    ap = argparse.ArgumentParser(description="Downsample images in batch for predictions or masks.")
    ap.add_argument("--in_dir", required=True, help="Folder with input PNGs (2048x2048).")
    ap.add_argument("--out_dir", required=True, help="Folder to write resized PNGs.")
    ap.add_argument("--kind", required=True, choices=["predictions", "masks"],
                    help="Input type: 'predictions' (RGB A/V/crossings 0/255) or 'masks' (grayscale ROI 0/255).")
    ap.add_argument("--width", type=int, default=1024, help="Target width (default: 1024).")
    ap.add_argument("--ext", default=".png", help="Image extension to match (default: .png).")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    files = sorted(p for p in in_dir.glob(f"*{args.ext}") if p.is_file())

    if not files:
        raise SystemExit(f"No {args.ext} files found in {in_dir}")

    print(f"Found {len(files)} file(s). Downsampling to width {args.width} as {args.kind}…")
    for i, src in enumerate(files, 1):
        dst = out_dir / src.name
        downsample_one(src, dst, args.kind, args.width)
        if i % 50 == 0:
            print(f"  {i} done…")
    print(f"Done. Wrote {len(files)} file(s) to {out_dir}")

if __name__ == "__main__":
    main()

