from pathlib import Path
from PIL import Image
import argparse

def main():
    ap = argparse.ArgumentParser(description="Convert RGB images to grayscale (batch).")
    ap.add_argument("--in_dir", required=True, help="Input folder with images")
    ap.add_argument("--out_dir", required=True, help="Output folder")
    ap.add_argument("--ext", default=".png", help="Extension to match (e.g., .png, .jpg)")
    ap.add_argument("--keep_alpha", action="store_true",
                    help="If image has alpha, keep it (save as LA instead of L)")
    args = ap.parse_args()

    in_dir  = Path(args.in_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in in_dir.glob(f"*{args.ext}") if p.is_file())
    if not files:
        raise SystemExit(f"No {args.ext} files in {in_dir}")

    for p in files:
        im = Image.open(p)
        if args.keep_alpha and im.mode in ("RGBA", "LA"):
            gray = im.convert("LA")   # grayscale + alpha
        else:
            gray = im.convert("L")    # grayscale
        gray.save(out_dir / p.name)
    print(f"Done. Converted {len(files)} file(s) → {out_dir}")

if __name__ == "__main__":
    main()
