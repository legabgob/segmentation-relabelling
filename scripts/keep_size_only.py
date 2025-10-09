#!/usr/bin/env python3
# keep_size_only.py
from pathlib import Path
from PIL import Image
import argparse, sys

def main():
    ap = argparse.ArgumentParser(description="Remove images that are not a given resolution.")
    ap.add_argument("dir", help="Directory to scan")
    ap.add_argument("--width",  type=int, required=True, help="Target width (e.g., 1024)")
    ap.add_argument("--height", type=int, required=True, help="Target height (e.g., 1024)")
    ap.add_argument("--ext", nargs="*", default=[".png", ".jpg", ".jpeg", ".tif", ".tiff"],
                    help="Extensions to consider (default: common image types)")
    ap.add_argument("--dry-run", action="store_true", help="Only print what would be deleted")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subdirectories")
    args = ap.parse_args()

    root = Path(args.dir)
    if not root.is_dir():
        sys.exit(f"Not a directory: {root}")

    exts = {e.lower() for e in args.ext}
    it = root.rglob("*") if args.recursive else root.iterdir()

    kept = deleted = 0
    for p in it:
        if not p.is_file(): 
            continue
        if p.suffix.lower() not in exts:
            continue
        try:
            with Image.open(p) as im:
                if im.size != (args.width, args.height):
                    msg = "would delete" if args.dry_run else "delete"
                    print(f"{msg}: {p}")
                    if not args.dry_run:
                        p.unlink()
                        deleted += 1
                else:
                    kept += 1
        except Exception as e:
            print(f"skip (error): {p} ({e})", file=sys.stderr)

    print(f"Kept: {kept}  Deleted: {deleted}" + ("  (dry-run)" if args.dry_run else ""))

if __name__ == "__main__":
    main()

