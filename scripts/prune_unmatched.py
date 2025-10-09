from pathlib import Path
import argparse, sys

def main():
    ap = argparse.ArgumentParser(
        description="Remove files in SRC that don't exist (by filename) in REF."
    )
    ap.add_argument("src", help="Directory to clean (files removed here).")
    ap.add_argument("ref", help="Reference directory (kept filenames come from here).")
    ap.add_argument("--ext", default=".png",
                    help="Only consider this extension (e.g., .png). Use 'all' for all files.")
    ap.add_argument("--recursive", action="store_true",
                    help="Recurse into SRC (defaults to top-level only).")
    ap.add_argument("--delete", action="store_true",
                    help="Actually delete files. Omit to do a dry-run.")
    args = ap.parse_args()

    src = Path(args.src); ref = Path(args.ref)
    if not src.is_dir() or not ref.is_dir():
        sys.exit("Both SRC and REF must be directories.")

    # Build set of reference filenames
    if args.ext.lower() == "all":
        ref_names = {p.name for p in ref.iterdir() if p.is_file()}
        ext = None
    else:
        ext = args.ext.lower()
        ref_names = {p.name for p in ref.glob(f"*{ext}") if p.is_file()}

    it = src.rglob("*") if args.recursive else src.iterdir()
    deleted = kept = skipped = 0

    for p in it:
        if not p.is_file():
            continue
        if ext and p.suffix.lower() != ext:
            skipped += 1
            continue
        if p.name not in ref_names:
            print(f"{'delete' if args.delete else 'would delete'}: {p}")
            if args.delete:
                p.unlink()
                deleted += 1
        else:
            kept += 1

    print(f"Kept: {kept}  Deleted: {deleted}" + ("" if args.delete else "  (dry-run)"))

if __name__ == "__main__":
    main()