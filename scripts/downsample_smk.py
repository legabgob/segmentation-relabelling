from pathlib import Path
from PIL import Image


def downsample_one(src: Path, dst: Path, kind: str, target_w: int):
    im = Image.open(src)

    if kind == "segs":
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


# -------- Snakemake entrypoint --------
# Expect:
#   input.in_dir   -> input directory
#   output.out_dir -> output directory
#   params.kind    -> "predictions" or "masks"
#   params.width   -> target width (int)
#   params.ext     -> extension (".png" by default)

in_dir = Path(str(snakemake.input.in_dir))
out_dir = Path(str(snakemake.output.out_dir))
kind = str(snakemake.params.kind)
width = int(snakemake.params.width)
ext = str(getattr(snakemake.params, "ext", ".png"))

files = sorted(p for p in in_dir.glob(f"*{ext}") if p.is_file())

if not files:
    raise SystemExit(f"No {ext} files found in {in_dir}")

print(f"Found {len(files)} file(s). Downsampling to width {width} as {kind}…")
for i, src in enumerate(files, 1):
    dst = out_dir / src.name
    downsample_one(src, dst, kind, width)
    if i % 50 == 0:
        print(f"  {i} done…")
print(f"Done. Wrote {len(files)} file(s) to {out_dir}")
