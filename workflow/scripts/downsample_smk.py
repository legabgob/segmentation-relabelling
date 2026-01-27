from pathlib import Path
from PIL import Image


def downsample_one(src: Path, dst: Path, kind: str, target_w: int):
    im = Image.open(src)

    if kind == "segs_converted":
        # Keep as RGB discrete labels
        im = im.convert("RGB")
        resample = Image.NEAREST
    elif kind == "roi_masks":
        # Keep as grayscale discrete labels
        im = im.convert("L")
        resample = Image.NEAREST
    else:
        raise ValueError("--kind must be 'segs_converted' or 'roi_masks'")

    w, h = im.size
    # Preserve aspect ratio
    target_h = int(round(h * (target_w / float(w))))

    im_small = im.resize((target_w, target_h), resample=resample)

    dst.parent.mkdir(parents=True, exist_ok=True)
    im_small.save(dst)


# -------- Snakemake entrypoint --------
src = Path(str(snakemake.input[0]))
dst = Path(str(snakemake.output[0]))
kind = str(snakemake.params.kind)
width = int(snakemake.params.width)

downsample_one(src, dst, kind, width)
