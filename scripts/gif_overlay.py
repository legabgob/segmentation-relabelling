from PIL import Image, ImageDraw, ImageFont
import argparse
from pathlib import Path
import numpy as np
from dice_metrics import find_k_dirs_for_res

# -------- Defaults (edit if you want) --------
unref_1024_path = Path("/SSD/home/gabriel/rrwnet/data/Fundus-AVSeg/downsampled/1024px/segs/")
unref_576_path  = Path("/SSD/home/gabriel/rrwnet/data/Fundus-AVSeg/downsampled/576px/segs/")

cfi_1024_path   = Path("/SSD/home/gabriel/rrwnet/data/Fundus-AVSeg/downsampled/1024px/CFIs/")
cfi_576_path    = Path("/SSD/home/gabriel/rrwnet/data/Fundus-AVSeg/downsampled/576px/CFIs/")

gt_1024_path    = Path("/SSD/home/gabriel/rrwnet/data/Fundus-AVSeg/downsampled/1024px/GTs/")
gt_576_path     = Path("/SSD/home/gabriel/rrwnet/data/Fundus-AVSeg/downsampled/576px/GTs/")

refined_root    = Path("/SSD/home/gabriel/rrwnet/refined_predictions/Fundus-AVSeg/")
res_subdir_1024 = Path("downsampled/1024px")
res_subdir_576  = Path("downsampled/576px")

out_gif_dir     = Path("./gifs/")
# ---------------------------------------------

def load_font(size=16):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()

def add_header_strip(
    im: Image.Image,
    text: str,
    side: str = "top",
    pad: int = 8,
    bg=(255, 255, 255),
    fg=(0, 0, 0),
    sep_line: bool = True,
    font_size: int = 20,
) -> Image.Image:
    """
    Return a new image with an added header/footer/side strip containing `text`.
    Does not resize original content. side ∈ {'top','bottom','left','right'}.
    """
    font = load_font(size=font_size)
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        text_w, text_h = draw.textsize(text, font=font)

    W, H = im.size
    strip_w = text_w + 2 * pad
    strip_h = text_h + 2 * pad

    if side in ("top", "bottom"):
        new = Image.new("RGB", (W, H + strip_h), bg)
        d = ImageDraw.Draw(new)
        tx = (W - text_w) // 2
        ty = (strip_h - text_h) // 2
        if side == "top":
            d.text((tx, ty), text, font=font, fill=fg)
            if sep_line:
                d.line([(0, strip_h - 1), (W, strip_h - 1)], fill=(200, 200, 200))
            new.paste(im, (0, strip_h))
        else:
            new.paste(im, (0, 0))
            if sep_line:
                d.line([(0, H), (W, H)], fill=(200, 200, 200))
            d.text((tx, H + ty), text, font=font, fill=fg)
        return new
    else:
        new = Image.new("RGB", (W + strip_w, H), bg)
        d = ImageDraw.Draw(new)
        tx = (strip_w - text_w) // 2
        ty = (H - text_h) // 2
        if side == "left":
            d.text((tx, ty), text, font=font, fill=fg)
            if sep_line:
                d.line([(strip_w - 1, 0), (strip_w - 1, H)], fill=(200, 200, 200))
            new.paste(im, (strip_w, 0))
        else:
            new.paste(im, (0, 0))
            if sep_line:
                d.line([(W, 0), (W, H)], fill=(200, 200, 200))
            d.text((W + tx, ty), text, font=font, fill=fg)
        return new

def hstack_with_pad(
    left: Image.Image,
    right: Image.Image,
    pad: int = 12,
    bg=(255, 255, 255),
    sep_line: bool = True,
) -> Image.Image:
    """Horizontally stack two images with optional padding + a separator line."""
    H = max(left.height, right.height)
    if left.height != H:
        tmp = Image.new(left.mode, (left.width, H), bg); tmp.paste(left, (0, 0)); left = tmp
    if right.height != H:
        tmp = Image.new(right.mode, (right.width, H), bg); tmp.paste(right, (0, 0)); right = tmp

    sep_w = 1 if sep_line else 0
    W = left.width + pad + sep_w + right.width
    out = Image.new("RGB", (W, H), bg)
    out.paste(left, (0, 0))
    x = left.width + pad
    if sep_line:
        draw = ImageDraw.Draw(out)
        draw.line([(x, 0), (x, H)], fill=(200, 200, 200))
        x += sep_w
    out.paste(right, (x, 0))
    return out

def overlay_on_cfi(cfi_img: Image.Image, seg_img: Image.Image, alpha: float = 0.6, bg_thr: int = 0) -> Image.Image:
    """
    Overlay `seg_img` on `cfi_img` with alpha only where seg is not background.
    Background defined as all channels <= bg_thr.
    Returns an RGB composite (no transparency channel in the result).
    """
    cfi = np.array(cfi_img.convert("RGB"), dtype=np.float32)
    seg = np.array(seg_img.convert("RGB"), dtype=np.float32)
    if cfi.shape != seg.shape:
        raise SystemExit(f"Size mismatch: CFI {cfi.shape} vs SEG {seg.shape}")

    fg = (seg > bg_thr).any(axis=2)  # True where any channel is non-background
    out = cfi.copy()
    out[fg] = (1.0 - alpha) * cfi[fg] + alpha * seg[fg]
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")

def main():
    ap = argparse.ArgumentParser(description="Animated GIF: overlay segmentations on CFI, with a static GT panel.")
    ap.add_argument("--image", required=True, help="Filename to visualize (e.g., 123_A.png).")
    ap.add_argument("--res", choices=["1024", "576"], default="1024", help="Downsample set.")
    ap.add_argument("--duration", type=int, default=300, help="Per-frame duration (ms).")
    ap.add_argument("--loop", type=int, default=0, help="GIF loop count (0=forever).")
    ap.add_argument("--label-side", choices=["top","bottom","left","right"], default="top",
                    help="Where to place the label strip.")
    ap.add_argument("--font-size", type=int, default=24, help="Label font size.")
    ap.add_argument("--gt-side", choices=["left","right"], default="right",
                    help="Place GT panel to the left or right of the animated frame.")
    ap.add_argument("--alpha", type=float, default=0.6, help="Overlay alpha (0..1).")
    ap.add_argument("--bg-thr", type=int, default=0, help="Seg background threshold (<= thr treated as background).")
    # Optional overrides for directories:
    ap.add_argument("--unref-1024-dir", type=Path, default=unref_1024_path)
    ap.add_argument("--unref-576-dir",  type=Path, default=unref_576_path)
    ap.add_argument("--cfi-1024-dir",   type=Path, default=cfi_1024_path)
    ap.add_argument("--cfi-576-dir",    type=Path, default=cfi_576_path)
    ap.add_argument("--gt-1024-dir",    type=Path, default=gt_1024_path)
    ap.add_argument("--gt-576-dir",     type=Path, default=gt_576_path)
    args = ap.parse_args()

    # choose dirs for resolution
    if args.res == "1024":
        unref_dir = args.unref_1024_dir
        cfi_dir   = args.cfi_1024_dir
        gt_dir    = args.gt_1024_dir
        k_dirs    = find_k_dirs_for_res(refined_root, res_subdir_1024)
        tag       = "1024px"
    else:
        unref_dir = args.unref_576_dir
        cfi_dir   = args.cfi_576_dir
        gt_dir    = args.gt_576_dir
        k_dirs    = find_k_dirs_for_res(refined_root, res_subdir_576)
        tag       = "576px"

    unref_img_path = unref_dir / args.image
    gt_img_path    = gt_dir / args.image
    cfi_img_path   = cfi_dir / args.image

    # basic checks
    for pth, lab in [(unref_img_path,"Unrefined"), (gt_img_path,"GT"), (cfi_img_path,"CFI")]:
        if not pth.exists():
            raise SystemExit(f"{lab} image not found: {pth}")

    # Load base CFI and two segmentations (GT + unref)
    cfi_raw   = Image.open(cfi_img_path).convert("RGB")
    gt_raw    = Image.open(gt_img_path).convert("RGB")
    unref_raw = Image.open(unref_img_path).convert("RGB")
    if not (cfi_raw.size == gt_raw.size == unref_raw.size):
        raise SystemExit(f"Size mismatch among CFI {cfi_raw.size}, GT {gt_raw.size}, UNREF {unref_raw.size}")

    # Build the static GT panel (overlayed on CFI, then labeled)
    gt_overlay = overlay_on_cfi(cfi_raw, gt_raw, alpha=args.alpha, bg_thr=args.bg_thr)
    gt_panel   = add_header_strip(gt_overlay, "Ground Truth", side=args.label_side, font_size=args.font_size)

    # Animated frames: Unrefined first, then k=...
    if not k_dirs:
        raise SystemExit(f"No k* folders found under {refined_root} for {tag}.")

    frames = []

    def mk_labeled_overlay(seg_img: Image.Image, label: str) -> Image.Image:
        over = overlay_on_cfi(cfi_raw, seg_img, alpha=args.alpha, bg_thr=args.bg_thr)
        return add_header_strip(over, label, side=args.label_side, font_size=args.font_size)

    # Unrefined frame
    base_frame = mk_labeled_overlay(unref_raw, "Unrefined")

    # combine with GT panel
    def combine_with_gt(frame_img: Image.Image) -> Image.Image:
        return hstack_with_pad(frame_img, gt_panel, pad=12, sep_line=True) if args.gt_side == "right" \
               else hstack_with_pad(gt_panel, frame_img, pad=12, sep_line=True)

    frames.append(combine_with_gt(base_frame))

    # Refined frames
    for K in sorted(k_dirs.keys()):
        p = k_dirs[K] / args.image
        if p.exists():
            ref_raw = Image.open(p).convert("RGB")
            if ref_raw.size != cfi_raw.size:
                raise SystemExit(f"Size mismatch at k={K}: REF {ref_raw.size} vs CFI {cfi_raw.size}")
            k_frame = mk_labeled_overlay(ref_raw, f"k={K}")
            frames.append(combine_with_gt(k_frame))
        else:
            print(f"[warn] missing refined frame for k={K}: {p}")

    if len(frames) <= 1:
        raise SystemExit("No refined frames found for this image; nothing to animate.")

    out_gif_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_gif_dir / f"{Path(args.image).stem}_{tag}_overlay_withGT.gif"

    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=args.duration,
        loop=args.loop,
        optimize=True
    )
    print(f"Saved GIF: {out_path}  ({len(frames)} frames)")

if __name__ == "__main__":
    main()
