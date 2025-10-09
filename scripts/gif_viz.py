from PIL import Image, ImageDraw, ImageFont
import argparse
from pathlib import Path
from dice_metrics import find_k_dirs_for_res

# Convert a sequence of images in different directory to an animated GIF
unref_1024_path = Path("/SSD/home/gabriel/rrwnet/data/Fundus-AVSeg/downsampled/1024px/segs/")
unref_576_path  = Path("/SSD/home/gabriel/rrwnet/data/Fundus-AVSeg/downsampled/576px/segs/")
refined_root    = Path("/SSD/home/gabriel/rrwnet/refined_predictions/Fundus-AVSeg/")
res_subdir_1024 = Path("downsampled/1024px")
res_subdir_576  = Path("downsampled/576px")
gt_1024_path    = Path("/SSD/home/gabriel/rrwnet/data/Fundus-AVSeg/downsampled/1024px/GTs/")
gt_576_path     = Path("/SSD/home/gabriel/rrwnet/data/Fundus-AVSeg/downsampled/576px/GTs/")
out_gif_path    = Path("./gifs/")

def load_font(size=16):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except:
        return ImageFont.load_default()

def add_header_strip(
    im: Image.Image,
    text: str,
    side: str = "top",
    pad: int = 8,
    bg=(255, 255, 255),
    fg=(0, 0, 0),
    sep_line: bool = True,
    font_size: int = 16,      
) -> Image.Image:
    """
    Return a new image with an added header/footer/side strip containing `text`.
    Does not resize the original content.
    side: 'top' (default), 'bottom', 'left', 'right'
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
    """
    Horizontally stack two images with optional padding and a vertical separator.
    Assumes both images have the same height; pads the shorter if needed.
    """
    H = max(left.height, right.height)
    # Pad to equal heights if needed
    if left.height != H:
        tmp = Image.new(left.mode, (left.width, H), bg)
        tmp.paste(left, (0, 0))
        left = tmp
    if right.height != H:
        tmp = Image.new(right.mode, (right.width, H), bg)
        tmp.paste(right, (0, 0))
        right = tmp

    sep_w = 1 if sep_line else 0
    W = left.width + pad + sep_w + right.width
    out = Image.new("RGB", (W, H), bg)
    out.paste(left, (0, 0))
    if sep_line:
        draw = ImageDraw.Draw(out)
        x = left.width + pad
        draw.line([(x, 0), (x, H)], fill=(200, 200, 200))
        x += sep_w
        out.paste(right, (x, 0))
    else:
        out.paste(right, (left.width + pad, 0))
    return out

def main():
    ap = argparse.ArgumentParser(description="Make a GIF with labels and a side-by-side GT panel.")
    ap.add_argument("--image", required=True, help="Filename to visualize (e.g., 123_A.png).")
    ap.add_argument("--res", choices=["1024", "576"], default="1024", help="Downsample set.")
    ap.add_argument("--duration", type=int, default=300, help="Per-frame duration (ms).")
    ap.add_argument("--loop", type=int, default=0, help="GIF loop count (0=forever).")
    ap.add_argument("--label-side", choices=["top","bottom","left","right"], default="top",
                    help="Where to place the label strip.")
    ap.add_argument("--font-size", type=int, default=24, help="Label font size (default: 24).")
    ap.add_argument("--gt-side", choices=["left","right"], default="right",
                    help="Place GT panel to the left or right of the animated frame.")
    ap.add_argument("--gt-1024-dir", type=Path, default=gt_1024_path, help="GT dir for 1024px.")
    ap.add_argument("--gt-576-dir",  type=Path, default=gt_576_path,  help="GT dir for 576px.")
    args = ap.parse_args()

    # choose dirs for resolution
    if args.res == "1024":
        unref_dir = unref_1024_path
        gt_dir    = args.gt_1024_dir
        k_dirs    = find_k_dirs_for_res(refined_root, res_subdir_1024)
        tag       = "1024px"
    else:
        unref_dir = unref_576_path
        gt_dir    = args.gt_576_dir
        k_dirs    = find_k_dirs_for_res(refined_root, res_subdir_576)
        tag       = "576px"

    unref_img_path = unref_dir / args.image
    gt_img_path    = gt_dir / args.image

    if not unref_img_path.exists():
        raise SystemExit(f"Unrefined image not found: {unref_img_path}")
    if not gt_img_path.exists():
        raise SystemExit(f"GT image not found: {gt_img_path}")

    # Load static GT once and label it
    gt_raw = Image.open(gt_img_path).convert("RGB")
    unref_raw = Image.open(unref_img_path).convert("RGB")
    if gt_raw.size != unref_raw.size:
        raise SystemExit(f"Size mismatch: GT {gt_raw.size} vs UNREF {unref_raw.size}")

    gt_panel = add_header_strip(gt_raw, "Ground Truth", side=args.label_side, font_size=args.font_size)

    # Build frames: [Unrefined, k=..., k=..., ...]
    if not k_dirs:
        raise SystemExit(f"No k* folders found under {refined_root} for {tag}.")

    base_frame = add_header_strip(unref_raw, "Unrefined", side=args.label_side, font_size=args.font_size)
    frames = []

    # Combine each animated frame with the static GT panel
    def combine_with_gt(frame_img: Image.Image) -> Image.Image:
        if args.gt_side == "right":
            return hstack_with_pad(frame_img, gt_panel, pad=12, sep_line=True)
        else:
            return hstack_with_pad(gt_panel, frame_img, pad=12, sep_line=True)

    frames.append(combine_with_gt(base_frame))

    for K in sorted(k_dirs.keys()):
        p = k_dirs[K] / args.image
        if p.exists():
            ref_raw = Image.open(p).convert("RGB")
            if ref_raw.size != unref_raw.size:
                raise SystemExit(f"Size mismatch at k={K}: REF {ref_raw.size} vs UNREF {unref_raw.size}")
            k_frame = add_header_strip(ref_raw, f"k={K}", side=args.label_side, font_size=args.font_size)
            frames.append(combine_with_gt(k_frame))
        else:
            print(f"[warn] missing refined frame for k={K}: {p}")

    if len(frames) <= 1:
        raise SystemExit("No refined frames found for this image; nothing to animate.")

    out_gif_path.mkdir(parents=True, exist_ok=True)
    out_path = out_gif_path / f"{Path(args.image).stem}_{tag}_withGT.gif"

    # Save GIF (all frames now same size because we stack with the same GT panel each time)
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