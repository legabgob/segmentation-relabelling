from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Script to plot a grid of segmentations overlayed on original CFIs for visual comparison
# 1024 set
CFIs_1024  = Path("/SSD/home/gabriel/rrwnet/data/Fundus-AVSeg/downsampled/1024px/CFIs/")
GTs_1024   = Path("/SSD/home/gabriel/rrwnet/data/Fundus-AVSeg/downsampled/1024px/GTs/")
UNREF_1024 = Path("/SSD/home/gabriel/rrwnet/data/Fundus-AVSeg/downsampled/1024px/segs/")
K4_1024    = Path("/SSD/home/gabriel/rrwnet/refined_predictions/Fundus-AVSeg/k4/downsampled/1024px/")
K5_1024    = Path("/SSD/home/gabriel/rrwnet/refined_predictions/Fundus-AVSeg/k5/downsampled/1024px/")
K6_1024    = Path("/SSD/home/gabriel/rrwnet/refined_predictions/Fundus-AVSeg/k6/downsampled/1024px/")
K7_1024    = Path("/SSD/home/gabriel/rrwnet/refined_predictions/Fundus-AVSeg/k7/downsampled/1024px/")
OUT_1024   = Path("/SSD/home/gabriel/rrwnet/overlay_1024_Fundus-AVSeg.png")

# 576 set
CFIs_576   = Path("/SSD/home/gabriel/rrwnet/data/Fundus-AVSeg/downsampled/576px/CFIs/")
GTs_576    = Path("/SSD/home/gabriel/rrwnet/data/Fundus-AVSeg/downsampled/576px/GTs/")
UNREF_576  = Path("/SSD/home/gabriel/rrwnet/data/Fundus-AVSeg/downsampled/576px/segs/")
K4_576     = Path("/SSD/home/gabriel/rrwnet/refined_predictions/Fundus-AVSeg/k4/downsampled/576px/")
K5_576     = Path("/SSD/home/gabriel/rrwnet/refined_predictions/Fundus-AVSeg/k5/downsampled/576px/")
K6_576     = Path("/SSD/home/gabriel/rrwnet/refined_predictions/Fundus-AVSeg/k6/downsampled/576px/")
K7_576     = Path("/SSD/home/gabriel/rrwnet/refined_predictions/Fundus-AVSeg/k7/downsampled/576px/")
OUT_576    = Path("/SSD/home/gabriel/rrwnet/overlay_576_Fundus-AVSeg.png")

# Selection
CATS    = ["A", "N", "G", "D"]   # choose subset if you like
PER_CAT = 1                      # rows per category
EXT     = ".png"

# Overlay settings
ALPHA    = 0.75   # 0..1, how strongly to show segmentation colors
BG_THR   = 2      # treat pixels with all channels <= BG_THR as background
# ----------------------------------

COL_HEADERS = ["CFI", "GT", "unref", "k=4", "k=5", "k=6", "k=7"]

def common_names(*dirs: Path):
    sets = [{p.name for p in d.glob(f"*{EXT}") if p.is_file()} for d in dirs]
    return sorted(set.intersection(*sets))

def pick_names_by_category(all_names):
    chosen = []
    for c in CATS:
        cat = [n for n in all_names if n.endswith(f"_{c}{EXT}")]
        chosen.extend(sorted(cat)[:PER_CAT])
    return chosen

def pil2np_rgb(path: Path):
    return np.array(Image.open(path).convert("RGB"))

def overlay_on_cfi(cfi_rgb: np.ndarray, seg_rgb: np.ndarray, alpha: float = 0.6, bg_thr: int = 0) -> np.ndarray:
    """
    Alpha-blend the colored segmentation onto the CFI image.
    Background (all channels <= bg_thr) becomes transparent.
    """
    if cfi_rgb.shape != seg_rgb.shape:
        raise ValueError(f"Size mismatch: CFI {cfi_rgb.shape} vs SEG {seg_rgb.shape}")

    cfi_f = cfi_rgb.astype(np.float32)
    seg_f = seg_rgb.astype(np.float32)

    # Foreground where any channel > BG_THR
    fg = (seg_rgb[..., 0] > bg_thr) | (seg_rgb[..., 1] > bg_thr) | (seg_rgb[..., 2] > bg_thr)
    if not np.any(fg):
        return cfi_rgb.copy()

    out = cfi_f.copy()
    # Blend only on foreground pixels: out = (1-alpha)*base + alpha*seg
    out[fg] = (1.0 - alpha) * cfi_f[fg] + alpha * seg_f[fg]
    return np.clip(out, 0, 255).astype(np.uint8)

def plot_grid_overlay(cfi_dir: Path, gts: Path, unref: Path, k4: Path, k5: Path, k6: Path, k7: Path, out_path: Path):
    # Make sure we have images in all columns for chosen names
    names = common_names(cfi_dir, gts, unref, k4, k5, k6, k7)
    if not names:
        raise SystemExit(f"No common {EXT} across CFI/GT/unref/k* in: {cfi_dir}")

    names = pick_names_by_category(names)
    if not names:
        raise SystemExit("No filenames matched the requested categories.")

    rows, cols = len(names), len(COL_HEADERS)

    # Sanity check: identical sizes
    w0, h0 = Image.open(cfi_dir / names[0]).size
    for name in names:
        for d in (cfi_dir, gts, unref, k4, k5, k6, k7):
            if Image.open(d / name).size != (w0, h0):
                raise ValueError(f"Size mismatch for {name} in {d}; script assumes identical sizes.")

    W, H = w0, h0            # tile size in pixels
    g = 16                   # gutter in pixels between images
    dpi = 200                # any dpi works if figsize matches pixels

    total_w = cols * W + (cols - 1) * g
    total_h = rows * H + (rows - 1) * g
    figsize = (total_w / dpi, total_h / dpi)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=dpi)

    # Make gutters exactly g pixels, no outer margins
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0,
                    wspace=g / W, hspace=g / H)

    #fig_w = cols * 3.0
    #fig_h = rows * 3.0
    #fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), dpi=300)
    if rows == 1:
        axes = np.array([axes])

    col_dirs = [cfi_dir, gts, unref, k4, k5, k6, k7]

    for i, name in enumerate(names):
        # Load base once
        base = pil2np_rgb(cfi_dir / name)

        for j, d in enumerate(col_dirs):
            ax = axes[i, j]

            if j == 0:
                # First column: show CFI alone
                ax.imshow(base, interpolation="nearest")
            else:
                seg = pil2np_rgb(d / name)
                over = overlay_on_cfi(base, seg, alpha=ALPHA, bg_thr=BG_THR)
                ax.imshow(over, interpolation="nearest")

            ax.set_xticks([]); ax.set_yticks([])
            ax.set_frame_on(False)

            #if i == 0:
                #ax.set_title(COL_HEADERS[j], fontsize=11, pad=6)
            #if j == 0:
                #ax.set_ylabel(name, fontsize=10, rotation=0, labelpad=24, va="center")
            if i == 0:
                ax.text(0.5, 1.0, COL_HEADERS[j], ha="center", va="bottom",
                        transform=ax.transAxes, fontsize=11,
                        bbox=dict(facecolor="white", alpha=0.85, pad=2, edgecolor="none"))

            if j == 0:
                ax.text(0.0, 0.0, name, ha="left", va="bottom",
                        transform=ax.transAxes, fontsize=9,
                        bbox=dict(facecolor="white", alpha=0.85, pad=1, edgecolor="none"))


    #fig.tight_layout(pad=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    #fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path, dpi=dpi, facecolor="white", bbox_inches=None, pad_inches=0)
    plt.close(fig)
    print(f"Saved: {out_path}")

def main():
    plot_grid_overlay(CFIs_1024, GTs_1024, UNREF_1024, K4_1024, K5_1024, K6_1024, K7_1024, OUT_1024)
    plot_grid_overlay(CFIs_576,  GTs_576,  UNREF_576,  K4_576,  K5_576,  K6_576,  K7_576,  OUT_576)

if __name__ == "__main__":
    main()
