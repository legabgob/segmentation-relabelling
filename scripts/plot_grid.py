from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# -------- EDIT THESE PATHS --------
# 1024 set
GTs_1024   = Path("/SSD/home/gabriel/rrwnet/data/Fundus-AVSeg/downsampled/1024px/GTs/")
UNREF_1024 = Path("/SSD/home/gabriel/rrwnet/data/Fundus-AVSeg/downsampled/1024px/segs/")
K4_1024    = Path("/SSD/home/gabriel/rrwnet/refined_predictions/Fundus-AVSeg/k4/downsampled/1024px/")
K5_1024    = Path("/SSD/home/gabriel/rrwnet/refined_predictions/Fundus-AVSeg/k5/downsampled/1024px/")
K6_1024    = Path("/SSD/home/gabriel/rrwnet/refined_predictions/Fundus-AVSeg/k6/downsampled/1024px/")
K7_1024    = Path("/SSD/home/gabriel/rrwnet/refined_predictions/Fundus-AVSeg/k7/downsampled/1024px/")
OUT_1024   = Path("/SSD/home/gabriel/rrwnet/out_grid_1024_Fundus-AVSeg.png")

# 576 set
GTs_576    = Path("/SSD/home/gabriel/rrwnet/data/Fundus-AVSeg/downsampled/576px/GTs/")
UNREF_576  = Path("/SSD/home/gabriel/rrwnet/data/Fundus-AVSeg/downsampled/576px/segs/")
K4_576     = Path("/SSD/home/gabriel/rrwnet/refined_predictions/Fundus-AVSeg/k4/downsampled/576px/")
K5_576     = Path("/SSD/home/gabriel/rrwnet/refined_predictions/Fundus-AVSeg/k5/downsampled/576px/")
K6_576     = Path("/SSD/home/gabriel/rrwnet/refined_predictions/Fundus-AVSeg/k6/downsampled/576px/")
K7_576     = Path("/SSD/home/gabriel/rrwnet/refined_predictions/Fundus-AVSeg/k7/downsampled/576px/")
OUT_576    = Path("/SSD/home/gabriel/rrwnet/out_grid_576_Fundus-AVSeg.png")

# Selection
CATS    = ["A", "N", "G", "D"]  # choose subset if you like
PER_CAT = 3                     # rows per category
EXT     = ".png"
# ----------------------------------

COL_HEADERS = ["Ground truth", "unref", "k=4", "k=5", "k=6", "k=7"]

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

def plot_grid(GTs, unref, k4, k5, k6, k7, out_path: Path):
    names = common_names(GTs, unref, k4, k5, k6, k7)
    if not names:
        raise SystemExit(f"No common {EXT} files across columns in: {unref}")
    names = pick_names_by_category(names)
    if not names:
        raise SystemExit("No filenames matched the requested categories.")

    rows, cols = len(names), len(COL_HEADERS)

    # Sanity check: identical sizes (no resizing in this script)
    w0, h0 = Image.open(unref / names[0]).size
    for name in names:
        for d in (GTs, unref, k4, k5, k6, k7):
            if Image.open(d / name).size != (w0, h0):
                raise ValueError(f"Size mismatch for {name} in {d}; script assumes identical sizes.")

    fig_w = cols * 3.0   # tweak if you want larger/smaller output
    fig_h = rows * 3.0
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), dpi=150)
    if rows == 1:
        axes = np.array([axes])  # normalize to 2D array

    col_dirs = [GTs, unref, k4, k5, k6, k7]

    for i, name in enumerate(names):
        for j, d in enumerate(col_dirs):
            ax = axes[i, j]
            ax.imshow(pil2np_rgb(d / name))
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_frame_on(False)
            # column headers on first row
            if i == 0:
                ax.set_title(COL_HEADERS[j], fontsize=11, pad=6)
            # row labels on first column
            if j == 0:
                ax.set_ylabel(name, fontsize=10, rotation=0, labelpad=24, va="center")

    fig.tight_layout(pad=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")

def main():
    plot_grid(GTs_1024, UNREF_1024, K4_1024, K5_1024, K6_1024, K7_1024, OUT_1024)
    plot_grid(GTs_576,  UNREF_576,  K4_576,  K5_576,  K6_576,  K7_576,  OUT_576)

if __name__ == "__main__":
    main()
