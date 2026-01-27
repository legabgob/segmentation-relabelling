from pathlib import Path
import re
import numpy as np
import pandas as pd
from PIL import Image

# Downsampled (1024x1024) GTs and vessel segs
GT_2048_DIR   = Path("/SSD/home/gabriel/rrwnet/data/FIVES/train/downsampled/GTs/")        # grayscale vessel GT (0/255)
SEG_2048_DIR  = Path("/SSD/home/gabriel/rrwnet/data/FIVES/train/vessel_segs/")       # grayscale vessel segmentation (0/255)

# Downsampled RGB sets for unrefined vs refined comparisons
# 1024 set (unrefined RGB + refined RGB under k* folders)
UNREF_1024_DIR     = Path("/SSD/home/gabriel/rrwnet/data/FIVES/train/downsampled/1024px/segs/")     # RGB unrefined

# 576 set
UNREF_576_DIR      = Path("/SSD/home/gabriel/rrwnet/data/FIVES/train/downsampled/576px/segs/")      # RGB unrefined

# Refined root (contains k{X}/downsampled/{1024px|576px}/ subfolders)
REFINED_ROOT   = Path("/SSD/home/gabriel/rrwnet/refined_predictions/FIVES/")

# Names of the resolution subfolders under each kX directory
RES_SUBDIR_1024 = Path("downsampled/1024px")
RES_SUBDIR_576  = Path("downsampled/576px")
# Output CSVs
OUT_CSV_1024 = Path("./metrics_1024.csv")
OUT_CSV_576  = Path("./metrics_576.csv")

# Filenames extension
EXT = ".png"
# ======================================================

def load_gray_bool(p: Path) -> np.ndarray:
    """Load grayscale (L) as boolean mask: True = foreground (nonzero)."""
    a = np.array(Image.open(p).convert("L"))
    return a > 0

def load_rgb_bools(p: Path):
    """Load RGB and return three boolean masks for (R>0), (G>0), (B>0)."""
    a = np.array(Image.open(p).convert("RGB"))
    return a[...,0] > 0, a[...,1] > 0, a[...,2] > 0

def dice_bool(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    inter = np.logical_and(a, b).sum(dtype=np.float64)
    return float((2.0*inter + eps) / (a.sum(dtype=np.float64) + b.sum(dtype=np.float64) + eps))

def rgb_macro_dice(p_unref: Path, p_ref: Path) -> float:
    """Macro-average DICE over R/G/B channels between two RGB label images."""
    r1,g1,b1 = load_rgb_bools(p_unref)
    r2,g2,b2 = load_rgb_bools(p_ref)
    return float((dice_bool(r1,r2) + dice_bool(g1,g2) + dice_bool(b1,b2)) / 3.0)

def compute_seg_vs_gt_map(gt_dir: Path, seg_dir: Path) -> dict:
    """
    Compute DICE between GT (grayscale) and segmentation (grayscale) at 2048x2048.
    Returns {filename: dice}.
    """
    gt_names  = {p.name for p in gt_dir.glob(f"*{EXT}") if p.is_file()}
    seg_names = {p.name for p in seg_dir.glob(f"*{EXT}") if p.is_file()}
    common = sorted(gt_names & seg_names)
    if not common:
        raise SystemExit(f"No common {EXT} between {gt_dir} and {seg_dir}")
    out = {}
    for name in common:
        gt  = load_gray_bool(gt_dir / name)
        seg = load_gray_bool(seg_dir / name)
        out[name] = dice_bool(gt, seg)
    return out

def find_k_dirs_for_res(refined_root, res_subdir):
    rex = re.compile(r"^[kK]\s*=?\s*(\d+)$")
    out = {}
    for d in refined_root.iterdir():
        if d.is_dir():
            m = rex.match(d.name)
            if m:
                K = int(m.group(1))
                p = d / res_subdir
                if p.is_dir():
                    out[K] = p
    return dict(sorted(out.items()))

def build_df_for_res(seg_vs_gt_map: dict, unref_dir: Path, k_dirs: dict[int, Path]) -> pd.DataFrame:
    """
    Build one DataFrame for a given resolution:
      - 'dice_seg_vs_gt' comes from seg_vs_gt_map (computed at 2048).
      - one column per K: dice between unrefined RGB and refined RGB (macro over R/G/B).
    Here k_dirs is a dict like {4: Path(.../k4/downsampled/1024px), 5: Path(.../k5/...)}.
    """
    if not k_dirs:
        raise SystemExit("No k* subfolders provided to build_df_for_res")

    unref_names = {p.name for p in unref_dir.glob(f"*{EXT}") if p.is_file()}
    if not unref_names:
        raise SystemExit(f"No {EXT} files in {unref_dir}")

    # union of refined names across all K
    refined_union = set()
    for d in k_dirs.values():
        refined_union |= {p.name for p in d.glob(f"*{EXT}") if p.is_file()}

    names = sorted(unref_names & refined_union)
    if not names:
        raise SystemExit(f"No overlapping {EXT} files between {unref_dir} and any provided k* directory")

    rows = []
    for name in names:
        row = {"image": name, "dice_seg_vs_gt": seg_vs_gt_map.get(name, np.nan)}
        p_unref = unref_dir / name
        for K, dK in k_dirs.items():
            p_ref = dK / name
            row[f"dice_unref_vs_k{K}"] = rgb_macro_dice(p_unref, p_ref) if p_ref.exists() else np.nan
        rows.append(row)

    df = pd.DataFrame(rows).set_index("image")
    k_cols = [f"dice_unref_vs_k{K}" for K in sorted(k_dirs.keys())]
    return df[["dice_seg_vs_gt"] + k_cols]


def main():
    # Compute once (shared across both tables)
    seg_vs_gt = compute_seg_vs_gt_map(GT_2048_DIR, SEG_2048_DIR)

    # 1024px table
    k_dirs_1024 = find_k_dirs_for_res(REFINED_ROOT, RES_SUBDIR_1024)
    if not k_dirs_1024:
        raise SystemExit(f"No k* folders with {RES_SUBDIR_1024} under {REFINED_ROOT}")
    df_1024 = build_df_for_res(seg_vs_gt, UNREF_1024_DIR, k_dirs_1024)
    df_1024.to_csv(OUT_CSV_1024, float_format="%.6f")
    print(f"Saved {OUT_CSV_1024} with shape {df_1024.shape}")

    # 576px table
    k_dirs_576 = find_k_dirs_for_res(REFINED_ROOT, RES_SUBDIR_576)
    if not k_dirs_576:
        raise SystemExit(f"No k* folders with {RES_SUBDIR_576} under {REFINED_ROOT}")
    df_576 = build_df_for_res(seg_vs_gt, UNREF_576_DIR, k_dirs_576)
    df_576.to_csv(OUT_CSV_576, float_format="%.6f")
    print(f"Saved {OUT_CSV_576} with shape {df_576.shape}")


if __name__ == "__main__":
    main()