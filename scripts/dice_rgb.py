from pathlib import Path
import re
import numpy as np
import pandas as pd
from PIL import Image

# ---------- Paths (edit) ----------
# RGB GTs already resized to match each resolution exactly:
GT_1024_DIR     = Path("/SSD/home/gabriel/rrwnet/data/Fundus-AVSeg/downsampled/1024px/GTs/")
GT_576_DIR      = Path("/SSD/home/gabriel/rrwnet/data/Fundus-AVSeg/downsampled/576px/GTs/")

# Unrefined RGB segmentations
UNREF_1024_DIR  = Path("/SSD/home/gabriel/rrwnet/data/Fundus-AVSeg/downsampled/1024px/segs/")
UNREF_576_DIR   = Path("/SSD/home/gabriel/rrwnet/data/Fundus-AVSeg/downsampled/576px/segs/")

# Refined root (contains k{X}/downsampled/{1024px|576px}/)
REFINED_ROOT    = Path("/SSD/home/gabriel/rrwnet/refined_predictions/Fundus-AVSeg/")
RES_SUBDIR_1024 = Path("downsampled/1024px")
RES_SUBDIR_576  = Path("downsampled/576px")

# Optional ROI masks (boolean, nonzero = inside ROI)
ROI_1024_DIR    = Path("/SSD/home/gabriel/rrwnet/data/Fundus-AVSeg/downsampled/1024px/masks/")
ROI_576_DIR     = Path("/SSD/home/gabriel/rrwnet/data/Fundus-AVSeg/downsampled/576px/masks/")

# Outputs
OUT_CSV_1024    = Path("./metrics_rgbgt_macro_1024.csv")
OUT_CSV_576     = Path("./metrics_rgbgt_macro_576.csv")

EXT = ".png"
# ----------------------------------

def dice_bool(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    inter = np.logical_and(a, b).sum(dtype=np.float64)
    return float((2.0*inter + eps) / (a.sum(dtype=np.float64) + b.sum(dtype=np.float64) + eps))

def dice_bool_masked(a: np.ndarray, b: np.ndarray, roi: np.ndarray, eps: float = 1e-8) -> float:
    """
    Dice between boolean masks a and b, but only within roi==True.
    If ROI has zero True pixels, returns np.nan (no area to evaluate).
    """
    if roi.dtype != bool:
        roi = roi.astype(bool)
    n_roi = roi.sum()
    if n_roi == 0:
        return np.nan
    aR = np.logical_and(a, roi)
    bR = np.logical_and(b, roi)
    inter = np.logical_and(aR, bR).sum(dtype=np.float64)
    return float((2.0*inter + eps) / (aR.sum(dtype=np.float64) + bR.sum(dtype=np.float64) + eps))

def load_rgb_arr(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("RGB"), dtype=np.uint8)

def load_roi_bool(p: Path) -> np.ndarray:
    """
    Load ROI as boolean mask: True = inside ROI (nonzero).
    """
    arr = np.array(Image.open(p).convert("L"))
    return arr > 0


def av_masks(arr: np.ndarray):
    """A from Red>0, V from Green>0. (Blue ignored for GT comparisons.)"""
    A = arr[...,0] > 0
    V = arr[...,1] > 0
    return A, V

def rgb_bool_channels(arr: np.ndarray):
    """R/G/B channel > 0 as booleans (for change metric)."""
    return arr[...,0] > 0, arr[...,1] > 0, arr[...,2] > 0

def av_macro_dice(A1,V1,A2,V2) -> float:
    """Macro-average Dice over A and V."""
    return 0.5 * (dice_bool(A1,A2) + dice_bool(V1,V2))

def av_macro_dice_masked(A1,V1,A2,V2, roi: np.ndarray) -> float:
    return 0.5 * (dice_bool_masked(A1,A2,roi) + dice_bool_masked(V1,V2,roi))

def find_k_dirs_for_res(refined_root: Path, res_subdir: Path) -> dict[int, Path]:
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

def build_df_for_res(gt_dir: Path, unref_dir: Path, k_dirs: dict[int, Path], roi_dir: Path | None = None) -> pd.DataFrame:
    if not k_dirs:
        raise SystemExit("No k* subfolders provided")

    unref_names = {p.name for p in unref_dir.glob(f"*{EXT}") if p.is_file()}
    gt_names    = {p.name for p in gt_dir.glob(f"*{EXT}") if p.is_file()}
    names = sorted(unref_names & gt_names)
    if not names:
        raise SystemExit(f"No overlapping {EXT} files between GT {gt_dir} and UNREF {unref_dir}")

    rows = []
    for name in names:
        p_unref = unref_dir / name
        p_gt    = gt_dir / name

        arr_unref = load_rgb_arr(p_unref)
        arr_gt    = load_rgb_arr(p_gt)

        if arr_gt.shape != arr_unref.shape:
            raise SystemExit(f"Size mismatch for {name}: GT {arr_gt.shape} vs UNREF {arr_unref.shape}")

        # Load ROI if provided; else full-True mask
        if roi_dir is not None:
            p_roi = roi_dir / name
            if not p_roi.exists():
                raise SystemExit(f"ROI mask missing: {p_roi}")
            roi = load_roi_bool(p_roi)
            if roi.shape != arr_unref.shape[:2]:
                raise SystemExit(f"ROI size mismatch for {name}: ROI {roi.shape} vs UNREF {arr_unref.shape[:2]}")
        else:
            roi = np.ones(arr_unref.shape[:2], dtype=bool)

        Au, Vu = av_masks(arr_unref)
        Ag, Vg = av_masks(arr_gt)

        row = {"image": name}
        # Unref vs GT (macro + per-class) within ROI
        row["dice_unref_vs_gt_macro"] = av_macro_dice_masked(Au, Vu, Ag, Vg, roi)
        row["dice_unref_vs_gt_A"]     = dice_bool_masked(Au, Ag, roi)
        row["dice_unref_vs_gt_V"]     = dice_bool_masked(Vu, Vg, roi)

        # Precompute UNREF per-channel bools (for change metrics)
        r_u, g_u, b_u = rgb_bool_channels(arr_unref)

        for K, dK in k_dirs.items():
            p_ref = dK / name
            if not p_ref.exists():
                row[f"dice_k{K}_vs_gt_macro"]    = np.nan
                row[f"dice_k{K}_vs_gt_A"]        = np.nan
                row[f"dice_k{K}_vs_gt_V"]        = np.nan
                row[f"dice_unref_vs_k{K}_rgbmacro"] = np.nan
                row[f"dice_unref_vs_k{K}_avmacro"]  = np.nan
                row[f"dice_unref_vs_k{K}_A"]        = np.nan   
                row[f"dice_unref_vs_k{K}_V"]        = np.nan   
                continue

            arr_ref = load_rgb_arr(p_ref)
            if arr_ref.shape != arr_unref.shape:
                raise SystemExit(f"Size mismatch for {name} at k={K}: REF {arr_ref.shape} vs UNREF {arr_unref.shape}")

            Ar, Vr = av_masks(arr_ref)

            # Refined vs GT (macro + per-class), masked
            row[f"dice_k{K}_vs_gt_macro"] = av_macro_dice_masked(Ar, Vr, Ag, Vg, roi)
            row[f"dice_k{K}_vs_gt_A"]     = dice_bool_masked(Ar, Ag, roi)
            row[f"dice_k{K}_vs_gt_V"]     = dice_bool_masked(Vr, Vg, roi)

            # Change metrics within ROI
            r_r, g_r, b_r = rgb_bool_channels(arr_ref)
            row[f"dice_unref_vs_k{K}_rgbmacro"] = float((
                dice_bool_masked(r_u, r_r, roi) +
                dice_bool_masked(g_u, g_r, roi) +
                dice_bool_masked(b_u, b_r, roi)
            ) / 3.0)
            row[f"dice_unref_vs_k{K}_avmacro"] = 0.5 * (
                dice_bool_masked(r_u, r_r, roi) +
                dice_bool_masked(g_u, g_r, roi)
            )

            row[f"dice_unref_vs_k{K}_A"] = dice_bool_masked(r_u, r_r, roi)  # arteries (red channel)
            row[f"dice_unref_vs_k{K}_V"] = dice_bool_masked(g_u, g_r, roi)  # veins (green channel)

        rows.append(row)

    df = pd.DataFrame(rows).set_index("image")
    k_list = sorted(k_dirs.keys())
    cols = ["dice_unref_vs_gt_macro", "dice_unref_vs_gt_A", "dice_unref_vs_gt_V"]
    for K in k_list:
        cols += [f"dice_k{K}_vs_gt_macro", f"dice_k{K}_vs_gt_A", f"dice_k{K}_vs_gt_V"]
    for K in k_list:
        cols += [f"dice_unref_vs_k{K}_rgbmacro", f"dice_unref_vs_k{K}_avmacro", f"dice_unref_vs_k{K}_A", f"dice_unref_vs_k{K}_V"]
    cols = [c for c in cols if c in df.columns]
    return df[cols]


def main():
    # 1024px
    k_dirs_1024 = find_k_dirs_for_res(REFINED_ROOT, RES_SUBDIR_1024)
    if not k_dirs_1024:
        raise SystemExit(f"No k* with {RES_SUBDIR_1024} under {REFINED_ROOT}")
    df_1024 = build_df_for_res(GT_1024_DIR, UNREF_1024_DIR, k_dirs_1024, roi_dir=ROI_1024_DIR)
    df_1024.to_csv(OUT_CSV_1024, float_format="%.6f")
    print(f"Saved {OUT_CSV_1024} with shape {df_1024.shape}")

    # 576px
    k_dirs_576 = find_k_dirs_for_res(REFINED_ROOT, RES_SUBDIR_576)
    if not k_dirs_576:
        raise SystemExit(f"No k* with {RES_SUBDIR_576} under {REFINED_ROOT}")
    df_576 = build_df_for_res(GT_576_DIR, UNREF_576_DIR, k_dirs_576, roi_dir=ROI_576_DIR)
    df_576.to_csv(OUT_CSV_576, float_format="%.6f")
    print(f"Saved {OUT_CSV_576} with shape {df_576.shape}")

if __name__ == "__main__":
    main()
