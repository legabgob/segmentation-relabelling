# workflow/rules/vascx_dataset_from_refined.smk
import os
import re
from pathlib import Path
from snakemake.io import glob_wildcards, directory

LEGACY_ROOT = config.get("legacy_root", ".")
VASCX_VIEW_ROOT = config.get("vascx", {}).get("dataset_view_root", "results/vascx_datasets")

# Where your grayscale label maps live (produced by refined_rgb_to_labels)
REF_LABEL_ROOT = config.get("labels_out_root", "results/refined_labels")

# --- Discover datasets (and other_dir) from seg_legacy/original ---
# We only need to locate "original" reliably.
pat_simple = os.path.join(LEGACY_ROOT, "{dataset}", "seg_legacy", "original")
pat_other  = os.path.join(LEGACY_ROOT, "{dataset}", "{other_dir}", "seg_legacy", "original")

ds1 = []
if os.path.exists(os.path.join(LEGACY_ROOT)):  # avoid crash if root missing
    ds1 = glob_wildcards(pat_simple)[0]
ds2 = []
od2 = []
try:
    ds2, od2 = glob_wildcards(pat_other)
except ValueError:
    # glob_wildcards raises if pattern can't match; safe to ignore
    ds2, od2 = [], []

OTHERDIR_DATASETS = sorted(set(ds2))
SIMPLE_DATASETS = sorted(set(ds1) - set(ds2))  # if in both, treat as other_dir

OTHERDIRS = {}
for d, od in zip(ds2, od2):
    OTHERDIRS.setdefault(d, set()).add(od)
OTHERDIRS = {d: sorted(v) for d, v in OTHERDIRS.items()}

# Resolutions + k-values should match your refinement config
RESOLUTIONS = [str(r) for r in config.get("resolutions", ["576", "1024"])]
k_start, k_end = config.get("k_range", [3, 9])
K_VALUES = list(range(int(k_start), int(k_end)))


def original_dir_simple(wc):
    return os.path.join(LEGACY_ROOT, wc.dataset, "seg_legacy", "original")

def original_dir_other(wc):
    return os.path.join(LEGACY_ROOT, wc.dataset, wc.other_dir, "seg_legacy", "original")


rule make_vascx_view_simple:
    """
    Build a VascX-like dataset view for refined runs (simple layout).
    Creates symlinks:
      - original -> legacy_root/{dataset}/seg_legacy/original
      - av       -> results/refined_labels/{dataset}/k{k}/downsampled/{res}px
    """
    wildcard_constraints:
        dataset="|".join(map(re.escape, SIMPLE_DATASETS)) if SIMPLE_DATASETS else "NO_MATCH"
    input:
        original = original_dir_simple,
        av = directory(f"{REF_LABEL_ROOT}" + "/{dataset}/k{k}/downsampled/{res}px"),
    output:
        view = directory(f"{VASCX_VIEW_ROOT}" + "/{dataset}/k{k}/downsampled/{res}px"),
    run:
        view = Path(output.view)
        view.mkdir(parents=True, exist_ok=True)

        # Create/replace symlinks
        orig_link = view / "original"
        av_link = view / "av"

        for link in (orig_link, av_link):
            if link.exists() or link.is_symlink():
                link.unlink()

        os.symlink(os.path.abspath(input.original), orig_link)
        os.symlink(os.path.abspath(input.av), av_link)


rule make_vascx_view_otherdir:
    """
    Same as above, but preserves {other_dir}.
    """
    wildcard_constraints:
        dataset="|".join(map(re.escape, OTHERDIR_DATASETS)) if OTHERDIR_DATASETS else "NO_MATCH"
    input:
        original = original_dir_other,
        av = directory(f"{REF_LABEL_ROOT}" + "/{dataset}/k{k}/downsampled/{res}px"),
    output:
        view = directory(f"{VASCX_VIEW_ROOT}" + "/{dataset}/{other_dir}/k{k}/downsampled/{res}px"),
    run:
        view = Path(output.view)
        view.mkdir(parents=True, exist_ok=True)

        orig_link = view / "original"
        av_link = view / "av"

        for link in (orig_link, av_link):
            if link.exists() or link.is_symlink():
                link.unlink()

        os.symlink(os.path.abspath(input.original), orig_link)
        os.symlink(os.path.abspath(input.av), av_link)

