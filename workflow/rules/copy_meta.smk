# rules/copy_meta.smk

import os
from glob import glob

def find_meta_like_csv(wc):
    """
    Look for meta.csv or bounds.csv under:
      ./{dataset}/seg_legacy/
      ./{dataset}/*/seg_legacy/
    """
    candidates = []

    # case 1: dataset/seg_legacy/
    candidates += glob(f"./{wc.dataset}/seg_legacy/meta.csv")
    candidates += glob(f"./{wc.dataset}/seg_legacy/bounds.csv")

    # case 2: dataset/*/seg_legacy/
    candidates += glob(f"./{wc.dataset}/*/seg_legacy/meta.csv")
    candidates += glob(f"./{wc.dataset}/*/seg_legacy/bounds.csv")

    if not candidates:
        raise FileNotFoundError(
            f"No meta.csv or bounds.csv found for dataset {wc.dataset}"
        )

    # deterministic choice
    return sorted(candidates)[0]


rule copy_meta_csv:
    input:
        src = find_meta_like_csv
    output:
        dst = "data/{dataset}/meta/meta.csv"
    run:
        os.makedirs(os.path.dirname(output.dst), exist_ok=True)
        shell("cp {input.src} {output.dst}")

