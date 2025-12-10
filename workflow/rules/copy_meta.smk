# rules/copy_meta.smk

import os
from pathlib import Path


def find_meta_like_csv(wc):
    """
    Look for either meta.csv or bounds.csv under ./{dataset}/**/seg_legacy/.
    Priority: meta.csv > bounds.csv.
    """
    root = Path(f"./{wc.dataset}")
    meta_candidate = None
    bounds_candidate = None

    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    for p in root.rglob("*.csv"):
        if p.name not in ("meta.csv", "bounds.csv"):
            continue
        if p.parent.name != "seg_legacy":
            continue
        if p.name == "meta.csv":
            meta_candidate = p
        elif p.name == "bounds.csv" and bounds_candidate is None:
            bounds_candidate = p

    if meta_candidate is not None:
        return str(meta_candidate)
    if bounds_candidate is not None:
        return str(bounds_candidate)

    raise FileNotFoundError(
        f"No meta.csv or bounds.csv in any seg_legacy/ under {root}/"
    )


rule copy_meta_csv:
    """
    Copy meta-like CSV (meta.csv or bounds.csv) into data/{dataset}/meta/meta.csv.
    """
    input:
        src = find_meta_like_csv
    output:
        dst = "data/{dataset}/meta/meta.csv"
    run:
        os.makedirs(os.path.dirname(output.dst), exist_ok=True)
        shell("cp {input.src} {output.dst}")

