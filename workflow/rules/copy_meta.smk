# rules/copy_meta.smk

import os

def find_meta_like_csv(wc):
    """
    Look for meta-like CSV under ./{dataset}/**/seg_legacy/{meta,bounds}.csv
    Works for:
      - ./{dataset}/seg_legacy/meta.csv
      - ./{dataset}/{other_dir}/seg_legacy/bounds.csv
    """
    root = f"./{wc.dataset}"
    for dirpath, dirnames, filenames in os.walk(root):
        if os.path.basename(dirpath) == "seg_legacy":
            for name in ("bounds.csv", "meta.csv"):
                if name in filenames:
                    return os.path.join(dirpath, name)

    raise FileNotFoundError(
        f"No meta.csv or bounds.csv under {root}/**/seg_legacy/"
    )

rule copy_meta_csv:
    input:
        src = find_meta_like_csv
    output:
        dst = "data/{dataset}/meta/meta.csv"
    shell:
        r"""
        mkdir -p $(dirname {output.dst})
        cp {input.src} {output.dst}
        """
