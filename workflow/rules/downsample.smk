# rules/downsample.smk
from snakemake.io import glob_wildcards

# We downsample:
#  - converted segs  -> data/{dataset}/segs_converted/{sample}.png
#  - ROI masks       -> data/{dataset}/roi_masks/{sample}.png

KINDS = ["segs_converted", "roi_masks"]
WIDTHS = ["576", "1024"]   # width wildcard is a string; we cast to int in params

# Discover dataset/sample pairs from segs_converted; roi_masks should match the same names
DATASETS, SAMPLES = glob_wildcards("data/{dataset}/segs_converted/{sample}.png")

rule downsample:
    """
    Downsample one image for a given dataset/kind/width.
    Input:
        data/{dataset}/{kind}/{sample}.png
    Output:
        data/{dataset}/downsampled/{width}px/{kind}/{sample}.png
    """
    input:
        "data/{dataset}/{kind}/{sample}.png"
    output:
        "data/{dataset}/downsampled/{width}px/{kind}/{sample}.png"
    params:
        kind = "{kind}",
        width = lambda wc: int(wc.width),
    script:
        "scripts/downsample_smk.py"

