
from snakemake.io import directory

# Optionally define datasets here or in your main Snakefile
DATASETS = ["FIVES", "Fundus-AVSeg"]  # adapt as needed

rule replace_1_to_255:
    """
    Batch replace grayscale pixel value 1 -> 255 for all images in a directory.
    """
    input:
        # Directory containing original grayscale images
        in_dir = directory("data/{dataset}/masks/orig")
    output:
        # Directory to write modified images
        out_dir = directory("data/{dataset}/masks/binarized")
    params:
        ext = ".png"
    script:
        "scripts/binarize_masks_smk.py"

