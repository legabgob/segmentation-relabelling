# workflow/rules/refinement.smk
from snakemake.io import directory

rule refinement:
    input:
        weights = lambda wc: (
            "data/weights/rrwnet_HRF_0.pth" if wc.res == "1024"
            else "data/weights/rrwnet_RITE_refinement.pth"
        ),
        segmentations = "data/{dataset}/downsampled/{res}px/segs_converted",
        masks = "data/{dataset}/downsampled/{res}px/roi_masks_binarized",
    output:
        refined = directory("results/refined/{dataset}/k{k}/downsampled/{res}px")
    resources:
        gpu_jobs = 1  # Limit GPU concurrent jobs
    shell:
        r"""
        python3 workflow/scripts/refinement_wrapper.py \
            --weights {input.weights} \
            --images-path {input.segmentations} \
            --masks-path {input.masks} \
            --save-path {output.refined} \
            --k {wildcards.k} \
            --refine
        """

