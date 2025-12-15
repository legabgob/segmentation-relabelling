rule refinement:
    input:
      weights = lambda wildcards: (
          "data/weights/rrwnet_HRF_0.pth"
          if wildcards.res == "1024"
          else "data/weights/rrwnet_RITE_refinement.pth"
          ),
      segmentations = lambda wildcards: (
            f"data/{wildcards.dataset}/downsampled/{wildcards.res}px/segs_converted/"
        ),
        masks = lambda wildcards: (
            f"data/{wildcards.dataset}/downsampled/{wildcards.res}px/roi_masks/"
        ),
    output:
        refined = directory(
            "data/refined/{dataset}/k{k}/downsampled/{res}px/"
        ),
    shell:
        r"""
        python3 /SSD/home/gabriel/rrwnet/clone/get_predictions.py \
            --weights {input.weights} \
            --images-path {input.segmentations} \
            --masks-path {input.masks} \
            --output-path {output.refined} \
            --num-iterations {wildcards.k} \
            --refine
        """


