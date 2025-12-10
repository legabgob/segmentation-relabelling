rule refinement:
    input:
      weights = lambda wildcards: (
          "/SSD/home/gabriel/rrwnet/data/weights/rrwnet_HRF_0.pth"
          if wildcards.res == "1024"
          else "/SSD/home/gabriel/rrwnet/data/weights/rrwnet_RITE_refinement.pth"
          ),
      segmentations = lambda wildcards: (
            f"/SSD/home/gabriel/rrwnet/data/{wildcards.dataset}/downsampled/{wildcards.res}px/segs/"
        ),
        masks = lambda wildcards: (
            f"/SSD/home/gabriel/rrwnet/data/{wildcards.dataset}/downsampled/{wildcards.res}px/masks/"
        ),
    output:
        refined = directory(
            "/SSD/home/gabriel/rrwnet/refined/{dataset}/k{k}/downsampled/{res}px/"
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


