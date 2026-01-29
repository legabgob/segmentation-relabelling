#!/usr/bin/env python3
"""
Wrapper for get_predictions.py that handles non-square images.
Creates temporary filtered directories with only square images before refinement.
"""
import sys
import argparse
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import subprocess

def filter_square_images(src_dir, dst_dir):
    """Copy only square images from src to dst"""
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    copied = []
    skipped = []
    
    for img_file in sorted(src_dir.glob("*.png")):
        with Image.open(img_file) as img:
            w, h = img.size
            
        if w == h:
            shutil.copy2(img_file, dst_dir / img_file.name)
            copied.append(img_file.name)
        else:
            skipped.append((img_file.name, w, h))
    
    return copied, skipped

def main():
    parser = argparse.ArgumentParser(description="Wrapper for refinement that filters non-square images")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--images-path", required=True)
    parser.add_argument("--masks-path", required=True)
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--k", required=True)
    parser.add_argument("--refine", action="store_true")
    
    args = parser.parse_args()
    
    # Create temporary directories for filtered images
    with tempfile.TemporaryDirectory(prefix="refinement_") as tmpdir:
        tmpdir = Path(tmpdir)
        filtered_segs = tmpdir / "segs"
        filtered_masks = tmpdir / "masks"
        
        print(f"Filtering non-square images...")
        print(f"  Input segs: {args.images_path}")
        print(f"  Input masks: {args.masks_path}")
        
        # Filter segmentations
        copied_segs, skipped_segs = filter_square_images(args.images_path, filtered_segs)
        
        # Filter masks (only copy masks for images we kept)
        filtered_masks.mkdir(parents=True, exist_ok=True)
        masks_src = Path(args.masks_path)
        for seg_name in copied_segs:
            mask_file = masks_src / seg_name
            if mask_file.exists():
                shutil.copy2(mask_file, filtered_masks / seg_name)
        
        if skipped_segs:
            print(f"  Skipped {len(skipped_segs)} non-square images:")
            for name, w, h in skipped_segs[:5]:
                print(f"    {name}: {w}x{h}")
            if len(skipped_segs) > 5:
                print(f"    ... and {len(skipped_segs) - 5} more")
        
        print(f"  Processing {len(copied_segs)} square images")
        
        # Call original refinement script with filtered directories
        cmd = [
            "python3",
            "/SSD/home/gabriel/rrwnet/clone/get_predictions.py",
            "--weights", args.weights,
            "--images-path", str(filtered_segs),
            "--masks-path", str(filtered_masks),
            "--save-path", args.save_path,
            "--k", args.k,
        ]
        if args.refine:
            cmd.append("--refine")
        
        print(f"\nRunning refinement on filtered images...")
        result = subprocess.run(cmd)
        
        if result.returncode != 0:
            print(f"\nRefinement failed with code {result.returncode}", file=sys.stderr)
            sys.exit(result.returncode)
        
        print(f"\nRefinement complete. Output in: {args.save_path}")

if __name__ == "__main__":
    main()
