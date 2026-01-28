#!/usr/bin/env python3
"""
VascX Feature Extraction Script for Snakemake
Extracts vessel features from retinal images using VascX toolkit.
"""
from pathlib import Path
import sys
import pandas as pd

from vascx.fundus.loader import RetinaLoader
from vascx.utils.analysis import extract_in_parallel


def main():
    # Get Snakemake inputs/outputs/params
    ds_dir = Path(str(snakemake.input.ds_dir))
    out_file = Path(str(snakemake.output.features))
    
    n_jobs = int(getattr(snakemake.params, "n_jobs", 64))
    feature_set = str(getattr(snakemake.params, "feature_set", "bergmann"))
    av_subfolder = str(getattr(snakemake.params, "av_subfolder", "av"))
    sep = str(getattr(snakemake.params, "sep", "\t"))
    na_rep = str(getattr(snakemake.params, "na_rep", "NaN"))
    
    print(f"\n{'='*70}", file=sys.stderr)
    print(f"VascX Feature Extraction", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)
    print(f"Dataset directory: {ds_dir}", file=sys.stderr)
    print(f"Output file: {out_file}", file=sys.stderr)
    print(f"Feature set: {feature_set}", file=sys.stderr)
    print(f"AV subfolder: {av_subfolder}", file=sys.stderr)
    print(f"Parallel jobs: {n_jobs}", file=sys.stderr)
    
    # Convert to absolute path (CRITICAL for OpenCV)
    ds_dir = ds_dir.resolve()
    print(f"\nAbsolute dataset path: {ds_dir}", file=sys.stderr)
    
    # Validation: Check directory structure
    print(f"\n{'='*70}", file=sys.stderr)
    print("Validating directory structure...", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)
    
    if not ds_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {ds_dir}")
    
    original_dir = ds_dir / "original"
    if not original_dir.exists():
        raise FileNotFoundError(f"Missing 'original/' subdirectory in {ds_dir}")
    
    av_dir = ds_dir / av_subfolder
    if not av_dir.exists():
        raise FileNotFoundError(f"Missing '{av_subfolder}/' subdirectory in {ds_dir}")
    
    meta_file = ds_dir / "meta.csv"
    if not meta_file.exists():
        raise FileNotFoundError(f"Missing 'meta.csv' in {ds_dir}")
    
    # Count files
    original_files = list(original_dir.glob("*.png"))
    av_files = list(av_dir.glob("*.png"))
    
    print(f"✓ Found {len(original_files)} files in original/", file=sys.stderr)
    print(f"✓ Found {len(av_files)} files in {av_subfolder}/", file=sys.stderr)
    print(f"✓ Found meta.csv", file=sys.stderr)
    
    # Validate meta.csv
    try:
        meta_df = pd.read_csv(meta_file)
        print(f"✓ meta.csv contains {len(meta_df)} entries", file=sys.stderr)
        print(f"  Columns: {', '.join(meta_df.columns.tolist())}", file=sys.stderr)
    except Exception as e:
        raise ValueError(f"Failed to read meta.csv: {e}")
    
    # Check for file mismatches
    original_stems = {f.stem for f in original_files}
    av_stems = {f.stem for f in av_files}
    
    if original_stems != av_stems:
        missing_in_av = original_stems - av_stems
        missing_in_orig = av_stems - original_stems
        if missing_in_av:
            print(f"⚠ Warning: {len(missing_in_av)} files in original/ but not in {av_subfolder}/", file=sys.stderr)
            print(f"  Examples: {list(missing_in_av)[:3]}", file=sys.stderr)
        if missing_in_orig:
            print(f"⚠ Warning: {len(missing_in_orig)} files in {av_subfolder}/ but not in original/", file=sys.stderr)
            print(f"  Examples: {list(missing_in_orig)[:3]}", file=sys.stderr)
    else:
        print(f"✓ File counts match between original/ and {av_subfolder}/", file=sys.stderr)
    
    # Test loading a single image
    print(f"\n{'='*70}", file=sys.stderr)
    print("Testing image loading...", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)
    
    if original_files and av_files:
        test_orig = original_files[0]
        test_av = av_files[0]
        print(f"Test files:", file=sys.stderr)
        print(f"  Original: {test_orig}", file=sys.stderr)
        print(f"  AV: {test_av}", file=sys.stderr)
        
        try:
            import cv2
            import numpy as np
            
            img_orig = cv2.imread(str(test_orig))
            img_av = cv2.imread(str(test_av))
            
            if img_orig is None:
                print(f"✗ ERROR: Failed to load original image!", file=sys.stderr)
            else:
                print(f"  ✓ Original: {img_orig.shape}, dtype={img_orig.dtype}", file=sys.stderr)
            
            if img_av is None:
                print(f"✗ ERROR: Failed to load AV image!", file=sys.stderr)
            else:
                print(f"  ✓ AV: {img_av.shape}, dtype={img_av.dtype}", file=sys.stderr)
                print(f"    Unique values: {np.unique(img_av)[:10]}", file=sys.stderr)
                
        except Exception as e:
            print(f"✗ ERROR during test load: {e}", file=sys.stderr)
    
    # Create output directory
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load dataset with VascX
    print(f"\n{'='*70}", file=sys.stderr)
    print("Loading dataset with VascX RetinaLoader...", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)
    
    try:
        # CRITICAL: Use absolute path for VascX
        loader = RetinaLoader.from_folder(
            str(ds_dir),  # VascX might need string, not Path
            av_subfolder=av_subfolder
        )
        loader_dict = loader.to_dict()
        print(f"✓ Loaded {len(loader_dict)} retinas from dataset", file=sys.stderr)
    except Exception as e:
        print(f"✗ ERROR: Failed to load dataset with VascX!", file=sys.stderr)
        print(f"  Error: {e}", file=sys.stderr)
        raise
    
    # Extract features in parallel
    print(f"\n{'='*70}", file=sys.stderr)
    print(f"Extracting features (feature_set='{feature_set}', n_jobs={n_jobs})...", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)
    
    try:
        res = extract_in_parallel(loader_dict, feature_set, n_jobs=n_jobs)
    except Exception as e:
        print(f"✗ ERROR: Feature extraction failed!", file=sys.stderr)
        print(f"  Error: {e}", file=sys.stderr)
        raise
    
    # Ensure result is DataFrame
    if not isinstance(res, pd.DataFrame):
        res = pd.DataFrame(res)
    
    print(f"✓ Extracted features for {len(res)} images", file=sys.stderr)
    print(f"  Feature columns: {len(res.columns)}", file=sys.stderr)
    
    # Save results
    print(f"\n{'='*70}", file=sys.stderr)
    print(f"Saving results to {out_file}...", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)
    
    res.to_csv(out_file, sep=sep, na_rep=na_rep, index=True)
    print(f"✓ Results saved successfully", file=sys.stderr)
    print(f"\n{'='*70}", file=sys.stderr)
    print(f"Feature extraction complete!", file=sys.stderr)
    print(f"{'='*70}\n", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n{'='*70}", file=sys.stderr)
        print(f"FATAL ERROR", file=sys.stderr)
        print(f"{'='*70}", file=sys.stderr)
        print(f"{e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
