import argparse
from pathlib import Path
import os

import torch
import pandas as pd

from rtnls_fundusprep.utils import preprocess_for_inference
from rtnls_inference import (
    HeatmapRegressionEnsemble,
    SegmentationEnsemble,
)
from vascx.fundus.loader import RetinaLoader
from vascx.utils.analysis import extract_in_parallel
from datetime import date


def print_usage_examples():
    """Print usage examples for the pipeline"""
    examples = """
╔════════════════════════════════════════════════════════════════╗
║                    USAGE EXAMPLES                              ║
╚════════════════════════════════════════════════════════════════╝

Basic usage (run all steps):
  python run_full_pipeline.py /path/to/dataset

Expected dataset structure:
  /path/to/dataset/
    ├── original/          (original fundus images)
    ├── rgb/               (created by preprocessing)
    ├── ce/                (created by preprocessing)
    ├── av/                (created by segmentation)
    ├── discs/             (created by segmentation)
    └── extracted_features/ (created by feature extraction)

Skip specific steps:
  python run_full_pipeline.py /path/to/dataset --skip-preprocessing
  python run_full_pipeline.py /path/to/dataset --skip-segmentation
  python run_full_pipeline.py /path/to/dataset --skip-feature-extraction

Customize parallel jobs and device:
  python run_full_pipeline.py /path/to/dataset --n-jobs 32 --device cuda:1
  python run_full_pipeline.py /path/to/dataset --n-jobs 16 --device cpu

All options combined:
  python run_full_pipeline.py /path/to/dataset \\
    --n-jobs 32 \\
    --device cuda:0 \\
    --skip-preprocessing

For help:
  python run_full_pipeline.py --help
    """
    print(examples)


def run_preprocessing(ds_path, n_jobs=64):
    """Step 0: Preprocessing"""
    print(f"\n{'='*60}")
    print("STEP 0: Running preprocessing...")
    print(f"{'='*60}")
    
    files = list((ds_path / "original").glob("*"))
    print(f"Found {len(files)} images to preprocess")

    bounds = preprocess_for_inference(
        files,
        rgb_path=ds_path / "rgb",
        ce_path=ds_path / "ce",
        n_jobs=n_jobs,
    )

    df_bounds = pd.DataFrame(bounds).set_index("id")
    df_bounds.to_csv(ds_path / "meta.csv")
    print(f"Preprocessing complete. Metadata saved to {ds_path / 'meta.csv'}")


def run_segmentation(ds_path, n_jobs=64, device=None):
    """Step 1: Segmentation"""
    print(f"\n{'='*60}")
    print("STEP 1: Running segmentation...")
    print(f"{'='*60}")
    
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # input folders
    rgb_path = ds_path / "rgb"
    ce_path = ds_path / "ce"

    # output folders
    av_path = ds_path / "av"
    discs_path = ds_path / "discs"

    rgb_paths = sorted(list(rgb_path.glob("*.png")))
    ce_paths = sorted(list(ce_path.glob("*.png")))
    paired_paths = list(zip(rgb_paths, ce_paths))
    print(f"Found {len(paired_paths)} image pairs")

    # Set model path
    os.environ["RTNLS_MODEL_RELEASES"] = "/SSD/home/cbg/git/rtnls_vascx_models/vascx_models"

    # A-V segmentation
    print("Running A-V segmentation...")
    av_ensemble = SegmentationEnsemble.from_release("av_july24.pt").to(device)
    av_ensemble.predict_preprocessed(paired_paths, dest_path=av_path, num_workers=n_jobs)

    # Optic disc segmentation
    print("Running optic disc segmentation...")
    disc_ensemble = SegmentationEnsemble.from_release("disc_july24.pt").to(device)
    disc_ensemble.predict_preprocessed(paired_paths, dest_path=discs_path, num_workers=n_jobs)

    # Fovea regression
    print("Running fovea detection...")
    fovea_ensemble = HeatmapRegressionEnsemble.from_release("fovea_july24.pt").to(device)
    df = fovea_ensemble.predict_preprocessed(paired_paths, num_workers=n_jobs)
    df.columns = ["mean_x", "mean_y"]
    df.to_csv(ds_path / "fovea.csv")
    
    print(f"Segmentation complete. Results saved to {av_path}, {discs_path}, and {ds_path / 'fovea.csv'}")


def run_feature_extraction(ds_path, n_jobs=64):
    """Step 2: Feature Extraction"""
    print(f"\n{'='*60}")
    print("STEP 2: Running feature extraction...")
    print(f"{'='*60}")
    
    destination_path = Path(ds_path.parent, 'extracted_features')
    filename_path = Path(destination_path, str(date.today()) + '_vascx_features.csv')

    # Create output directory if it doesn't exist
    if not destination_path.exists():
        destination_path.mkdir(parents=True)
        print(f"Created output directory: {destination_path}")

    loader = RetinaLoader.from_folder(ds_path, av_subfolder='av')
    print(f"Loaded retina data from {ds_path}")
    
    print("Extracting features in parallel...")
    res = extract_in_parallel(loader.to_dict(), "bergmann", n_jobs=n_jobs)
    res.to_csv(filename_path, sep='\t', na_rep='NaN', index=True)
    
    print(f"Feature extraction complete. Results saved to {filename_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run the full vascx pipeline: preprocessing, segmentation, and feature extraction",
        epilog="Use --examples to see usage examples"
    )
    parser.add_argument(
        "--examples",
        action="store_true",
        help="Show usage examples and exit"
    )
    parser.add_argument(
        "ds_path",
        type=Path,
        nargs="?",
        help="Path to the dataset directory"
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip the preprocessing step"
    )
    parser.add_argument(
        "--skip-segmentation",
        action="store_true",
        help="Skip the segmentation step"
    )
    parser.add_argument(
        "--skip-feature-extraction",
        action="store_true",
        help="Skip the feature extraction step"
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=64,
        help="Number of parallel jobs (default: 64)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for inference (default: cuda:0)"
    )

    args = parser.parse_args()

    # Show examples if requested
    if args.examples:
        print_usage_examples()
        return

    # Validate that ds_path was provided
    if args.ds_path is None:
        parser.print_help()
        print("\nError: ds_path is required. Use --examples to see usage examples.")
        return

    # Validate dataset path
    ds_path = args.ds_path
    if not ds_path.exists():
        raise ValueError(f"Dataset path does not exist: {ds_path}")

    device = torch.device(args.device)
    print(f"\nStarting vascx pipeline for: {ds_path}")

    try:
        if not args.skip_preprocessing:
            run_preprocessing(ds_path, n_jobs=args.n_jobs)

        if not args.skip_segmentation:
            run_segmentation(ds_path, n_jobs=args.n_jobs, device=device)

        if not args.skip_feature_extraction:
            run_feature_extraction(ds_path, n_jobs=args.n_jobs)

        print(f"\n{'='*60}")
        print("PIPELINE COMPLETE!")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"\nERROR: {e}")
        raise


if __name__ == "__main__":
    main()
