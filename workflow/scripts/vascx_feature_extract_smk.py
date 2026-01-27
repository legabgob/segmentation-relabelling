# scripts/vascx_feature_extract_smk.py
from pathlib import Path
import pandas as pd

from vascx.fundus.loader import RetinaLoader
from vascx.utils.analysis import extract_in_parallel


# Snakemake inputs/outputs/params
ds_dir = Path(str(snakemake.input.ds_dir))          # the seg_legacy folder
out_file = Path(str(snakemake.output.features))    # deterministic output path

n_jobs = int(getattr(snakemake.params, "n_jobs", 64))
feature_set = str(getattr(snakemake.params, "feature_set", "bergmann"))
av_subfolder = str(getattr(snakemake.params, "av_subfolder", "av"))

sep = str(getattr(snakemake.params, "sep", "\t"))
na_rep = str(getattr(snakemake.params, "na_rep", "NaN"))

# Minimal validation (helps catch path mistakes early)
if not ds_dir.exists():
    raise FileNotFoundError(f"Dataset folder does not exist: {ds_dir}")

if not (ds_dir / "original").exists():
    raise FileNotFoundError(f"Missing 'original/' inside {ds_dir} (expected a link or folder)")

if not (ds_dir / av_subfolder).exists():
    raise FileNotFoundError(f"Missing '{av_subfolder}/' inside {ds_dir}")

out_file.parent.mkdir(parents=True, exist_ok=True)

# Load + extract
loader = RetinaLoader.from_folder(ds_dir, av_subfolder=av_subfolder)
res = extract_in_parallel(loader.to_dict(), feature_set, n_jobs=n_jobs)

# Save deterministically (Snakemake output path)
# res is typically a DataFrame already, but ensure it:
if not isinstance(res, pd.DataFrame):
    res = pd.DataFrame(res)

res.to_csv(out_file, sep=sep, na_rep=na_rep, index=True)

print(f"[vascx_feature_extract] ds={ds_dir} feature_set={feature_set} n_jobs={n_jobs} -> {out_file}")

