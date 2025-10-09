# Minimal: pick up to 5 per category (A,N,G,D), show original vs refined side by side, save one figure.

from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# Folders
orig_dir = Path("/SSD/home/gabriel/rrwnet/data/FIVES/train/downsampled/segs/")
pred_dir = Path("/SSD/home/gabriel/rrwnet/refined_predictions/FIVES/downsampled/")
out_path = Path("/SSD/home/gabriel/rrwnet/figures/comparison_grid.png")
out_path.parent.mkdir(parents=True, exist_ok=True)

# Collect pairs (first 5 per category if available)
pairs = []
for cat in ["A", "N", "G", "D"]:
    for p in sorted(orig_dir.glob(f"*_{cat}.png"))[:5]:
        q = pred_dir / p.name
        if q.exists():
            pairs.append((p, q))

if not pairs:
    raise SystemExit("No matching files found.")

# Plot
rows = len(pairs)
fig, axes = plt.subplots(rows, 2, figsize=(10, 3*rows), dpi=150)
if rows == 1:
    axes = [axes]  # unify indexing

for i, (p_orig, p_ref) in enumerate(pairs):
    im_o = Image.open(p_orig).convert("RGB")
    im_r = Image.open(p_ref).convert("RGB")
    if im_r.size != im_o.size:
        im_r = im_r.resize(im_o.size, Image.NEAREST)

    axes[i][0].imshow(im_o); axes[i][0].set_title(f"{p_orig.name} - Original"); axes[i][0].axis("off")
    axes[i][1].imshow(im_r); axes[i][1].set_title(f"{p_ref.name} - Refined");   axes[i][1].axis("off")

plt.tight_layout()
fig.savefig(out_path, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")
