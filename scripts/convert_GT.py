import os
import argparse
import numpy as np
from PIL import Image

# Define color mappings: (R,G,B) -> (R,G,B)
MAPPINGS = [
    ((255,   0,   0), (255,   0,   255)),  # arteries
    ((  0, 0,   255), (  0, 255,   255)),  # veins
    ((  0,   255, 0), (  255,   255, 255)),  # crossings
    ((  255,   255,   255), (  0,   0,   255)),  # vessels
]

def rgb_labels_to_rgb(arr: np.ndarray) -> np.ndarray:
    # Recolor RGB labels to new RGB colors by exact matching.
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("Input array must be HxWx3 (RGB)")

    out = arr.copy()
    for src, dst in MAPPINGS:
        src = np.array(src, dtype=np.uint8)
        mask = (arr == src).all(axis=2)
        out[mask] = np.array(dst, dtype=np.uint8)
    return out

def convert(input_path: str, output_path: str):
    # Load the RGB image
    img = Image.open(input_path).convert('RGB')
    arr = np.array(img)
    rgb = rgb_labels_to_rgb(arr)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(output_path)
    print(f"Saved {output_path}")

def main():
    ap = argparse.ArgumentParser(description="Recolor RGB label images via exact color mapping.")
    ap.add_argument("--image", help="Path to one RGB label image.")
    ap.add_argument("--out", help="Output path for the converted RGB image.")
    ap.add_argument("--in_dir", help="Directory of RGB labels (batch).")
    ap.add_argument("--out_dir", help="Output directory for RGB results (batch).")
    ap.add_argument("--ext", default=".png", help="Filename extension to match in batch mode (e.g., .png, .tif).")
    args = ap.parse_args()

    if args.image and args.out:
        convert(args.image, args.out)
        return

    if args.in_dir and args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        for name in os.listdir(args.in_dir):
            if not name.lower().endswith(args.ext.lower()):
                continue
            inp = os.path.join(args.in_dir, name)
            out = os.path.join(args.out_dir, name.rsplit(".", 1)[0] + ".png")
            convert(inp, out)
        return

if __name__ == "__main__":
    main()