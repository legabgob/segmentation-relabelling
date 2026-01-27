import os
import argparse
import numpy as np
from PIL import Image


def gray_labels_to_rgb(arr: np.ndarray) -> np.ndarray:
    # Convert grayscale labels to RGB colors

    if arr.ndim != 2:
        raise ValueError("Input array must be 2-dimensional")

    arr = arr.astype(np.int32, copy=False) # Make sure we work in integers

    H, W = arr.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)

    # Define each class
    arteries = (arr == 1)
    veins = (arr == 2)
    crossings = (arr == 3)

    rgb[arteries, 0] = 255  # Red channel for arteries
    rgb[arteries, 2] = 255
    rgb[veins, 1] = 255     # Green channel for veins
    rgb[veins, 2] = 255
    rgb[crossings] = [255, 255, 255] # White for crossings

    return rgb

def convert(input_path:str, output_path:str):
    # Load the grayscale image
    img = Image.open(input_path).convert('L')
    arr = np.array(img)
    rgb = gray_labels_to_rgb(arr)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(output_path)
    print(f"Saved {output_path}")

def main():
    ap = argparse.ArgumentParser(description="Convert grayscale label images {0,1,2,3} to RGB A/V/BV mapping.")
    ap.add_argument("--image", help="Path to one grayscale label image.")
    ap.add_argument("--out", help="Output path for the converted RGB image.")
    ap.add_argument("--in_dir", help="Directory of grayscale labels (batch).")
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
