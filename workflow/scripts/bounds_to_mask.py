import os
import csv
import argparse
import ast 
import numpy as np
from PIL import Image

def make_circular_mask(h, w, cx, cy, r):
    # Create a circular mask with center (cx, cy) and radius r
    yy, xx = np.ogrid[:h, :w]
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    mask = dist2 <= r ** 2
    return mask

def parse_meta(meta_str):
    # Safely parse the metadata string into a dictionary
    meta = ast.literal_eval(meta_str)
    # Expected keys: 'hw' = (H, W), 'center' = (cx, cy), 'radius' = float
    return meta

def main():
    ap = argparse.ArgumentParser(description="Generate ROI binary masks (0/255) from bounds.csv with circle params.")
    ap.add_argument("--csv", required=True, help="Path to bounds.csv")
    ap.add_argument("--out_dir", required=True, help="Where to save ROI masks (PNG).")
    ap.add_argument("--img_dir", help="Optional: directory of actual images; if given, masks will be resized to match each image's size.")
    ap.add_argument("--img_ext", default=".png", help="Image extension in img-dir to match (e.g., .png, .jpg)")
    ap.add_argument("--center_order", choices=["xy", "yx"], default="xy",
                    help="Interpret 'center' as (x,y)=columns,rows (xy) or (y,x)=rows,columns (yx). Default xy.")
    ap.add_argument("--skip_if_flag_false", action="store_true",
                    help="Second CSV column is a boolean; skip rows where it is not True.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing masks.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True) 

    with open(args.csv, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:

            if len(row) < 3:
                print(f"[skip] malformed row: {row}")
                continue

            name = row[0].strip()                # e.g., "289_D"
            flag_str = row[1].strip()            # e.g., "True"
            meta_str = row[2].strip()

            if args.skip_if_flag_false:
                if flag_str.lower() not in ("true", "1", "yes"):
                    print(f"[skip] {name} flag={flag_str}")
                    continue

            try:
                meta = parse_meta(meta_str)
            except Exception as e:
                print(f"[skip] {name}: failed to parse meta: {e}")
                continue

            try:
                H, W = meta["hw"]          # (H, W)
                cx, cy = meta["center"]    # assume (x, y) unless overridden
                R = float(meta["radius"])
            except KeyError as e:
                print(f"[skip] {name}: missing key {e}")
                continue

            if args.center_order == "yx":
                # If the CSV uses (y,x), swap
                cy, cx = float(cx), float(cy)
            else:
                cx, cy = float(cx), float(cy)

            H, W = int(H), int(W)
            out_path = os.path.join(args.out_dir, f"{name}.png")
            if (not args.overwrite) and os.path.exists(out_path):
                print(f"[skip] exists: {out_path}")
                continue

            # 1) Build mask at native (H, W) from CSV
            mask = make_circular_mask(H, W, cx, cy, R)  # uint8 0/255

            # 2) Optionally resize to actual image size if images exist and might be different
            if args.img_dir:
                img_path = os.path.join(args.img_dir, f"{name}{args.img_ext}")
                if os.path.exists(img_path):
                    with Image.open(img_path) as im:
                        w_img, h_img = im.size  # PIL gives (W,H)
                    if (h_img, w_img) != (H, W):
                        # Nearest keeps mask binary & crisp
                        mask_img = Image.fromarray(mask, mode="L")
                        mask_img = mask_img.resize((w_img, h_img), resample=Image.NEAREST)
                        mask = np.array(mask_img, dtype=np.uint8)
                        print(f"[info] resized ROI for {name}: {(H,W)} -> {(h_img,w_img)}")
                else:
                    print(f"[warn] image not found for size match: {img_path}")

            # 3) Save mask (single-channel L, 0/255)
            Image.fromarray(mask, mode="L").save(out_path)
            print(f"[ok] {name}: saved {out_path}")

if __name__ == "__main__":
    main()

