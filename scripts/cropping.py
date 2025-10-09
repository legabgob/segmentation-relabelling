from PIL import Image
import os

# Test cropping 2656 x 1992 images to 1992x1992 center square
img = Image.open("./data/Fundus-AVSeg/binary_masks/008_N.png").convert("L")
w, h = img.size
print(f"Original size: {w} x {h}")
left = (w - h) // 2
right = left + h
top = 0
bottom = h
img_cropped = img.crop((left, top, right, bottom))
# save test image
img_cropped.save("./test_cropped.png")