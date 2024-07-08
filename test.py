import numpy as np
from utils.utils import match_images, find_similar_regions


data = "./data/pixel-images/pixel_frame-000023.rgb.jpg"

data1 = "./data/pixel-images/pixel_frame-000024.rgb.jpg"

img1, img2 = match_images(data, data1, draw=True)

a, b, c, d = find_similar_regions(img1, img2)

print(a)
print("--------------------------------")
print(b)
print("--------------------------------")
print(c)
print("--------------------------------")
print(d)