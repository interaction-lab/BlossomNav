import numpy as np
from utils.utils import calculate_E_or_H


data = np.load("./data/pixel-depth-images/pixel_frame-000022.depth.npy")

data1 = np.load("./data/pixel-depth-images/pixel_frame-000021.depth.npy")

# Convert to a signed integer type (e.g., int16) to prevent overflow
data = data.astype(np.int16)
data1 = data1.astype(np.int16)

delta_D = data1 - data

print(delta_D)

print(np.sum(delta_D < 0) / delta_D.size > 0.50)