import numpy as np
from utils.utils import calculate_E_or_H

'''
data = np.load("./data/pixel-depth-images/pixel_frame-000035.depth.npy")

data1 = np.load("./data/pixel-depth-images/pixel_frame-000036.depth.npy")

data2 = np.load("./data/pixel-depth-images/pixel_frame-000037.depth.npy")

print(data.shape)

start_row, end_row = 0, 480   # Extract rows from index 1 to 3
start_col, end_col = 0, 640   # Extract columns from index 2 to 4

#start_row, end_row = 150, 330   # Extract rows from index 1 to 3
#start_col, end_col = 200, 480   # Extract columns from index 2 to 4

data = data[start_row:end_row, start_col:end_col]

data1 = data1[start_row:end_row, start_col:end_col]

data2 = data2[start_row:end_row, start_col:end_col]


value = np.median(data)
max = np.max(data)

value1 = np.median(data1)
max1 = np.max(data1)

value2 = np.median(data2)
max2 = np.max(data2)


print("Array data:\n", data)
print("Total Sum:\n", value)
print("Max Depth:\n", max)

print("Array data:\n", data1)
print("Total Sum:\n", value1)
print("Max Depth:\n", max1)

print("Array data:\n", data1)
print("Total Sum:\n", value1)
print("Max Depth:\n", max1)
'''

file_path1 = "./data/pixel-images/pixel_frame-000001.rgb.jpg"
file_path2 = "./data/pixel-images/pixel_frame-000009.rgb.jpg"
file_path3 = "./data/pixel-images/down.jpg"
file_path4 = "./data/pixel-images/up.jpg"
print(calculate_E_or_H(file_path4, file_path3, draw=True))