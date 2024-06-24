# MIT License

# Copyright (c) 2023 Nate Simon

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Nate Simon

"""

This script reads in a stream of RGB images, transforms them to the Kinect intrinsics,
and estimates metric depth using ZoeDepth.

The following are saved to file:
│   ├── <[camera_source]-rgb-images> # images transformed to match kinect intrinsics
│   ├── <[camera_source]-depth-images> # estimated depth (.npy for fusion and .jpg for visualization)

"""

import time
import os
import sys
import torch
import cv2

# Add ZoeDepth to path
sys.path.insert(0, "ZoeDepth")
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

import open3d as o3d
from PIL import Image
import numpy as np

from utils.utils import read_yaml, compute_depth, get_calibration_values, transform_image

""""
This script runs a depth estimation model on a directory of RGB images and saves the depth images.
"""

# LOAD CONFIG
CONFIG_PATH = "config.yaml"
config = read_yaml(CONFIG_PATH)
data_dir = config["data_dir"] # parent directory to look for RGB images, and save depth images
camera_source = config["camera_source"] # what camera was used for the RGB images?
print("Loading" + camera_source + "images from: ", data_dir, ".")

# Set & create directories for images
rgb_dir = os.path.join(data_dir, camera_source + "-images")
img_dir = os.path.join(data_dir, camera_source + "-rgb-images")
os.mkdir(img_dir) if not os.path.exists(img_dir) else None
depth_dir = os.path.join(data_dir, camera_source + "-depth-images")
os.mkdir(depth_dir) if not os.path.exists(depth_dir) else None
print("Saving Depth images to: ", depth_dir)

# Load the calibration values
camera_calibration_path = config["camera_calibration_path"]
mtx, dist = get_calibration_values(camera_calibration_path)
# Kinect intrinsic matrix
kinect = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

# Load the ZoeDepth model
conf = get_config("zoedepth", config["zoedepth_mode"]) # NOTE: "eval" runs slightly slower, but is stated to be more metrically accurate
model_zoe = build_model(conf)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device is: ", DEVICE)
zoe = model_zoe.to(DEVICE)

# Figure out how many images are in folder by counting .jpg files
end_frame = len([name for name in os.listdir(rgb_dir) if os.path.isfile(os.path.join(rgb_dir, name)) and name.endswith(".jpg")])
end_frame = end_frame - 1

start_time = time.time()

for frame_number in range(1, end_frame): # ignore first frame as that has been set to ground truth
    print("Applying ZoeDepth to:  %d/%d"%(frame_number+1,end_frame))
    filename = rgb_dir + "/" + camera_source + "_frame-%06d.rgb.jpg"%(frame_number)
    # Read in image with Pillow and convert to RGB
    camera_source_rgb = Image.open(filename)#.convert("RGB")  # load
    # Resize, Undistort, and Warp image to kinect's dimensions and intrinsics
    kinect_rgb = transform_image(np.asarray(camera_source_rgb), mtx, dist, kinect)
    kinect_rgb = cv2.cvtColor(kinect_rgb, cv2.COLOR_BGR2RGB)
    # Compute depth
    depth_numpy, depth_colormap = compute_depth(kinect_rgb, zoe)
    # Save images
    cv2.imwrite(img_dir + "/" + camera_source +"_frame-%06d.rgb.jpg"%(frame_number-1), kinect_rgb)
    cv2.imwrite(depth_dir + "/" + camera_source + "_frame-%06d.depth.jpg"%(frame_number-1), depth_colormap)
    np.save(depth_dir +"/" + camera_source + "_frame-%06d.depth.npy"%(frame_number-1), depth_numpy) # saved in meters

print("Time to compute depth for %d images: %f"%(end_frame, time.time()-start_time))
# On Nvidia GeForce RTX 4090: 13.6 s for 80 images