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

from utils.utils import compute_depth, read_yaml, get_calibration_values, transform_image

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
kinect_img_dir = os.path.join(data_dir, "kinect-rgb-images")
os.mkdir(kinect_img_dir) if not os.path.exists(kinect_img_dir) else None
kinect_depth_dir = os.path.join(data_dir, "kinect-depth-images")
os.mkdir(kinect_depth_dir) if not os.path.exists(kinect_depth_dir) else None
print("Saving Depth images to: ", kinect_depth_dir)

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
end_frame = end_frame - 1 # Ignore last image since sometimes pose information is not saved for it

start_time = time.time()

for frame_number in range(0, end_frame): # ignore first frame as that has been set to ground truth
    print("Applying ZoeDepth to:  %d/%d"%(frame_number+1,end_frame))
    filename = rgb_dir + "/" + camera_source + "_frame-%06d.rgb.jpg"%(frame_number)
    # Read in image with Pillow and convert to RGB
    crazyflie_rgb = Image.open(filename)#.convert("RGB")  # load
    # Resize, Undistort, and Warp image to kinect's dimensions and intrinsics
    kinect_rgb = transform_image(np.asarray(crazyflie_rgb), mtx, dist, kinect)
    kinect_rgb = cv2.cvtColor(kinect_rgb, cv2.COLOR_BGR2RGB)
    # Compute depth
    depth_numpy, depth_colormap = compute_depth(kinect_rgb, zoe)
    # Save images
    cv2.imwrite(kinect_img_dir + "/kinect_frame-%06d.rgb.jpg"%(frame_number), kinect_rgb)
    cv2.imwrite(kinect_depth_dir + "/" + "kinect_frame-%06d.depth.jpg"%(frame_number), depth_colormap)
    np.save(kinect_depth_dir + "/" + "kinect_frame-%06d.depth.npy"%(frame_number), depth_numpy) # saved in meters

print("Time to compute depth for %d images: %f"%(end_frame, time.time()-start_time))
# On Nvidia GeForce RTX 4090: 13.6 s for 80 images