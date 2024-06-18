"""
__________.__                                      _______               
\______   \  |   ____  ______ __________   _____   \      \ _____ ___  __
 |    |  _/  |  /  _ \/  ___//  ___/  _ \ /     \  /   |   \\__  \\  \/ /
 |    |   \  |_(  <_> )___ \ \___ (  <_> )  Y Y  \/    |    \/ __ \\   / 
 |______  /____/\____/____  >____  >____/|__|_|  /\____|__  (____  /\_/  
        \/                \/     \/            \/         \/     \/      

Copyright (c) 2024 Interactions Lab
License: MIT
Authors: Anthony Song and Nathan Dennler, Cornell University & University of Southern California
Project Page: https://github.com/interaction-lab/BlossomNav.git

This script contains functions for Visual Odometry techniques used in BlossomNav

The following are saved to file:
│   ├── <kinect_rgb_images> # images transformed to match kinect intrinsics
│   ├── <kinect_depth_images> # estimated depth (.npy for fusion and .jpg for visualization)

"""

import os
import numpy as np
from utils.utils import read_yaml, calculate_E_or_H, rotation_matrix_to_euler_angles

CONFIG_PATH = "config.yaml"
config = read_yaml(CONFIG_PATH)
data_dir = config["paths"]["data_dir"]
camera_source = config["camera"]["name"]

images_dir = os.path.join(data_dir, camera_source + "-images")
poses_dir = os.path.join(data_dir, camera_source + "-poses")
os.mkdir(poses_dir) if not os.path.exists(poses_dir) else None

# Figure out how many images are in folder by counting .jpg files
end_frame = len([name for name in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, name)) and name.endswith(".jpg")])

for frame in range(1, end_frame): # first image has been set as ground truth, can't obtain pose data for it
    ground_truth = images_dir + "/" + camera_source + "_frame-%06d.rgb.jpg"%(frame - 1)
    new_image = images_dir + "/" + camera_source + "_frame-%06d.rgb.jpg"%(frame)
    R, t = calculate_E_or_H(ground_truth, new_image)
    pitch, yaw, roll = rotation_matrix_to_euler_angles(R)
    pitch_deg, yaw_deg, roll_deg = np.degrees(pitch), np.degrees(yaw), np.degrees(roll)
    print(pitch_deg)
    print(yaw)
    print(roll)
    print("----------------------")
