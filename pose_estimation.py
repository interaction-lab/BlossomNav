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

The following is saved to file ():
│   ├── <[camera_source]-poses> # camera poses calculated from pictures in [camera_source]-images folder


"""

import os
import numpy as np
from utils.utils import read_yaml, calculate_E_or_H, convert_Rt_Open3D

CONFIG_PATH = "config.yaml"
config = read_yaml(CONFIG_PATH)
data_dir = config["data_dir"]
camera_source = config["camera_source"]

images_dir = os.path.join(data_dir, camera_source + "-images")
depth_dir = os.path.join(data_dir, camera_source + "-depth-images")
poses_dir = os.path.join(data_dir, camera_source + "-poses")
os.mkdir(poses_dir) if not os.path.exists(poses_dir) else None

# Figure out how many images are in folder by counting .jpg files
end_frame = len([name for name in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, name)) and name.endswith(".jpg")])

curr_state = convert_Rt_Open3D(np.eye(3), np.zeros(3)) # 1, 3 if no work

for frame in range(1, end_frame - 2): # first image has been set as ground truth, can't obtain pose data for it
    ground_truth = images_dir + "/" + camera_source + "_frame-%06d.rgb.jpg"%(frame - 1)
    new_image = images_dir + "/" + camera_source + "_frame-%06d.rgb.jpg"%(frame)
    ground_truth_depth = np.load(depth_dir + "/" + camera_source + "_frame-%06d.depth.npy"%(frame - 1))
    new_image_depth = np.load(depth_dir + "/" + camera_source + "_frame-%06d.depth.npy"%(frame))
    print(frame - 1, frame)
    R, t = calculate_E_or_H(ground_truth, new_image)
    print("--------------------------------------")
    distance_traveled = np.median(new_image_depth - ground_truth_depth)
    scaled_t = distance_traveled * 0.00001 * t
    Open3D_matrix = convert_Rt_Open3D(R, scaled_t)
    Open3D_matrix[0][3], Open3D_matrix[1][3] = -Open3D_matrix[0][3],-Open3D_matrix[1][3] # Open3D is Right, Down, Front
                                                                                            # OpenCV is Left, Up, Front
    if np.median(new_image_depth) < np.median(ground_truth_depth) and Open3D_matrix[2][3] < 0:
        Open3D_matrix[2][3] = -Open3D_matrix[2][3]
    curr_state = curr_state @ Open3D_matrix
    file_name = camera_source + "_frame-%06d.pose.txt"%(frame-1)
    path = os.path.join(poses_dir, file_name)
    np.savetxt(path, curr_state)