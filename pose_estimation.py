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
import time

from utils.utils import read_yaml, calculate_E_or_H, convert_Rt_Open3D, calculate_distance_between_images, rotation_matrix_to_euler_angles

CONFIG_PATH = "config.yaml"
config = read_yaml(CONFIG_PATH)
data_dir = config["data_dir"]
camera_source = config["camera_source"]
camera_info = config["camera_calibration_path"]

images_dir = os.path.join(data_dir, camera_source + "-images")
depth_dir = os.path.join(data_dir, camera_source + "-depth-images")
poses_dir = os.path.join(data_dir, camera_source + "-poses")
os.mkdir(poses_dir) if not os.path.exists(poses_dir) else None

# Figure out how many images are in folder by counting .jpg files
end_frame = len([name for name in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, name)) and name.endswith(".jpg")])

curr_state = convert_Rt_Open3D(np.eye(3), np.zeros(3)) # 1, 3 if no work

start_time = time.time()

for frame in range(1, end_frame - 2): # first image has been set as ground truth, can't obtain pose data for it
    ground_truth = images_dir + "/" + camera_source + "_frame-%06d.rgb.jpg"%(frame - 1)
    new_image = images_dir + "/" + camera_source + "_frame-%06d.rgb.jpg"%(frame)
    ground_truth_depth = np.load(depth_dir + "/" + camera_source + "_frame-%06d.depth.npy"%(frame - 1))
    new_image_depth = np.load(depth_dir + "/" + camera_source + "_frame-%06d.depth.npy"%(frame))
    ground_truth_d = ground_truth_depth.astype(np.int16)
    new_image_d = new_image_depth.astype(np.int16)
    R, t = calculate_E_or_H(ground_truth, new_image) # find Rotation and translation matrix
    print("---------------------------------------------------------")
    print(frame - 1, frame)
    distance_traveled = calculate_distance_between_images(ground_truth_depth, new_image_depth, (config["fx"], config["fy"]), (config["cx"], config["cy"]))
    scaled_t = distance_traveled * 0.001 * t
    pitch, yaw, roll = rotation_matrix_to_euler_angles(R)
    Open3D_matrix = convert_Rt_Open3D(R, scaled_t)
    Open3D_matrix[1][3] = -Open3D_matrix[1][3]

    delta_D = new_image_d - ground_truth_d
    print(pitch)
    if abs(pitch) > 0.20: # rolling at least 10+ degrees
        if (np.sum(delta_D > 0) / delta_D.size) > 0.50 and Open3D_matrix[2][3] < 0:
            print("Turn - Reversed Forward & Backward")
            Open3D_matrix[2][3] =  -Open3D_matrix[2][3]
    if (np.sum(delta_D < 0) / delta_D.size) > 0.50 and Open3D_matrix[2][3] < 0:
        print("Nonturn - Reversed Forward & Backward")
        Open3D_matrix[2][3] = -Open3D_matrix[2][3]
    print(Open3D_matrix)
    curr_state = curr_state @ Open3D_matrix
    file_name = camera_source + "_frame-%06d.pose.txt"%(frame-1)
    path = os.path.join(poses_dir, file_name)
    np.savetxt(path, curr_state)

print("Time to compute poses: %f"%(time.time()-start_time))