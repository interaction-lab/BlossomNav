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
poses_dir = os.path.join(data_dir, camera_source + "-poses")
os.mkdir(poses_dir) if not os.path.exists(poses_dir) else None

# Figure out how many images are in folder by counting .jpg files
end_frame = len([name for name in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, name)) and name.endswith(".jpg")])

for frame in range(1, end_frame): # first image has been set as ground truth, can't obtain pose data for it
    ground_truth = images_dir + "/" + camera_source + "_frame-%06d.rgb.jpg"%(frame - 1)
    new_image = images_dir + "/" + camera_source + "_frame-%06d.rgb.jpg"%(frame)
    R, t = calculate_E_or_H(ground_truth, new_image)
    Open3D_matrix = convert_Rt_Open3D(R, t)
    file_name = camera_source + "_frame-%06d.pose.txt"%(frame)
    path = os.path.join(poses_dir, file_name)
    np.savetxt(path, Open3D_matrix)