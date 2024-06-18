"""
__________.__                                      _______               
\______   \  |   ____  ______ __________   _____   \      \ _____ ___  __
 |    |  _/  |  /  _ \/  ___//  ___/  _ \ /     \  /   |   \\__  \\  \/ /
 |    |   \  |_(  <_> )___ \ \___ (  <_> )  Y Y  \/    |    \/ __ \\   / 
 |______  /____/\____/____  >____  >____/|__|_|  /\____|__  (____  /\_/  
        \/                \/     \/            \/         \/     \/      

Copyright (c) 2024 Interactions Lab
License: MIT
Authors: Anthony Song and Nathan Dennler, University of Southern California
Project Page: https://github.com/interaction-lab/BlossomNav.git

This script allows users to merge multiple .jpeg images into a videos

"""
import cv2
import os

def merge_video(frame_folder, output_path, fps):
    frame_files = sorted([f for f in os.listdir(frame_folder) if f.endswith('.jpg')])
    frame = cv2.imread(os.path.join(frame_folder, frame_files[0]))
    height, width, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_file in frame_files:
        frame = cv2.imread(os.path.join(frame_folder, frame_file))
        out.write(frame)

    out.release()

# Example usage:
frame_folder = './image_preprocessing/kinect-rgb-images'
output_path = './videos/processed_video.mp4'
fps = 25  # Frames per second
merge_video(frame_folder, output_path, fps)