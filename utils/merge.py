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
import sys
from utils import read_yaml

def merge_video(frame_folder, output_path, fps):
    os.mkdir(output_path) if not os.path.exists(output_path) else None
    frame_files = sorted([f for f in os.listdir(frame_folder) if f.endswith('.jpg')])
    frame = cv2.imread(os.path.join(frame_folder, frame_files[0]))
    height, width, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_file in frame_files:
        frame = cv2.imread(os.path.join(frame_folder, frame_file))
        out.write(frame)

    out.release()

def main():
    # Check if enough arguments are provided
    if len(sys.argv) < 3:
        print("Usage: python merge.py <frame_folder> <output_path> <fps>")
        return

    # Get the arguments from the command line
    frame_folder = sys.argv[1]
    output_path = sys.argv[2]
    fps = int(sys.argv[3])

    merge_video(frame_folder, output_path, fps)

if __name__ == "__main__":
    main()