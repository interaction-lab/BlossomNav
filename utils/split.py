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

This script allows users to split a MP4 video into multiple .jpeg images

"""

import cv2
import os
import sys
from utils import read_yaml

def split_video(video_path : str, output_folder : str, camera_type : str, frame_interval = 10):
    # Open the video files
    cap = cv2.VideoCapture(video_path)

    # Get the frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create a counter for frame numbers
    frame_count = 0
    
    # Loop through the video frames
    while True:
        # Read the next frame
        success, frame = cap.read()
        
        # If no more frames, break the loop
        if not success:
            break
        
        if frame_count % frame_interval == 0:
            # Save the frame as a JPG image if it's at the specified interval
            temp = frame_count // frame_interval
            frame_path = f"{output_folder}/{camera_type}_frame-{temp:06d}.rgb.jpg"
            cv2.imwrite(frame_path, frame)

        # Increment frame counter
        frame_count += 1
    
    # Release the video capture object
    cap.release()

def main():
    # Check if enough arguments are provided
    if len(sys.argv) < 3:
        print("Usage: python split.py <video_path> <output_path> <frame_interval>")
        return
    
    # Get camera source
    CONFIG_PATH = "../config.yaml"
    config = read_yaml(CONFIG_PATH)
    camera_source = config["camera_source"]

    dir = "../data/pixel-images"
    os.mkdir(dir) if not os.path.exists(dir) else None

    # Get the arguments from the command line
    video_path = sys.argv[1]
    output_folder = sys.argv[2]
    frame_interval = int(sys.argv[3])

    split_video(video_path, output_folder, camera_source, frame_interval)

if __name__ == "__main__":
    main()