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

def split_video(video_path : str, output_folder : str, camera_type : str, frame_interval = 0):
    # Open the video files
    cap = cv2.VideoCapture(video_path)

    # Get the frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate the frame interval in terms of frame count
    frame_interval_count = int(frame_interval * fps)
    
    # Create a counter for frame numbers
    frame_count = 0
    
    # Loop through the video frames
    while True:
        # Read the next frame
        success, frame = cap.read()
        
        # If no more frames, break the loop
        if not success:
            break
        
        # Save the frame as a JPG image if it's at the specified interval
        frame_path = f"{output_folder}/{camera_type}_frame-{frame_count:06d}.rgb.jpg"
        cv2.imwrite(frame_path, frame)

        # Increment frame counter
        frame_count += 1
    
    # Release the video capture object
    cap.release()

# Path to the input video
video_path = "./test_vid.mp4"

# Camera type
camera_type = "pixel"

# Path to the output folder
output_folder = f"{camera_type}-rgb-images"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Split frames every x seconds (e.g., every 5 seconds)
frame_interval = 5

# Call the function to split the video into frames
split_video(video_path, output_folder, camera_type, frame_interval)