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

This script contains functions for Visual Odometry techniques used in BlossomNav
as well as functions for MonoNav that were reproduced with permission.

"""
 
import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
import statistics as st
import yaml, json

def read_yaml(file_path='config.yaml'):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def camera_intrinsics():
    """
    Obtains the camera intrinsics data from some .yaml file
    returns a matrix configuration of the data
    """
    config = read_yaml()
    fx = config['fx']
    fy = config['fy']
    cx = config['cx']
    cy = config['cy']

    # Camera Intrinsics
    return np.array([[fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]])

def homography_error(H, points1, points2):
    """
    Calculate the symmetric transfer error of a homography matrix H.

    H: Homography matrix (3x3)
    points1: Array of original points (Nx2)
    points2: Array of transformed points (Nx2)

    Returns the symmetric transfer error as a scalar value.
    """
    # Convert points to homogeneous coordinates (add a column of ones)
    points1_h = np.hstack([points1, np.ones((len(points1), 1))])
    points2_h = np.hstack([points2, np.ones((len(points2), 1))])

    # Transform points using the homography matrix
    transformed_points1 = np.dot(H, points1_h.T).T
    transformed_points2 = np.dot(np.linalg.inv(H), points2_h.T).T

    # Calculate the distances between original and transformed points
    errors1 = np.linalg.norm(points2_h - transformed_points1)
    errors2 = np.linalg.norm(points1_h - transformed_points2)

    # Compute the symmetric transfer error as the average of errors1 and errors2
    symmetric_error = np.mean(errors1 + errors2)

    return symmetric_error

def essential_error(E, points1, points2):
    """
    Calculate the symmetric transfer error of a essential matrix E.

    E: Essential matrix (3x3)
    points1: Array of original points (Nx2)
    points2: Array of transformed points (Nx2)

    Returns the symmetric transfer error as a scalar value.
    """
    # Obtain the camera intrinsics
    K = camera_intrinsics()

    # Convert points to homogeneous coordinates (add a column of ones)
    points1_h = np.hstack([points1, np.ones((len(points1), 1))])
    points2_h = np.hstack([points2, np.ones((len(points2), 1))])

    # Find the fundamental matrix
    F = np.dot(np.dot(np.transpose(np.linalg.inv(K)), E), np.linalg.inv(K))
    
    # Find all the epipolar lines using the fundamental matrix
    epipolarlines_1 = np.dot(F, points1_h.T).T
    epipolarlines_2 = np.dot(np.transpose(F), points2_h.T).T

    err_sums = []
    # Calculate the distances between original and transformed points
    for i in range(len(points1_h)):
        error2 = (epipolarlines_1[i][0] * points2[i][0] + epipolarlines_1[i][1] * points2[i][1] + epipolarlines_1[i][2]) ** 2 / (points2[i][0] ** 2 + points2[i][1] ** 2)
        error1 = (epipolarlines_2[i][0] * points1[i][0] + epipolarlines_2[i][1] * points1[i][1] + epipolarlines_2[i][2]) ** 2 / (points1[i][0] ** 2 + points1[i][1] ** 2)
        err_sums.append(error1 + error2)

    # Compute the symmetric transfer error as the average of errors1 and errors2
    symmetric_error = st.mean(err_sums)

    return symmetric_error

def match_images(file_path_1, file_path_2, draw=False):
    """
    Extracts ORB keypoints from the two frames and matches these keypoints

    file_path_1: File path to image 1 (ground truth image)
    file_path_2: File path to image 2
    draw: Draws the matches onto the images so that you can see the matches. Default has been set to False

    returns the matched points as two list of coordinates
    """
    # Read the two images
    img1 = cv2.imread(file_path_1)
    img2 = cv2.imread(file_path_2)

    # Convert the images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Initialize the ORB detector
    orb = cv2.ORB_create(nfeatures=1000)

    # Detect key points and extract descriptors
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    image_with_keypoints1 = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0), flags=0)
    image_with_keypoints2 = cv2.drawKeypoints(img2, kp2, None, color=(0, 255, 0), flags=0)

    cv2.imshow("img1", image_with_keypoints1)
    cv2.imshow("img2", image_with_keypoints2)

    # Match the descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort the matches by their confidence level
    matches = sorted(matches, key=lambda x: x.distance)

    # Choose a confidence level
    confidence_level = 0.30

    # Keep only the matches with confidence level above the chosen threshold
    good_matches = [match for match in matches if match.distance < confidence_level * max(match.distance for match in matches)]

    img1_pts = np.array([kp1[m.queryIdx].pt for m in good_matches])
    img2_pts = np.array([kp2[m.trainIdx].pt for m in good_matches])

    # Draws the good matches on the images if draw is true
    if draw:
        image_with_good_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Display the image with good matches
        cv2.imshow("matching", image_with_good_matches)

        # Wait for a key press, but allow Ctrl+C to terminate the program
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or 'q' to quit
                break

        # Close all windows
        cv2.destroyAllWindows()

    return img1_pts, img2_pts

def calculate_E_or_H(file_path_1, file_path_2):
    """
    Calculates, chooses, and decomposes the Essential Matrix or the Homography based on symmetric transfer error.

    file_path_1: File path to image 1 (ground truth image)
    file_path_2: File path to image 2

    Returns two matrices, the Rotation (R) and Translation (t) matrices in this order.
    """
    config = read_yaml()
    # get the camera intrinsics from config.yaml
    K = camera_intrinsics()

    # obtain matched points from match_images function
    img1_pts, img2_pts = match_images(file_path_1, file_path_2, draw=False)

    # Calculate the essential matrix
    E, Essential_Mask = cv2.findEssentialMat(img1_pts,
                                img2_pts, K,
                                method=cv2.RANSAC, prob=0.999, threshold=3)

    H, Homography_Mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC,5.0)

    # computing Rotational (R) matrix and Translational (t) matrix from Essential Matrix (E)
    _, R_E, t_E, _ = cv2.recoverPose(E, img1_pts, img2_pts)

    _, R_H, t_H, N_H = cv2.decomposeHomographyMat(H, K)

    # Convert keypoint coordinates to an input filterHomographyDecompByVisibleRefPoints likes
    kp1_pts_float32 = np.array(img1_pts, dtype=np.float32)[:, np.newaxis, :]
    kp2_pts_float32 = np.array(img2_pts, dtype=np.float32)[:, np.newaxis, :]

    # Find the two best Rotation and translation matrices from the four different solutions in the Homography Matrix
    filtered = cv2.filterHomographyDecompByVisibleRefpoints(R_H, N_H, kp1_pts_float32, kp2_pts_float32)
    print("Symmetric Error for Essential Matrix", essential_error(E, img1_pts, img2_pts))
    print("Symmetric Error for Homography Matrix", homography_error(H, img1_pts, img2_pts))

    if essential_error(E, img1_pts, img2_pts) < homography_error(H, img1_pts, img2_pts):
        return R_E, t_E
    elif essential_error(E, img1_pts, img2_pts) > homography_error(H, img1_pts, img2_pts):
        return R_H, t_H
    else:
        print("Error. Moving To Next Frame!")

def rotation_matrix_to_euler_angles(R):
    """
    Convert a rotation matrix to Euler angles (pitch, yaw, roll).
    
    R: 3x3 rotation matrix

    Returns the pitch, yaw, and roll angles in radians.
    """
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)

    singular = sy < 1e-6

    if not singular:
        yaw = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        roll = np.arctan2(R[1, 0], R[0, 0])
    else:
        yaw = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        roll = 0

    return pitch, yaw, roll

def convert_Rt_Open3D(R, t):
    # Constructing the 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T

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
Compute depth from an RGB image using ZoeDepth
Returns depth_numpy (uint16 in mm), depth_colormap (for visualization)
"""
def compute_depth(color, zoe):
    # Compute depth
    depth = zoe.infer_pil(color, output_type="tensor")  # as torch tensor
    depth_numpy = np.asarray(depth) # Convert to numpy array
    depth_numpy = 1000*depth_numpy # Convert to mm
    depth_numpy = depth_numpy.astype(np.uint16) # Convert to uint16

    # Save images and depth array
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_numpy, alpha=0.03), cv2.COLORMAP_JET)

    return depth_numpy, depth_colormap

"""
Read in the intrinsics.json file and return the camera matrix and distortion coefficients
"""
def get_calibration_values(camera_calibration_path):
    # Load the camera calibration file
    with open(camera_calibration_path, "r") as json_file:
        data = json.load(json_file)
    mtx = np.array(data['CameraMatrix'])
    dist = np.array(data['DistortionCoefficients'])
    return mtx, dist

"""
Transform the raw image to match the kinect image: dimensions and intrinsics.
This involves resizing the image, scaling the camera matrix, and undistorting the image.
"""
def transform_image(image, mtx, dist, kinect):
    if image.shape[0] != kinect.height or image.shape[1] != kinect.width:
        # Resize the camera matrix to match new dimensions
        scale_vec = np.array([kinect.width / image.shape[1], kinect.height / image.shape[0], 1]).reshape((3,1))
        mtx = mtx * scale_vec
        # Resize image to match the kinect dimensions & new intrinsics
        image = cv2.resize(image, (kinect.width, kinect.height))
    # Transform to the kinect camera matrix
    transformed_image = cv2.undistort(np.asarray(image), mtx, dist, None, kinect.intrinsic_matrix)
    return transformed_image