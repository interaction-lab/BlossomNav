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
        # Define the desired frame size
        size1 = (640, 480)
        size2 = (1280, 480)

        # Resize the frames
        resized_img1 = cv2.resize(image_with_keypoints1, size1, interpolation=cv2.INTER_LINEAR)
        resized_img2 = cv2.resize(image_with_keypoints2, size1, interpolation=cv2.INTER_LINEAR)

        cv2.imshow("img1", resized_img1)
        cv2.imshow("img2", resized_img2)

        image_with_good_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        resized_matches = cv2.resize(image_with_good_matches, size2, interpolation=cv2.INTER_LINEAR)

        # Display the image with good matches
        cv2.imshow("matching", resized_matches)

        # Wait for a key press, but allow Ctrl+C to terminate the program
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or 'q' to quit
                break

        # Close all windows
        cv2.destroyAllWindows()

    return img1_pts, img2_pts

def calculate_E_or_H(file_path_1, file_path_2, draw=False, preference=None):
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
    img1_pts, img2_pts = match_images(file_path_1, file_path_2, draw)

    # Calculate the essential matrix
    E, Essential_Mask = cv2.findEssentialMat(img1_pts,
                                img2_pts, K,
                                method=cv2.RANSAC, prob=0.999, threshold=1)

    H, Homography_Mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC,5.0)

    # computing Rotational (R) matrix and Translational (t) matrix from Essential Matrix (E)
    _, R_E, t_E, _ = cv2.recoverPose(E, img1_pts, img2_pts)

    if R_E is None or t_E is None:
        raise ValueError("Pose recovery failed")
    
    # decompose the homography matrix
    _, R_H, t_H, N_H = cv2.decomposeHomographyMat(H, K)

    # Convert keypoint coordinates to an input filterHomographyDecompByVisibleRefPoints likes
    kp1_pts_float32 = np.array(img1_pts, dtype=np.float32)[:, np.newaxis, :]
    kp2_pts_float32 = np.array(img2_pts, dtype=np.float32)[:, np.newaxis, :]

    # Find the two best Rotation and translation matrices from the four different solutions in the Homography Matrix
    filtered = cv2.filterHomographyDecompByVisibleRefpoints(R_H, N_H, kp1_pts_float32, kp2_pts_float32)
    print("Symmetric Error for Essential Matrix", essential_error(E, img1_pts, img2_pts))
    print("Symmetric Error for Homography Matrix", homography_error(H, img1_pts, img2_pts))


    if preference == "E" or (preference == None and essential_error(E, img1_pts, img2_pts) < homography_error(H, img1_pts, img2_pts)):
        return R_E, t_E
    elif preference == "H" or (preference == None and essential_error(E, img1_pts, img2_pts) > homography_error(H, img1_pts, img2_pts)):
        return filtered
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


import time
import cv2 as cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rotation
from scipy.spatial import distance
import os
import open3d as o3d
import open3d.core as o3c
import copy
import yaml, json

# For bufferless video capture
import queue, threading

"""
VoxelBlockGrid class (adapted from Open3D) for ease of initialization and integration.
You can read more about the VoxelBlockGrid here:
https://www.open3d.org/docs/latest/tutorial/t_reconstruction_system/voxel_block_grid.html
"""
class VoxelBlockGrid:
    def __init__(self, depth_scale=1000.0, depth_max=5.0, trunc_voxel_multiplier=8.0, device=o3d.core.Device("CUDA:0")):
        # Reconstruction Information
        self.depth_scale = depth_scale
        self.depth_max = depth_max
        self.trunc_voxel_multiplier = trunc_voxel_multiplier
        self.device = device
        self.camera = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault) # Kinect Intrinsics (default)
        self.depth_intrinsic = o3d.core.Tensor(self.camera.intrinsic_matrix, o3d.core.Dtype.Float64)

        # Initialize the VoxelBlockGrid
        self.vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=('tsdf', 'weight', 'color'),
            attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
            attr_channels=(1, 1, 3),
            voxel_size=3.0 / 64, # this sets the resolution of the voxel grid
            block_resolution=1,
            block_count=50000,
            device=device)

    def integration_step(self, color, depth_numpy, cam_pose):
        # Integration Step (TSDF Fusion)
        depth_numpy = depth_numpy.astype(np.uint16)  # Convert to uint16
        depth = o3d.t.geometry.Image(depth_numpy).to(self.device)
        extrinsic = o3d.core.Tensor(np.linalg.inv(cam_pose), o3d.core.Dtype.Float64)
        frustum_block_coords = self.vbg.compute_unique_block_coordinates(
            depth, self.depth_intrinsic, extrinsic, self.depth_scale, self.depth_max, self.trunc_voxel_multiplier)
        color = o3d.t.geometry.Image(np.asarray(color)).to(self.device)
        color_intrinsic = o3d.core.Tensor(self.camera.intrinsic_matrix, o3d.core.Dtype.Float64)
        self.vbg.integrate(frustum_block_coords, depth, color, self.depth_intrinsic,
                       color_intrinsic, extrinsic, self.depth_scale, self.depth_max, self.trunc_voxel_multiplier)


"""
Bufferless VideoCapture, courtesy of Ulrich Stern (https://stackoverflow.com/a/54577746)
Otherwise, a lag builds up in the video stream.
"""
class VideoCapture:

  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get()

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
Load the poses (after navigation, for analysis) from the posedir.
Returns a list of pose arrays.
"""
def poses_from_posedir(posedir):
    poses = []
    pose_files = [name for name in os.listdir(posedir) if os.path.isfile(os.path.join(posedir, name)) and name.endswith(".txt")]
    pose_files = sorted(pose_files)

    for pose_file in pose_files:
        cam_pose = np.loadtxt(posedir +"/"+pose_file)
        poses.append(cam_pose)
    return poses

"""
Convert a list of poses (after navigation, for analysis) into a trajectory lineset.
This object is used to visualize the trajectory in Open3D.
Returns a list of of lineset objects representing the camera's pose.
"""
def get_poses_lineset(poses):
    points = []
    lines = []
    for pose in poses:
        position = pose[0:3,3] # meters
        points.append(position)
        lines.append([len(points)-1, len(points)])

    pose_lineset = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines[:-1]),
    )
    pose_lineset.paint_uniform_color([1, 0, 0]) #optional: change the color here
    return pose_lineset

"""
Load the trajectory primitives (before navigation).
Read a list of motion primitives (trajectories) from a the "trajlib_dir" (trajectory library) directory.
Returns a list of trajectory objects.
"""
def get_trajlist(trajlib_dir):
    # Get the list of files in the directory
    file_list = os.listdir(trajlib_dir)
    # Filter only .npz files
    npz_files = [file for file in file_list if file.endswith('.npz')]
    # Sort the list of .npz files - important for indexing!
    sorted_files = sorted(npz_files)
    # Iterate over the sorted list of .npz files
    traj_list = []
    for trajfile in sorted_files:
        file_path = os.path.join(trajlib_dir, trajfile)
        traj_list.append(np.load(file_path))
    
    return traj_list

"""
Convert the trajectory list into a list of trajectory linesets.
These are used for visualizing the possible trajectories at each step.
Returns a list of trajectory lineset objects.
"""
def get_traj_linesets(traj_list):
    traj_linesets = []
    amplitudes = []
    for traj in traj_list:
        # traj_dict = {key: traj[key] for key in traj.files}
        z_tsdf = traj['x_sample']
        x_tsdf = -traj['y_sample']
        points = []
        lines = []
        for i in range(len(x_tsdf)):
            points.append([x_tsdf[i], 0, z_tsdf[i]])
            lines.append([len(points)-1, len(points)])
        traj_lineset = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines[:-1]),
        )
        traj_linesets.append(traj_lineset)
        amplitudes.append(traj['amplitude'])
        # get traj info
        period = traj['period']
        forward_speed = traj['forward_speed']

    return traj_linesets, period, forward_speed, amplitudes


"""
MonoNav Planner: Return the chosen trajectory index given the current position, current reconstruction, trajectory library, and goal position.
"""
def choose_primitive(vbg, camera_position, traj_linesets, goal_position, dist_threshold, filterYvals, filterWeights, filterTSDF, weight_threshold):

    # Boolean for stopping criteria
    shouldStop = False

    # Get weights and tsdf values from the voxel block grid
    weights = vbg.attribute("weight").reshape((-1))
    tsdf = vbg.attribute("tsdf").reshape((-1))
    # Get the voxel_coords, voxel_indices
    voxel_coords, voxel_indices = vbg.voxel_coordinates_and_flattened_indices()

    # IMPORTANT
    # Use voxel_indices to rearrange weights and tsdf to match voxel_coords
    # Otherwise, the ordering of voxels from the hashmap is non-deterministic
    weights = weights[voxel_indices]
    tsdf = tsdf[voxel_indices]

    # Generate mask to filter out y values (vertical) (+y is DOWN)
    # This is useful to filter out the floor, and avoid obstacles in-plane
    if filterYvals:
        mask = voxel_coords[:, 1] < -0.3
        # Apply mask to voxel_coords and weights
        voxel_coords = voxel_coords[mask]
        weights = weights[mask]
        tsdf = tsdf[mask]

    # Generate mask to filter by weights
    # This rejects voxels below a certain weight threshold
    if filterWeights:
        mask = weights > weight_threshold
        # Apply mask to voxel_coords and weights
        voxel_coords = voxel_coords[mask,:]
        tsdf = tsdf[mask]

    # Generate mask to filter by tsdf value
    if filterTSDF:
        # Generate mask to filter by tsdf values
        mask = tsdf < 0.0
        voxel_coords = voxel_coords[mask,:]

    # transfer to cpu for cdist
    voxel_coords_numpy = voxel_coords.cpu().numpy()

    # NOW WE HAVE A FILTERED SET OF VOXELS THAT REPRESENT OBSTACLES
    # NEXT, WE DETERMINE THE BEST TRAJECTORY ACCORDING TO A COST FUNCTION

    # Initialize scoring variables to evaluate the trajectories
    max_traj_score = -np.inf # track best trajectory
    min_goal_score = np.inf # track proximity to goal
    max_traj_idx = None # track the index of the best trajectory

    # iterate over the sorted traj linesets
    for traj_idx, traj_linset in enumerate(traj_linesets):
        traj_lineset_copy = copy.deepcopy(traj_linset)
        traj_lineset_copy.transform(camera_position) # transform the lineset (copy) to the camera position
        pts = np.asarray(traj_lineset_copy.points) # meters # extract the points from the lineset
        tmp = distance.cdist(voxel_coords_numpy, pts, "sqeuclidean") # compute the distance between all voxels and all points in the trajectory
        voxel_idx, pt_idx = np.unravel_index(np.argmin(tmp), tmp.shape) # extract indices of the nearest voxel to and nearest point in the trajectory
        nearest_voxel_dist = np.sqrt(tmp[voxel_idx, pt_idx])
        if nearest_voxel_dist > dist_threshold:
            # the trajectory meets the dist_threshold criterion
            if goal_position is not None:
                # the trajectory satisfies the dist_threshold; let's compute the goal score
                tmp_to_goal = distance.cdist(goal_position, pts, "sqeuclidean")
                dst_to_goal = np.sqrt(np.min(tmp_to_goal))
                if dst_to_goal < min_goal_score:
                    # we have a trajectory that gets us closer to the goal
                    # print("traj %d gets us closer to the goal: %f"%(traj_idx, dst_to_goal))
                    max_traj_idx = traj_idx
                    min_goal_score = dst_to_goal
            else:
                # no goal position, choose the index that maximizes distance from the obstacles
                if max_traj_score < nearest_voxel_dist:
                    # we have found a trajectory that gets us closer to goal
                    max_traj_idx = traj_idx
                    max_traj_score = nearest_voxel_dist

    if max_traj_idx is None:
        # No trajectory meets the dist_threshold criterion, robot should stop.
        shouldStop = True
    return shouldStop, max_traj_idx

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

"""
Helper function to extract the image frame number from the filename string.
"""
def split_filename(filename):
    return int(filename.split("-")[-1].split(".")[0])