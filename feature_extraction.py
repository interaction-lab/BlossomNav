import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import statistics as st
from utils.utils import read_yaml

config = read_yaml()

def camera_intrinsics():
    """
    Obtains the camera intrinsics data from some .yaml file
    
    file: the path to the .yaml file with your data (has been defaulted 
    to the config.yaml file that was created alongside this repository)
    
    returns a matrix configuration of the data 
    """
    fx = config['camera']['fx']
    fy = config['camera']['fy']
    cx = config['camera']['cx']
    cy = config['camera']['cy']

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
    # Read the two images
    img1 = cv.imread(file_path_1)
    img2 = cv.imread(file_path_2)

    # Convert the images to grayscale
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    # Initialize the ORB detector
    orb = cv.ORB_create(nfeatures=1000)

    # Detect key points and extract descriptors
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    image_with_keypoints1 = cv.drawKeypoints(img1, kp1, None, color=(0, 255, 0), flags=0)
    image_with_keypoints2 = cv.drawKeypoints(img2, kp2, None, color=(0, 255, 0), flags=0)

    cv.imshow("img1", image_with_keypoints1)
    cv.imshow("img2", image_with_keypoints2)

    # Match the descriptors
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
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
        image_with_good_matches = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Display the image with good matches
        cv.imshow("matching", image_with_good_matches)

        # Wait for a key press, but allow Ctrl+C to terminate the program
        while True:
            key = cv.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or 'q' to quit
                break

        # Close all windows
        cv.destroyAllWindows()

    return img1_pts, img2_pts

def calculate_E_or_H(file_path_1, file_path_2):
    K = camera_intrinsics()
    img1_pts, img2_pts = match_images(file_path_1, file_path_2, draw=config['utils']['draw'])

    # Calculate the essential matrix
    E, Essential_Mask = cv.findEssentialMat(img1_pts,
                                img2_pts, K,
                                method=cv.RANSAC, prob=0.999, threshold=3)

    H, Homography_Mask = cv.findHomography(img1_pts, img2_pts, cv.RANSAC,5.0)

    # computing Rotational (R) matrix and Translational (t) matrix from Essential Matrix (E)
    _, R_E, t_E, _ = cv.recoverPose(E, img1_pts, img2_pts)

    _, R_H, t_H, N_H = cv.decomposeHomographyMat(H, K)

    # Convert keypoint coordinates to an input filterHomographyDecompByVisibleRefPoints likes
    kp1_pts_float32 = np.array(img1_pts, dtype=np.float32)[:, np.newaxis, :]
    kp2_pts_float32 = np.array(img2_pts, dtype=np.float32)[:, np.newaxis, :]

    # Find the two best Rotation and translation matrices from the four different solutions in the Homography Matrix
    filtered = cv.filterHomographyDecompByVisibleRefpoints(R_H, N_H, kp1_pts_float32, kp2_pts_float32)
    print("Symmetric Error for Essential Matrix", essential_error(E, img1_pts, img2_pts))
    print("Symmetric Error for Homography Matrix", homography_error(H, img1_pts, img2_pts))

    if essential_error(E, img1_pts, img2_pts) < homography_error(H, img1_pts, img2_pts):
        print("Essential Matrix was better")
        print(E)
        print("---------------------------------------------------")
        print(R_E)
        print("---------------------------------------------------")
        print(t_E)
    elif essential_error(E, img1_pts, img2_pts) > homography_error(H, img1_pts, img2_pts):
        print("Homography Matrix was better")
        print(H)
        print("---------------------------------------------------")
        print(R_H)
        print("---------------------------------------------------")
        print(t_H)
    else:
        print("Error: Something went wrong")

if __name__ == "__main__":
    calculate_E_or_H('./data/pixel_frame-000000.rgb.jpg', './data/pixel_frame-000060.rgb.jpg')