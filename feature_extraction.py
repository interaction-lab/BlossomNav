# prompt: python code for detecting key points and extracting descriptors from two images and finding the same key points across the two frames
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

fx = 0
fy = 0
cx = 0
cy = 0



# Camera Intrinsics
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])


# Read the two images
img1 = cv.imread('./data/kinect_frame-000000.rgb.jpg')
img2 = cv.imread('./data/kinect_frame-000060.rgb.jpg')

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
confidence_level = 0.45

# Keep only the matches with confidence level above the chosen threshold
good_matches = [match for match in matches if match.distance < confidence_level * max(match.distance for match in matches)]

# Calculate the essential matrix
E, Essential_Mask = cv.findEssentialMat(np.array([kp1[m.queryIdx].pt for m in good_matches]),
                               np.array([kp2[m.trainIdx].pt for m in good_matches]),
                               method=cv.RANSAC, prob=0.90, threshold=3)

H, Homography_Mask = cv.findHomography(np.array([kp1[m.queryIdx].pt for m in good_matches]), np.array([kp2[m.trainIdx].pt for m in good_matches]), cv.RANSAC,5.0)

# computing Rotational (R) matrix and Translational (t) matrix from Essential Matrix (E)
_, R_E, t_E, _ = cv.recoverPose(E, np.array([kp1[m.queryIdx].pt for m in good_matches]), np.array([kp2[m.trainIdx].pt for m in good_matches]))

_, R_H, t_H, N_H = cv.decomposeHomographyMat(H, K)


# Draw the good matches on the images
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