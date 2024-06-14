# prompt: python code for detecting key points and extracting descriptors from two images and finding the same key points across the two frames
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

fx = 953.3723444283503
fy = 957.9130487518722
cx = 783.8421764606697
cy = 527.694475384533

# Camera Intrinsics
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])


# Read the two images
img1 = cv.imread('./data/pixel_frame-000000.rgb.jpg')
img2 = cv.imread('./data/pixel_frame-000060.rgb.jpg')

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

# Calculate the essential matrix
E, Essential_Mask = cv.findEssentialMat(np.array([kp1[m.queryIdx].pt for m in good_matches]),
                               np.array([kp2[m.trainIdx].pt for m in good_matches]), K,
                               method=cv.RANSAC, prob=0.999, threshold=3)

H, Homography_Mask = cv.findHomography(np.array([kp1[m.queryIdx].pt for m in good_matches]), np.array([kp2[m.trainIdx].pt for m in good_matches]), cv.RANSAC,5.0)

# computing Rotational (R) matrix and Translational (t) matrix from Essential Matrix (E)
_, R_E, t_E, _ = cv.recoverPose(E, np.array([kp1[m.queryIdx].pt for m in good_matches]), np.array([kp2[m.trainIdx].pt for m in good_matches]))

_, R_H, t_H, N_H = cv.decomposeHomographyMat(H, K)

# Convert keypoint coordinates to 32-bit floating point for filterHomographyDecompByVisibleRefPoints
kp1_pts_float32 = np.array([kp.pt for kp in kp1], dtype=np.float32)
kp2_pts_float32 = np.array([kp.pt for kp in kp2], dtype=np.float32)

kp1_pts_float32 = kp1_pts_float32[:, np.newaxis, :]
kp2_pts_float32 = kp2_pts_float32[:, np.newaxis, :]

# Find the two best Rotation and translation matrices from the four different solutions in the Homography Matrix
filtered = cv.filterHomographyDecompByVisibleRefpoints(R_H, N_H, kp1_pts_float32, kp2_pts_float32)

print("---------------------------------------------------")
print("checking to make sure that E is a essential matrix")

# Property 1: Rank of the matrix is 2
rank_E = np.linalg.matrix_rank(E)
print("\nRank of the matrix:", rank_E)

# Property 2: det(E) = 0
det_E = np.linalg.det(E)
print("\nDeterminant of the matrix:", det_E)

# Property 3: E^T * E should be a scalar multiple of the identity matrix
ETE = np.dot(E.T, E)
identity = np.eye(3)
scalar_multiple = ETE / ETE[0, 0]  # Normalize to make the first element 1
print("\nE^T * E:")
print(ETE)
print("\nScalar multiple of the identity matrix:")
print(scalar_multiple)

# Property 4: Singular value decomposition (SVD) of E should have two equal singular values and one zero singular value
U, S, Vt = np.linalg.svd(E)
print("\nSingular values of E:")
print(S)

print("-----------------------------------------------")

print(E)
print(R_E)
print(t_E)

print("---------------------------------------------------")

print(H)
print(R_H)
print(t_H)
print(N_H)

print("---------------------------------------------------")

print(filtered)

print("---------------------------------------------------")

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