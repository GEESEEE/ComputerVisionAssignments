import cv2
import numpy as np
import os
import itertools
import pickle
# import shutil

# Create a new directory
# os.makedirs('imfolder', exist_ok=True)

# Move images to the new directory
image_names = [f'West_wadden{i}.jpg' for i in range(1, 5)]
# for image_name in image_names:
#     shutil.move(image_name, os.path.join('imfolder', image_name))

# Load the images
image_folder = 'imfolder'
images = [cv2.imread(os.path.join(image_folder, name)) for name in image_names]


# region Manual
# # Define the scaling factor for resizing
# scaling_factor = 0.65  # e.g., 0.5 will reduce the size to 50%
#
# # Define the callback function
# def click_event(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         # Record the original (x, y) coordinates of the point and display it on the resized image
#         points.append((int(x / scaling_factor), int(y / scaling_factor)))
#         cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
#         cv2.imshow('image', img)
#
# # Initialize a dictionary to store the corresponding points for each image pair
# corresponding_points = {}
#
# # Iterate over the image pairs and collect corresponding points
# for n in range(4):
#     for m in range(n + 1, 4):
#         # Resize and display the first image in the pair
#         img = cv2.resize(images[n], None, fx=scaling_factor, fy=scaling_factor)
#         cv2.imshow('image', img)
#         cv2.setMouseCallback('image', click_event)
#
#         # Initialize the list to store points for this image
#         points = []
#         print(f'Select points in image {n + 1} and press any key when done')
#         cv2.waitKey(0)
#
#         # Store the selected points and close the window
#         corresponding_points[(n, m)] = np.array(points)
#         cv2.destroyAllWindows()
#
#         # Resize and display the second image in the pair
#         img = cv2.resize(images[m], None, fx=scaling_factor, fy=scaling_factor)
#         cv2.imshow('image', img)
#         cv2.setMouseCallback('image', click_event)
#
#         # Initialize the list to store points for this image
#         points = []
#         print(f'Select corresponding points in image {m + 1} and press any key when done')
#         cv2.waitKey(0)
#
#         # Store the selected points and close the window
#         corresponding_points[(m, n)] = np.array(points)
#         cv2.destroyAllWindows()
#
# # Save the corresponding points to a file
# with open('psets.pkl', 'wb') as f:
#     pickle.dump(corresponding_points, f)
#endregion

# region Automatic

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Dictionary to store the matching points for each pair
matching_points = {}

# Loop over all unique pairs of images
for i, j in itertools.combinations(range(len(images)), 2):
    # Find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(images[i], None)
    kp2, des2 = sift.detectAndCompute(images[j], None)

    # Use BFMatcher (BRUTEFORCE) to find the best matches
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test to find good matches
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    # Extract location of good matches
    points1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    points2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Store the points
    matching_points[(i, j)] = (points1, points2)

# endregion

# Assuming that you want to warp image 2 to image 1
points1, points2 = matching_points[(0, 1)]

# Estimate the geometric transform (homography)
h_matrix, _ = cv2.findHomography(points2, points1, method=cv2.RANSAC)

# Warp image 2 to the domain of image 1
warped_image = cv2.warpPerspective(images[1], h_matrix, (images[0].shape[1], images[0].shape[0]))

# Write the resulting image to file
cv2.imwrite(os.path.join(image_folder, 'warped_image2.jpg'), warped_image)

# Print the sizes of the images
print("Size of image 1:", images[0].shape)
print("Size of warped image:", warped_image.shape)

# 1. Create Identity Transform for Image 1
identity_matrix = np.eye(3)

# 2. Calculate Spatial Limits
def get_spatial_limits(h_matrix, image_shape):
    # Define the corners of the image
    corners = np.array([[0, 0],
                        [image_shape[1], 0],
                        [image_shape[1], image_shape[0]],
                        [0, image_shape[0]]], dtype=np.float32)

    # Apply the homography matrix to the corners
    transformed_corners = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), h_matrix)

    # Get the spatial limits (min and max for x and y coordinates)
    x_min, y_min = np.min(transformed_corners, axis=0).ravel()
    x_max, y_max = np.max(transformed_corners, axis=0).ravel()

    return x_min, x_max, y_min, y_max


x_min1, x_max1, y_min1, y_max1 = get_spatial_limits(identity_matrix, images[0].shape)
x_min2, x_max2, y_min2, y_max2 = get_spatial_limits(h_matrix, images[1].shape)

# 3. Create common imref
x_min = min(x_min1, x_min2)
y_min = min(y_min1, y_min2)
x_max = max(x_max1, x_max2)
y_max = max(y_max1, y_max2)

width = int(np.ceil(x_max - x_min))
height = int(np.ceil(y_max - y_min))

# 4. Warp both images into this common coordinate system
dst_size = (width, height)
dst_offset = [-x_min, -y_min]
dst_transform = np.array([[1, 0, dst_offset[0]], [0, 1, dst_offset[1]], [0, 0, 1]])

warped_image1 = cv2.warpPerspective(images[0], dst_transform @ identity_matrix, dst_size)
warped_image2 = cv2.warpPerspective(images[1], dst_transform @ h_matrix, dst_size)

# 5. Overlay Images
overlay_alpha = 0.5
overlay = cv2.addWeighted(warped_image1, 1 - overlay_alpha, warped_image2, overlay_alpha, 0)

# Save the overlay image
cv2.imwrite(os.path.join(image_folder, 'overlay.jpg'), overlay)

print("Needed image size:", dst_size)
print("xWorldLimits:", (x_min, x_max))
print("yWorldLimits:", (y_min, y_max))




# Initialize an array to hold the transformation matrices
tform = [np.eye(3) for _ in range(len(images))]

# Compute the cumulative homography matrices
for n in range(1, len(images)):
    # Extract points between image n and image n-1
    points_n_minus_1, points_n = matching_points[(n - 1, n)]

    # Compute homography matrix from image n to image n-1
    h_matrix_n, _ = cv2.findHomography(points_n, points_n_minus_1, method=cv2.RANSAC)

    # Update the transformation matrix for image n in reference to image 1
    tform[n] = tform[n - 1] @ h_matrix_n

# Calculate the common world coordinate system
x_mins, x_maxs, y_mins, y_maxs = [], [], [], []
for n in range(len(images)):
    x_min, x_max, y_min, y_max = get_spatial_limits(tform[n], images[n].shape)
    x_mins.append(x_min)
    x_maxs.append(x_max)
    y_mins.append(y_min)
    y_maxs.append(y_max)

x_min, x_max = min(x_mins), max(x_maxs)
y_min, y_max = min(y_mins), max(y_maxs)
width = int(np.ceil(x_max - x_min))
height = int(np.ceil(y_max - y_min))

dst_size = (width, height)
dst_offset = [-x_min, -y_min]
dst_transform = np.array([[1, 0, dst_offset[0]], [0, 1, dst_offset[1]], [0, 0, 1]])

# Initialize the stitched image canvas
stitched_image = np.zeros((height, width, 3), dtype=np.uint8)

for n in range(len(images)):
    # Warp each image according to the corresponding transformation matrix
    warped_image = cv2.warpPerspective(images[n], dst_transform @ tform[n], dst_size)

    # Create a binary mask of the warped image
    mask = (warped_image > 0).astype(np.uint8)

    # Where the mask is not zero, update the stitched_image with the pixel values of the warped_image
    stitched_image = stitched_image * (1 - mask) + warped_image * mask

# Save the stitched image
cv2.imwrite(os.path.join(image_folder, 'stitched_image.jpg'), stitched_image)

print("Needed image size:", dst_size)
print("xWorldLimits:", (x_min, x_max))
print("yWorldLimits:", (y_min, y_max))

# Initialize a dictionary to store the RMS errors
rms_errors = {}

# Loop through all pairs of images
for (i, j) in itertools.combinations(range(len(images)), 2):
    # If the transformation from i to j is not direct, concatenate the transformations
    h_matrix = np.linalg.inv(tform[i]) @ tform[j] if i < j else np.linalg.inv(tform[j]) @ tform[i]

    # Extract matching points for the image pair (i, j)
    points1, points2 = matching_points[(min(i, j), max(i, j))]

    # If the transformation is from j to i, swap the points
    if i > j:
        points1, points2 = points2, points1

    # Transform points2 to the coordinate system of image 1
    transformed_points2 = cv2.perspectiveTransform(points2, h_matrix)

    # Calculate the error distances
    error_distances = np.sqrt(np.sum((points1 - transformed_points2) ** 2, axis=2)).squeeze()

    # Calculate the RMS error for the pair (i, j)
    rms_errors[(i, j)] = np.sqrt(np.mean(error_distances ** 2))

# Calculate the overall RMS
overall_rms = np.sqrt(np.mean(np.array(list(rms_errors.values())) ** 2))

# Print the pairwise RMS errors and the overall RMS
for pair, rms_error in rms_errors.items():
    print(f"E_{pair[0] + 1},{pair[1] + 1} = {rms_error}")
print(f"Overall RMS = {overall_rms}")
