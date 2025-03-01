import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random as rd
from PIL import Image
import numpy as np
import cv2
import os

# Load an image from file
img = mpimg.imread(f'runway{rd.randint(1, 4)}.png')

# Get image dimensions
height, width, _ = img.shape

# Pick a random point
random_point = (rd.randint(0, width - 1), rd.randint(height // 5, 4 * height // 5))

print(f'Random point: {random_point}')

# Pick all files from /targets_small
target_files = os.listdir('/Users/teddytasman/Coding_Projects/PSU_UAS/ObjectDetection/targets_small')

for target_file in target_files:
    # Load the overlay image
    overlay_img = mpimg.imread(f'/Users/teddytasman/Coding_Projects/PSU_UAS/ObjectDetection/targets_small/{target_file}')

    print(f'Overlay image: {target_file}')

    # Get overlay image dimensions
    overlay_height, overlay_width, _ = overlay_img.shape

    # Calculate the position to place the overlay image
    x_start = random_point[0] - overlay_width // 2
    y_start = random_point[1] - overlay_height // 2

    # Ensure the overlay image is within the bounds of the original image
    x_start = max(0, min(x_start, width - overlay_width))
    y_start = max(0, min(y_start, height - overlay_height))

    # Overlay the image
    img_copy = img.copy()
    img_copy[y_start:y_start + overlay_height, x_start:x_start + overlay_width] = overlay_img

    # Convert the image to a format suitable for OpenCV
    img_copy = (img_copy * 255).astype('uint8')

    # preprocess the image
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)

    # Initialize ORB detector
    orb = cv2.ORB_create()  # Adjust number of features as needed

    # Detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(img_copy, None)

    # Get keypoint locations
    keypoint_locations = np.array([kp.pt for kp in keypoints])  # (x, y) coordinates

    # Draw keypoints on the image for visualization
    output_image = cv2.drawKeypoints(img_copy, keypoints, None, color=(0, 255, 0))

    # Display image with keypoints
    plt.imshow(output_image, cmap="gray")
    plt.title(f"ORB Keypoints - {target_file}")
    plt.axis("off")
    plt.show()