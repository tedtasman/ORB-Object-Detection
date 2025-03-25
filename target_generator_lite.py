import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random as rd
from PIL import Image, ImageEnhance
import numpy as np
import pandas as pd
import cv2
import os
from sklearn.cluster import KMeans # type: ignore
import detect_zone_generator as dzg

def create_target_image():
    # Load an image from file
    img = mpimg.imread(f'runway{rd.randint(1, 4)}.png')

    # Get image dimensions
    height, width, _ = img.shape

    # Create a list to store the points and targets used in the image
    used_points = []
    used_targets = []
    target_files = []

    # Use only one target per image
    img, used_points, used_targets, target_file = overlay_random_target(height, width, img, used_targets, used_points)
    target_files.append(target_file)

    # Convert the image to a format suitable for OpenCV
    img = (img * 255).astype('uint8')

    return img, used_points, used_targets, target_files

def overlay_random_target(height, width, img, used_targets, used_points):
    # Pick all files from /targets_small
    target_files = None
        
    # Pick a random target file ensuring it is not already used
    while True:
        target_file = target_files[rd.randint(0, len(target_files) - 1)]
        if target_file not in used_targets:
            used_targets.append(target_file)
            break

    # Load the overlay image
    overlay_img = mpimg.imread(f'./targets_small/{target_file}')

    # Get overlay image dimensions
    overlay_height, overlay_width, _ = overlay_img.shape

    # Pick a random point ensuring it is at least 500px from any point in used_points
    while True:
        random_point = (rd.randint(0, width - 1), rd.randint(height // 5, 4 * height // 5))
        if all(np.linalg.norm(np.array(random_point) - np.array(point)) >= 500 for point in used_points):
            break

    # Calculate the position to place the overlay image
    x_start = random_point[0] - overlay_width // 2
    y_start = random_point[1] - overlay_height // 2

    # Ensure the overlay image is within the bounds of the original image
    x_start = max(0, min(x_start, width - overlay_width))
    y_start = max(0, min(y_start, height - overlay_height))

    # Overlay the image
    img_copy = img.copy()
    for i in range(overlay_height):
        for j in range(overlay_width):
            if overlay_img[i, j, 3] > 0.5:  # Check if the pixel is not transparent
                img_copy[y_start + i, x_start + j] = overlay_img[i, j, :]  # Copy only RGB channels

    # Add the point to the list of used points
    used_points.append(random_point)

    # Add the overlay image to the list of used targets
    used_targets.append(target_file)

    return img_copy, used_points, used_targets, target_file

def run_osm(img):


    # Preprocess the image
    img = (img * 255).astype('uint8')
    pil_img = Image.fromarray(img)
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=10, 
                            scaleFactor=1.2, 
                            nlevels=8, 
                            firstLevel=15, 
                            WTA_K=2, 
                            fastThreshold=5)

    # Detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(img, None)

    # Draw keypoints on the image
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0), flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

    # Display the image with keypoints
    plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
    plt.title("Image with Keypoints")
    plt.axis("off")
    plt.show()


    print('\n')



def main():

    Runway = dzg.Runway("runway_smaller.png", height=860, y_offset=350, ratio=6, num_targets=4)
    Runway.assign_targets()
    photos = Runway.generate_photos(3)

    for i, photo in enumerate(photos):
        run_osm(photo[0])

if __name__ == '__main__':
    main()