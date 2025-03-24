import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random as rd
from PIL import Image, ImageEnhance
import numpy as np
import pandas as pd
import cv2
import os
from sklearn.cluster import KMeans # type: ignore

target_files = os.listdir('./targets_small')
SELECTED_FILES = rd.sample(target_files, 4)
print(SELECTED_FILES)

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
    target_files = SELECTED_FILES
        
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

def run_osm(epochs):
    # Collect all descriptors and their corresponding image and keypoint information
    all_descriptors = []
    descriptor_info = []

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}', end='\r')

        # Create a target image
        img, used_points, used_targets, target_files = create_target_image()

        # Increase sharpness
        pil_img = Image.fromarray(img)
        enhancer = ImageEnhance.Sharpness(pil_img)
        pil_img = enhancer.enhance(5.0)  # Increase sharpness by a factor of 2
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Initialize ORB detector
        orb = cv2.ORB_create(nfeatures=10, 
                             scaleFactor=1.2, 
                             nlevels=8, 
                             firstLevel=15, 
                             WTA_K=2, 
                             fastThreshold=5)

        # Detect keypoints and compute descriptors
        keypoints, descriptors = orb.detectAndCompute(img, None)

        # Calculate the average distance from each keypoint to the actual point
        distances = []
        for kp in keypoints:
            for point in used_points:
                distance = np.linalg.norm(np.array(kp.pt) - np.array(point))
                distances.append(distance)
        if distances:
            avg_distance = np.mean(distances)
            print(f'Average distance for epoch {epoch + 1}: {avg_distance}')

        # Store descriptors and their corresponding image and keypoint information
        if descriptors is not None:
            all_descriptors.append(descriptors)
            for kp in keypoints:
                descriptor_info.append((epoch, kp.pt, target_files[0]))  

    print('\n')

    # Stack all descriptors into a single array
    return np.vstack(all_descriptors), descriptor_info



def main():
    descriptors, descriptor_info = run_osm(30)


    kmeans = KMeans(n_clusters=4, random_state=0).fit(descriptors)



    # Analyze clusters
    clusters = kmeans.predict(descriptors)
    cluster_info = {i: [] for i in range(kmeans.n_clusters)}
    for idx, cluster in enumerate(clusters):
        cluster_info[cluster].append(descriptor_info[idx])

    # Print cluster information
    for cluster, info in cluster_info.items():
        print(f'Cluster {cluster}:')
        for epoch, kp_pt, target_file in info:
            print(f'\t{target_file}')

if __name__ == '__main__':
    main()