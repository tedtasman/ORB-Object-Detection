import cv2
import numpy as np
import random as rd
import os
from PIL import Image, ImageEnhance
from sklearn.cluster import KMeans
import detect_zone_generator as dzg

class LionSight:

    def __init__(self, num_targets=4, nfeatures=10, scale_factor=1.2, nlevels=8, edge_threshold=31, first_level=15, wta_k=2, score_type=cv2.ORB_HARRIS_SCORE, patch_size=31, fast_threshold=5):

        self.num_targets = num_targets
        self.orb = cv2.ORB_create(
            nfeatures=nfeatures,
            scaleFactor=1.2,
            nlevels=nlevels,
            firstLevel=first_level,
            WTA_K=wta_k,
            fastThreshold=fast_threshold
        )
        self.all_points = np.array([])


    def prepare_photo(self, photo):
        
        photo = (photo[0] * 255).astype(np.uint8)

        pil_img = Image.fromarray(photo[0])
        enhancer = ImageEnhance.Sharpness(pil_img)
        pil_img = enhancer.enhance(5.0)  # Increase sharpness by a factor of 2
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        return img
    

    def run_osm(self, photo):

        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.orb.detectAndCompute(photo, None)
        
        # Stack all descriptors into a single array
        return keypoints, descriptors
    

    def prepare_keypoints(self, keypoints, descriptors, photo):
        
        # Get the image coordinates
        image_coords = photo[1]

        # Initialize the list of true coordinates
        keypoints_true_coords = np.array([])

        # Iterate over the keypoints
        for i, kp in enumerate(keypoints):
            
            # Get the keypoint coordinates
            kp_coords = kp.pt

            # Append the true coordinates to the list
            true_coords = (kp_coords[0] + image_coords[0], kp_coords[1] + image_coords[1])
            current_row = np.array([descriptors[i], kp.size, kp.response, true_coords])
            keypoints_true_coords = np.vstack((keypoints_true_coords, current_row))
            
        return keypoints_true_coords
    

    def detect_and_locate(self, photo):

        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.run_osm(photo)

        # Locate keypoints
        keypoints_true_coords = self.prepare_keypoints(keypoints, descriptors, photo)

        # Convert keypoints and descriptors to numpy
        descriptors = np.array(descriptors)
        keypoints = np.array(keypoints)
        
        # Stack all descriptors into a single array
        combined = np.hstack((keypoints, descriptors, keypoints_true_coords))

        # Initialize self.all_points if it's empty
        if self.all_points.size == 0:
            self.all_points = combined
        else:
            self.all_points = np.vstack((self.all_points, combined))
         

    def cluster(self):

        # initialize k-means clustering
        kmeans = KMeans(n_clusters=self.num_targets, random_state=0)

        # Fit the k-means model
        kmeans.fit(self.all_points)

        # Analyze clusters
        clusters = kmeans.predict(self.all_points)
        cluster_info = {i: [] for i in range(kmeans.n_clusters)}
        for idx, cluster in enumerate(clusters):
            cluster_info[cluster].append(self.all_points[idx])
        
        return cluster_info
    

def main():
    lion_sight = LionSight()
    runway = dzg.Runway("runway_smaller.png", 860, 350, 6,4)
    photos = runway.generate_photos(20)
    for photo in photos:
        processed_photo = lion_sight.prepare_photo(photo)
        lion_sight.detect_and_locate(processed_photo)
    
    import matplotlib.pyplot as plt

    cluster_info = lion_sight.cluster()

    # Plot each point based on x, y coordinates, colored by cluster
    colors = plt.cm.get_cmap('tab10', lion_sight.num_targets)

    plt.figure(figsize=(10, 8))
    for cluster_id, points in cluster_info.items():
        points = np.array(points)
        x_coords = points[:, -1, 0]  # Extract x-coordinates
        y_coords = points[:, -1, 1]  # Extract y-coordinates
        plt.scatter(x_coords, y_coords, label=f'Cluster {cluster_id}', color=colors(cluster_id))

    plt.title('Clustered Keypoints')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.show()
    print(cluster_info)
    

if __name__ == "__main__":
    main()