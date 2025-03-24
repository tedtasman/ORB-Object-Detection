import cv2
import numpy as np
import random as rd
import os
from PIL import Image, ImageEnhance
from sklearn.cluster import KMeans # type: ignore
from sklearn.neighbors import NearestNeighbors
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
        
        photo = (photo[0] * 255).astype('uint8')

        pil_img = Image.fromarray(photo)
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

        # Iterate over the keypoints
        for i, kp in enumerate(keypoints):
            
            # Get the keypoint coordinates
            kp_coords = kp.pt

            # Append the true coordinates to the list
            true_coords = (kp_coords[0] + image_coords[0], kp_coords[1] + image_coords[1])

            current_row = np.hstack([descriptors[i].tolist(), kp.size, kp.response, true_coords])

            # Initialize keypoints_true_coords if it's empty
            if self.all_points.size == 0:
                self.all_points = current_row
            else:
                self.all_points = np.vstack((self.all_points, current_row))

    

    def detect_and_locate(self, photo):
        
        # Prepare the photo
        processed_photo = self.prepare_photo(photo)

        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.run_osm(processed_photo)

        # Locate keypoints
        self.prepare_keypoints(keypoints, descriptors, photo)
         

    def filter_points(self):
        
        # Initialize the NearestNeighbors model
        nn = NearestNeighbors(n_neighbors=3, algorithm='ball_tree')

        # Fit the model
        nn.fit(self.all_points)

        # Find the distances and indices of the nearest neighbors
        distances, indices = nn.kneighbors(self.all_points)

        # Get the distances to the nearest neighbor
        distances = distances[:, 1]

        # Filter the points based on the distance to the nearest neighbor
        self.all_points = self.all_points[(distances < 200 and distances > 30)]


    def cluster(self):
        
        # Filter the points
        self.filter_points()

        # initialize k-means clustering
        kmeans = KMeans(n_clusters=self.num_targets, random_state=0)

        # Fit the k-means model
        kmeans.fit(self.all_points)

        # Analyze clusters
        clusters = kmeans.predict(self.all_points)
        cluster_info = {i: [] for i in range(kmeans.n_clusters)}
        for idx, cluster in enumerate(clusters):
            cluster_info[cluster].append(self.all_points[idx])
        
        return cluster_info, kmeans.cluster_centers_
    

def main():
    lion_sight = LionSight()
    Runway = dzg.Runway("runway_smaller.png", 860, 350, 6,4)
    Runway.assign_targets()
    photos = Runway.generate_photos(30)
    
    real_coords = np.array(Runway.points)
    for i, photo in enumerate(photos):
        print(f"Processing Photo: {i + 1}")
        lion_sight.detect_and_locate(photo)
    
    import matplotlib.pyplot as plt

    cluster_info, cluster_centers = lion_sight.cluster()

    # Plot each point based on x, y coordinates, colored by cluster
    colors = plt.cm.get_cmap('tab10', lion_sight.num_targets)

    # Load the runway image
    runway_image = Runway.runway.copy()
    runway_image = cv2.cvtColor(runway_image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 8))
    plt.imshow(runway_image)  # Display the runway image as the background

    # Plot the clustered keypoints
    for cluster_id, points in cluster_info.items():
        points = np.array(points)
        x_coords = points[:, -2]  # Extract x-coordinates
        y_coords = points[:, -1]  # Extract y-coordinates
        plt.scatter(x_coords, y_coords, label=f'Cluster {cluster_id}', color=colors(cluster_id))

    # plot the cluster centers
    plt.scatter(cluster_centers[:, -2], cluster_centers[:, -1], label='Cluster Centers', color='red', marker='x')

    # Plot the real coordinates
    plt.scatter(real_coords[:, 0], real_coords[:, 1], label='Real Coordinates', color='black', marker='x')

    plt.title('Clustered Keypoints on Runway')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    main()