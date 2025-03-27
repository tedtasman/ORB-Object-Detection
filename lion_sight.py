import cv2
import numpy as np
import random as rd
import os
from PIL import Image, ImageEnhance
from sklearn.cluster import KMeans # type: ignore
from sklearn.neighbors import NearestNeighbors # type: ignore
import detect_zone_generator as dzg
import matplotlib.pyplot as plt
import multiprocessing
import time


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
        self.all_points = self.all_points[distances < 500]

    def cluster(self):
        
        # Filter the points
        self.filter_points()

        # initialize k-means clustering
        kmeans = KMeans(n_clusters=self.num_targets, random_state=0)

        # Fit the k-means model
        kmeans.fit(self.all_points[:, -2:])

        # Analyze clusters
        clusters = kmeans.predict(self.all_points[:, -2:])
        cluster_info = {i: [] for i in range(kmeans.n_clusters)}
        for idx, cluster in enumerate(clusters):
            cluster_info[cluster].append(self.all_points[idx])
        
        return cluster_info, kmeans.cluster_centers_
    

    def choose_clusters(self, cluster_info, cluster_centers, num_targets=4, offset=6, width=5, scale=0.5, proximity=100, runway=None):
        
        cluster_scores = []

        for cluster in cluster_info.values():
            
            # Extract x and y coordinates of the cluster points
            cluster_points = np.array(cluster)
            x_coords = cluster_points[:, -2]
            y_coords = cluster_points[:, -1]

            # Plot the cluster points on the runway
            # Load the runway image
            runway_image = runway.runway.copy()
            runway_image = cv2.cvtColor(runway_image, cv2.COLOR_BGR2RGB)

            # Display the runway image as the background
            plt.imshow(runway_image)

            # Plot the cluster points
            plt.scatter(x_coords, y_coords, label=f'Cluster {len(cluster_scores)}')
            plt.show()

            # Calculate the cluster center
            cluster_center = np.mean(np.array(cluster)[:, :], axis=0)

            # Calculate the Euclidean distances of points from the cluster center
            distances_from_center = np.linalg.norm(np.array(cluster)[:, :] - cluster_center, axis=1)

            # Use the average distance as the cluster spread
            cluster_spread = np.mean(distances_from_center)

            # Calculate the distance to the nearest cluster center
            distances = np.linalg.norm(cluster_centers - np.mean(np.array(cluster)[:, -2:], axis=0), axis=1)

            # Calculate the second smallest distance
            next_nearest = np.partition(distances, 1)[1]

            # Calculate the score
            # variance should be low, but not too low
            # distance to the next nearest cluster center should be reasonably high

            variance_score = max(0, scale * (cluster_spread - offset) ** 2 - width)

            proximity_score = 1 if next_nearest > proximity else float('inf')

            cluster_scores.append(variance_score * proximity_score)

        # Choose the best clusters
        best_clusters = np.argsort(cluster_scores)[:num_targets]

        # return best cluster centers
        return cluster_centers[best_clusters]


    def categorize_clusters(self, cluster_info, real_coords, accuracy):
        
        cluster_data = np.array([])

        for i, cluster in enumerate(cluster_info.values()):
            
            # Calculate the cluster center
            cluster_center = np.mean(np.array(cluster)[:, -2:], axis=0)

            # Calculate the Euclidean distances of points from the cluster center
            distances_from_center = np.linalg.norm(np.array(cluster)[:, -2:] - cluster_center, axis=1)

            # Use the average distance as the cluster spread
            cluster_spread = np.mean(distances_from_center)

            # Calculate the distance from the cluster center to the real coordinates
            distances = np.linalg.norm(real_coords - cluster_center, axis=1)

            # Calculate the closest real coordinate
            closest_real_coord = np.argmin(distances)

            # Calculate the distance from the cluster center to the closest real coordinate
            distance_to_real_coord = distances[closest_real_coord]

            # determine if the cluster is correct
            correct = 1 if distance_to_real_coord < accuracy else -1
            
            # calculate full dimensional center without the true coordinates
            full_center = np.mean(np.array(cluster)[:, :-2], axis=0)

            # Append the cluster data
            current_row = np.hstack([full_center, cluster_spread, correct])

            # Initialize cluster_data if it's empty
            if cluster_data.size == 0:
                cluster_data = current_row
            else:
                cluster_data = np.vstack((cluster_data, current_row))
        
        return cluster_data



    

def main():

    lion_sight = LionSight(num_targets=14, wta_k=3, nfeatures=50)

    epochs = 1
    num_photos = 30

    for i in range(epochs):

        print(f"\nEpoch: {i + 1}")

        Runway = dzg.Runway("runway_smaller.png", height=800, y_offset=400, ratio=8, num_targets=4)
        Runway.assign_targets()
        photos = Runway.generate_photos(num_photos)
        real_coords = np.array(Runway.points)

        for i, photo in enumerate(photos):
            #print(f".", end='', flush=True)
            #start_time = time.time()
            lion_sight.detect_and_locate(photo,)
            #print(f"Time taken: {time.time() - start_time}")

        cluster_info, cluster_centers = lion_sight.cluster()

        cluster_data = lion_sight.categorize_clusters(cluster_info, real_coords, 100)

        #with open('cluster_data_2.csv', 'a') as f:
            #np.savetxt(f, cluster_data, delimiter=',')
        
    

    
    for i in range(4, 20):
        lion_sight.num_targets = i

        cluster_info, cluster_centers = lion_sight.cluster()


        #lion_sight.categorize_clusters(cluster_info, cluster_centers, real_coords, 100, Runway)


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
        plt.scatter(cluster_centers[:, -2], cluster_centers[:, -1], label='Cluster Centers', color='red', marker='*')

        # Plot the real coordinates
        plt.scatter(real_coords[:, 0], real_coords[:, 1], label='Real Coordinates', color='black', marker='x')

        plt.title(f'K = {lion_sight.num_targets}')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.show()
    

def run_concurrent_instances(instance_id):
    print(f"Starting instance {instance_id}")
    main()  # Call the main function from your script
    print(f"Instance {instance_id} finished")

if __name__ == "__main__":
    main()
    '''num_instances = 4  # Number of concurrent instances to run

    # Create a pool of processes
    processes = []
    for i in range(num_instances):
        process = multiprocessing.Process(target=run_concurrent_instances, args=(i,))
        processes.append(process)
        process.start()

    # Wait for all processes to complete
    for process in processes:
        process.join()

    print("All instances finished.")'''