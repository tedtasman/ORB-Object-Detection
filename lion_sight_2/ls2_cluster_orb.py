import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans # type: ignore

class ClusterORB:

    def __init__(
        self,
        n_clusters=10,
        n_features=512,
        scale_factor=1.2,
        n_levels=8,
        edge_threshold=31,
        first_level=5,
        wta_k=3,
        score_type=cv2.ORB_HARRIS_SCORE,
        patch_size=31,
        fast_threshold=20):
        """
        Initialize an ORB feature detector with the specified parameters.
        """
        self.orb = cv2.ORB_create(
            n_features=n_features,
            scaleFactor=scale_factor,
            nLevels=n_levels,
            edgeThreshold=edge_threshold,
            firstLevel=first_level,
            WTA_K=wta_k,
            scoreType=score_type,
            patchSize=patch_size,
            fastThreshold=fast_threshold)
        
        self.n_clusters = n_clusters
        self.all_points = np.empty((0, 0)) 
        self.img = None
        self.keypoints = None
        self.descriptors = None

    def prepare_img(self, img):
        
        img = (img[0] * 255).astype('uint8')

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        self.current_img = img
    

    def run_orb(self, img):
        """
        Run the ORB feature detector on the image and return keypoints and descriptors.
        """
        if self.orb is None:
            raise ValueError("ORB detector is not initialized. Please call initialize_orb() first.")
        
        img = self.prepare_img(img)

        self.keypoints, self.descriptors = self.orb.detectAndCompute(img, None)
    

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
    

    def cluster(self):
        
        # Filter the points
        self.filter_points()

        # initialize k-means clustering
        kmeans = MiniBatchKMeans(n_clusters=self.num_targets, random_state=0)

        # Fit the k-means model
        kmeans.fit(self.all_points[:, -2:])

        # Analyze clusters
        clusters = kmeans.predict(self.all_points[:, -2:])
        cluster_info = {i: [] for i in range(kmeans.n_clusters)}
        for idx, cluster in enumerate(clusters):
            cluster_info[cluster].append(self.all_points[idx])
        
        return cluster_info, kmeans.cluster_centers_
    

    def process_image(self, img):
        """
        Process the image and prepare it for clustering.
        """
        img = self.prepare_img(img)

        # Run ORB on the image
        keypoints, descriptors = self.run_orb(img)

        # Prepare keypoints and descriptors for clustering
        self.prepare_keypoints(keypoints, descriptors, img)

    
    def process_images(self, images):
        """
        Process a list of images and then cluster the keypoints.
        """
        for image in images:
            self.process_image(image)
        
        return self.cluster()
        


        