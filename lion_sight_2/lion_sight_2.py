import torch
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans # type: ignore
from ls2_cluster_orb import ClusterORB

class LionSight2:

    def __init__(self, num_targets, net, orb):
        self.num_targets = num_targets
        self.orb = orb
        self.net = net
    
    
    def detect(self, images_path):
        '''
        Detect objects in the images using the neural network and ORB feature detector.
        '''

        # Load the images
        images = [cv2.imread(image_path) for image_path in images_path]

        # Process 
        keypoints, _ = self.orb.process_images(images)

        # TODO: Run the neural network on the keypoints and get the predictions
        



    


    

    
