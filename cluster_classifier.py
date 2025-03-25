from sklearn.svm import SVC # type: ignore
import lion_sight as ls
import detect_zone_generator as dzg
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import cv2

df = pd.read_csv('cluster_data.csv')

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

model = SVC()

model.fit(X, y)

LS = ls.LionSight(14)
Runway = dzg.Runway("runway_smaller.png", height=800, y_offset=400, ratio=8, num_targets=4)
Runway.assign_targets()
photos = Runway.generate_photos(30)
real_coords = np.array(Runway.points)

for i, photo in enumerate(photos):
    print(f"Processing Photo: {i + 1}")
    LS.detect_and_locate(photo)

cluster_info, cluster_centers = LS.cluster()

good_centers = []

for i, cluster in enumerate(cluster_info.values()):
    
    # Calculate the cluster center
    cluster_center = np.mean(np.array(cluster)[:, -2:], axis=0)

    # Calculate the Euclidean distances of points from the cluster center
    distances_from_center = np.linalg.norm(np.array(cluster)[:, -2:] - cluster_center, axis=1)

    # Use the average distance as the cluster spread
    cluster_spread = np.mean(distances_from_center)
    
    # calculate full dimensional center without the true coordinates
    full_center = np.mean(np.array(cluster)[:, :-2], axis=0)

    # Append the cluster data
    current_row = np.hstack([full_center, cluster_spread])

    # Make a prediction
    prediction = model.predict([current_row])

    # If the prediction is 1, the cluster is good
    if prediction == 1:
        good_centers.append(cluster_center)



# Plot each point based on x, y coordinates, colored by cluster
colors = plt.cm.get_cmap('tab10', LS.num_targets)

# Load the runway image
runway_image = Runway.runway.copy()

plt.figure(figsize=(10, 8))
plt.imshow(runway_image)  # Display the runway image as the background

# plot the cluster centers
good_centers_array = np.array(good_centers)
plt.scatter(good_centers_array[:, -2], good_centers_array[:, -1], label='Cluster Centers', color='red', marker='*')

# Plot the real coordinates
plt.scatter(real_coords[:, 0], real_coords[:, 1], label='Real Coordinates', color='black', marker='x')

plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()
