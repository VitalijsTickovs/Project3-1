# File for k nearest neighbour 

# Imports:
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# METHODS:

# description: 
#   - create a knn model and return it
# input:
#   - shape of "data" (n_samples, n_features)
# output:
#   - sklearn knn model
def kmeans(data):
    # n_neighbors -> default number of neighbours to return
    # radius -> size of the circle to consider as the neighbouhood
    km = KMeans(n_clusters=2, random_state=0, n_init="auto")
    cluster_labels = km.fit_predict(data)
    silhouette_avg = silhouette_score(data, cluster_labels)
    return silhouette_avg

# description:
#   - return array of labels for every point of data generated using knn
def lblKnn(data):
    return

# sklearn:
#   cluster-distance distance for different datapoints to all the clusters 
#   can use it to calculate the mean distance for each centroid

# Main (for testing):
if __name__ == "__main__":
    
    print(kmeans(data))
