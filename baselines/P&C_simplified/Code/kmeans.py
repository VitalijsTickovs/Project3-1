# File for k nearest neighbour 

# Imports:
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from dataset import getdataSS

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
    silhouette_avg = silhouette_score(data, cluster_labels) # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html#examples-using-sklearn-metrics-silhouette-score
    return silhouette_avg

# description:
#   - return array of labels for every point of data generated using knn
def lblKnn(data):
    return

# sklearn:
#   cluster-distance matrix for different datapoints to all the clusters 
#   can use it to calculate the mean distance for each centroid

# Main (for testing):
if __name__ == "__main__":
    data, _ = getdataSS(["bodies_1.json", "bodies_2.json", "bodies_3.json", 
                         "bodies_4.json", "bodies_5.json", "bodies_6.json"], "Data/SkeletonData/2Poses/")
    # flatten data
    data_f = []
    for row in data:
        data_f.append(row.flatten())

    data = np.array(data_f)

    print(kmeans(data))


# Results so far:
#   - silhouette - gives higher scores for larger cluster number
