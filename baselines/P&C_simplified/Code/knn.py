# File for k nearest neighbour 

# Imports:
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from model import ED_Network
from dataset import getdataSS
from main import modelSetup, innerLayer

# METHODS:

# description: 
#   - create a knn model and return it
def knn(X, Y):
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(X, Y)
    return neigh

# description:
#   - return array of labels for every point of data generated using knn
def predictLbl(X, Y, instances):
    mdl = knn(X,Y)
    return mdl.predict(instances)

# flatten from (N, 34, 3) to (N, 112)
def flatten(data_X):
    data_Xf = []
    for row in data_X:
        data_Xf.append(row.flatten())

    data_X = np.array(data_Xf)
    return data_X

# sklearn:
#   cluster-distance matrix for different datapoints to all the clusters 
#   can use it to calculate the mean distance for each centroid

# Main (for testing):
if __name__ == "__main__":
    # features
    data0, _ = getdataSS(["bodies_1.json", "bodies_2.json"], "Data/SkeletonData/2Poses/")
    
    data1, _ = getdataSS(["bodies_4.json", "bodies_5.json"], "Data/SkeletonData/2Poses/")
    data_X = np.concatenate((data0, data1), axis=0)

    # reshape features
    data_X = flatten(data_X)

    # model for latent space
    model = modelSetup()

    # latent space for training
    data_latent = []
    for row in data_X:
        data_latent.append(innerLayer(model, row))
    data_X = np.array(data_latent)


    # labels
    data0_Y = np.zeros(len(data0))
    data1_Y = np.ones(len(data1))
    data_Y = np.concatenate((data0_Y, data1_Y), axis=0)

    # test features
    data_test_0, _ = getdataSS(["bodies_3.json"], "Data/SkeletonData/2Poses/")
    data_test_1, _ = getdataSS(["bodies_6.json"], "Data/SkeletonData/2Poses/")
    real_0 = np.zeros(len(data_test_0))
    real_1 = np.ones(len(data_test_1))
    real = np.concatenate((real_0, real_1), axis=0)
    data_test = np.concatenate((data_test_0, data_test_1), axis=0)
    data_test = flatten(data_test)

    # latent space for testing
    data_latent = []
    for row in data_test:
        data_latent.append(innerLayer(model, row))
    data_test = np.array(data_latent)

    # knn
    predicted = predictLbl(data_X, data_Y, data_test)
    print(predicted==real)
    print((predicted==real).sum()/len(predicted))



# Results so far:
#   - silhouette - gives higher scores for larger cluster number
