# File created for the purpose of convertng 3D coordinates into graph 

# IMPORTS
from sklearn.neighbors import NearestNeighbors
import numpy as np


# METHODS
# input:
#   'crdLst' - list shape (N, 3) with coordinates of each object
#   'objLst' - list shape (N,) with types of objects corresponding to 'crdLst'
#   'threshold' - highest dimension of largest object to determine when to consider an object 
#               approximately cose to zero (i.e. when difference is below 'threshold' value)
def convertCtG(crdLst, objLst, threshold = 0.05, neighbrhd = 14):
    all_indcs = np.arange(len(crdLst))
    neighLst = findNN(crdLst, neighbrhd)
    strghtNghbrs = []
    for objs in neighLst:
        trgtCrd = crdLst[objs[0]]
        nnCrd = crdLst[objs[1:]]

        diffCrd = nnCrd - trgtCrd
        indcs = np.where(np.any(np.abs(diffCrd)<threshold, axis=1))
        strghtNghbrs.append(objs[1:][indcs])

    return strghtNghbrs # can't convert to numpy array because shape must be inhomogenous

def findNN(crdLst, neighbrhd, debug=False):
    if (len(crdLst)<neighbrhd): neighbrhd=len(crdLst)
    nbrs = NearestNeighbors(n_neighbors=neighbrhd, algorithm='ball_tree').fit(crdLst)
    
    if (debug):
        print(nbrs.kneighbors(crdLst, return_distance=False))
        print()

    return nbrs.kneighbors(crdLst, return_distance=False)

def crtGrp(dim, strghtNghbrs, objLst):
    graph = np.zeros((dim, dim))
    for i in range(dim):
        graph[i,strghtNghbrs[i]]=1
    print(graph)
    print(objLst)
    return graph


# MAIN
if __name__ == "__main__":
    crdLst = np.array([[1,3,0],[1,5,0],[-1,0,0],[-5,-5,0],[2,1,1]])
    objLst = np.array([0,1,2,0,0])
    strghtNghbrs = convertCtG(crdLst, objLst)
    crtGrp(len(crdLst), strghtNghbrs, objLst)
    print()

# CUSTOM PROCEDURES (examples)
def exmplNearestNeighbours():
    crdLst = np.array([[1,3,0],[1,5,0],[-1,0,0],[-5,-5,0],[2,1,1]])
    objLst = np.array([0,1,2,0,0])
    findNN(crdLst, 10, True)

def checkAnyAxis():
    print(np.any([[False, True], [False, False]], axis=1)) # axis 0 column; axis 1 row
    print()
    print(np.where([True, False, True]))
