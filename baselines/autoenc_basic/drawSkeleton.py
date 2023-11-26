import numpy as np
import matplotlib.pyplot as plt


# MEthod for plotting two points and a line between them
def ex3DPlot(): # from https://www.tutorialspoint.com/connecting-two-points-on-a-3d-scatter-plot-in-python-and-matplotlib
    # Plot points using scatter
    plt.rcParams["figure.figsize"] = [7.50, 6.00] # size of window on mac
    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    x, y, z = [1, 1.5], [1, 2.4], [3.4, 1.4]
    ax.scatter(x, y, z, c='red', s=100)
    ax.plot(x, y, z, color='black') # used to create a line between two points
    plt.show()

def skeletonPlot(skltnSq):
    # extract coordinates
    keyPts = skltnSq[0]
    x = keyPts[:,0] # 2,7,14,26 => (2,7),(2,14),(2,26)
    y = keyPts[:,1]
    z = keyPts[:,2]

    print(x)
    print(y)
    print(z)

    # create bones
    boneTrpls = np.array([keyPts[[0,1],:], keyPts[[0,2],:], keyPts[[0,3],:]])

    # make plot
    plt.rcParams["figure.figsize"] = [7.50, 6.00] # size of window on mac
    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(x, y, z, c='red', s=100)
    for bone in boneTrpls:
        ax.plot(bone[:,0], bone[:,1], bone[:,2], color='red') # used to create a line between two points
    plt.show()
    pass

if __name__ == "__main__":
    pass

    # Then plot bones using ???

