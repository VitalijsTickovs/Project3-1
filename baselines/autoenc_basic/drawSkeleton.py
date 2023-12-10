import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler


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

def singleSkeletonPlot(skltnSq):
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

def crtCycleArr(cycles, rgb, stp, step, ptsNum, boneNum):
    clrList = []
    for i in range(0, cycles, int(step)):
        for rep in range(ptsNum+boneNum):
            clrList.append(rgb.copy())
        rgb[0] = rgb[0] - stp
    return clrList


def skeletonPlot(skltnSq, step=1.0, title="default"):
    # set color tuple
    rgb =[1.0,0.0,0.0]
    clrStp = rgb[0]/((len(skltnSq)/step)) # how much to decrement by the redness

    # create figure
    plt.rcParams["figure.figsize"] = [7.50, 6.00] # size of window on mac
    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_title(title)

    # set color cycle for different plots
    ptsNum = 0
    boneNum = 3 # preset (can't be computed)
    clrCycleLst = crtCycleArr(len(skltnSq), rgb, clrStp, step, ptsNum, boneNum) #TODO: quadiple same color to prevent lines and points of different color
    ax.set_prop_cycle(cycler('color', clrCycleLst))

    # iterate through time points
    for i in range(0,len(skltnSq),int(step)):
        # extract coordinates
        keyPts = skltnSq[i]
        x = keyPts[:,0] # 2,7,14,26 => (2,7),(2,14),(2,26)
        y = keyPts[:,1]
        z = keyPts[:,2]

        # create bones
        boneTrpls = np.array([keyPts[[0,1],:], keyPts[[0,2],:], keyPts[[0,3],:]])

        # make plot
        #ax.scatter(x, y, z, s=100)
        for bone in boneTrpls:                               
            ax.plot(bone[:,0], bone[:,1], bone[:,2])        # used to create a line between two points
                                                            # (!!!) limited to one color for all points
    plt.show()

# Create a single plot for comparing predicted and real skeleton sequences
def cmprtvSkltPlt(skltnSq_P, skltnSq_R, step=1.0, title="default"):
    # set color tuple
    rgb1 =[1.0,0.0,0.0]
    rgb2 =[0.0,1.0,0.0]
    clrStp1 = rgb1[0]/((len(skltnSq_P)/step)) # how much to decrement by the redness
    clrStp2 = rgb2[1]/((len(skltnSq_R)/step)) 


    # create figure
    plt.rcParams["figure.figsize"] = [7.50, 6.00] # size of window on mac
    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_title(title)

    # set color cycle for different plots
    ptsNum = 0
    boneNum = 3 # preset (can't be computed)
    clrCycleLst1 = crtCycleArr(len(skltnSq_P), rgb1, clrStp1, step, ptsNum, boneNum)
    clrCycleLst2 = crtCycleArr(len(skltnSq_R), rgb2, clrStp2, step, ptsNum, boneNum)  
    

    # iterate through time points for predicted skeleton sequence
    ax.set_prop_cycle(cycler('color', clrCycleLst1))
    drawSkeletonSeq(ax, skltnSq_P, step)

    # similarly iterate through time points for real skeleton sequence
    ax.set_prop_cycle(cycler('color', clrCycleLst2))
    drawSkeletonSeq(ax, skltnSq_R, step)

    plt.show()
        
# sub-method of cmprtvSkltPlt. Allows to add to plot a single skeleton sequence
def drawSkeletonSeq(ax, skltnSq, step):
    # iterate through time points
    for i in range(0,len(skltnSq),int(step)):
        # extract coordinates
        keyPts = skltnSq[i]
        x = keyPts[:,0] # 2,7,14,26 => (2,7),(2,14),(2,26)
        y = keyPts[:,1]
        z = keyPts[:,2]

        # create bones
        boneTrpls = np.array([keyPts[[0,1],:], keyPts[[0,2],:], keyPts[[0,3],:]])

        # make plot
        for bone in boneTrpls:                               
            ax.plot(bone[:,0], bone[:,1], bone[:,2])        # used to create a line between two points
                                                            # (!!!) limited to one color for all points
    

if __name__ == "__main__":
    pass

    # Then plot bones using ???

