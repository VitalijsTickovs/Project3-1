import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle as pkl
import copy
from random import random

# DESCRIPTION: 
## copy of file from visual.ipynb
## objective to create the arrays of data needed to train the network:
##   - X/ features
##   - Y/ outputs

# METHODS:
# Method to breakdown data into equal time intervals (default: 0.5s) 
# Inputs:
#   contents - skeleton measurments
#   tm_wdw - time interval size
#   start 
#   stop
# Output:
#   intervals - contains skeleton measurment per 0.5 second interval
def intervalBreak(contents, start, stop, tm_wdw = 0.5):
    ## Create array
    total_time = (stop-start).total_seconds()
    num_itr = int(total_time//tm_wdw)
    intervals = []
    measurments = []
    prev_i=0

    ## Fill array
    for i in range(contents.shape[1]): # iterate through columns
        column = contents.iloc[:,i]
        key_lst = column["body_list"][0]["keypoint"]
        timepoint = datetime.fromtimestamp(column["timestamp"]/(1000**3)) - timedelta(hours=1, minutes=0) # subtraction to convert to GMT
        intr_i = int((timepoint-start).total_seconds()//tm_wdw)
        if (prev_i==intr_i):
            measurments.append(key_lst)
        else:
            intervals.append(measurments)
            measurments=[]
        prev_i = intr_i
    return intervals

# Method to move all keypoints to a list without breaking skeletons into X second intervals  
# Output:
#   measurments - list of skeletons 
def noBreak(contents):
    measurments = []
    for i in range(contents.shape[1]):
        column = contents.iloc[:,i]
        key_lst = column["body_list"][0]["keypoint"]
        measurments.append(key_lst)
    return measurments

# Method similar to intervalBreak, but this time created for AMASS dataset and not stereolabs dataset 
# (e.g. difference in framerate, AMASS uses 60 fps)
# 
# Inputs:
#   contents - skeleton measurments
# Output:
#   intervals - contains skeleton measurment per 0.5 second interval
def intervalBreakAMASS(contents, tm_wdw = 0.5):
    fps = (1.0/60.0)

    ## Create array
    intervals = []
    measurments = []
    prev_i=0

    ## Fill array
    timepoint = 0
    for i in range(len(contents)): # iterate through columns
        key_lst = contents[i]
        
        intr_i = int((timepoint)//tm_wdw)
        if (prev_i==intr_i):
            measurments.append(key_lst)
        else:
            intervals.append(measurments)
            measurments=[]
        timepoint += fps
        prev_i = intr_i
    return intervals


# Method to select XX (default 15) time points from a given interval of points. Takes care 
# of intervals non divisible by "tm_pts_num" by first selecting based on time step and if not enough 
# then selecting random points not already selected.
def selectTimePoints(slct_intrv, tm_pts_num=15):
    stp = round(len(slct_intrv) / tm_pts_num) # next extract relevant number (15 time points) of coordinates in relevant time intervals

    slct_idxs = list(range(0, len(slct_intrv), stp)) # range() in python 3 is an iterator object
    unslct_idxs = list(range(1, len(slct_intrv), stp))

    # remove if too many
    if len(slct_idxs)>tm_pts_num: slct_idxs = slct_idxs[:-1]

    # add from unselected if no enough
    missing = tm_pts_num-len(slct_idxs)
    missing = missing if missing>0 else 0  
    slct_idxs.extend(np.random.choice(unslct_idxs, missing, replace=False)) # add leftovers determined randomly
    slct_idxs.sort() # sort to ensure correct order when used as indexes


    tmp_arr = np.array(slct_intrv) # array of skeletons in order
    key_points = tmp_arr[slct_idxs,:] # array of skeletons in a correct order # (row, element/column) 
    return key_points

# Method that iterates through 0.5 sec of skeleton measurments. Based on wdw_len which determines how much
# data we are using and how far we want to predict in time (default wdw_len=4: use 1 second of input data 
# to predict 1 second forward) select 15 time points for X and for Y creating a single instance in each. 
# Input:
#   wdw_len:
#       4 slots 0.5 seconds each, first half of slots is input, second half of slots is output
#   tm_pts_num: 
#       number of time points to extract for input and output windows (default 15). Each time point 
#       contains 34 body key points
#   intervals (array):
#       array containing skeleton data chunked into intervals of equal time (default 0.5s)
#   X (array):
#       Needs to contain 4*15*3 data points per line because 4 features * 15 different time points * 3 
#       coordinates per feature. Order: features, x, time-point 1 -> features, y, time-point 1 -> 
#       features, z, time-point 1 -> features, x, time-point 2 -> ... .
#   Y (array):
#       Similar fashion, but this time it is for output and not input.
def appendInstances(X, Y, intervals, wdw_len = 4, tm_pts_num=15):
    for i in range(len(intervals)-3): # how much windows do I get in this interval? -2 from the end, 
                                        # -1 from the beginning. Hence total -3.
        wdw_hlf_end = round((wdw_len/2)+i) # move the sliding window
        wdw_end = wdw_len+i 

        slct_intrv = []
        for in_i in range(i, wdw_hlf_end): # extract relevant time intervals (input window)
            slct_intrv.extend(intervals[in_i])
        in_skltns = selectTimePoints(slct_intrv, tm_pts_num)

        slct_intrv = [] # use same array, but clear it
        for out_i in range(wdw_hlf_end, wdw_end): # (output window)
            slct_intrv.extend(intervals[out_i])
        out_skltns = selectTimePoints(slct_intrv, tm_pts_num)

        X.append(in_skltns) # 15 pts * 34 jnts * 3 crdnts in each row  # easier to append to standard arrays then to numpy
        Y.append(out_skltns) 

# Method similar to appendInstances, but with the ability to define the size of input and output 
#   windows instead of the size of the entire window 
# Input: 
#   wdwSz:
#       array containing length of input window and output window (indices 0 and 1 respectively). 
#       One unit corresponds to one interval (default an 0.5 second interval but can be any depending on
#       definiton in intervalBreak )
def appendInstances2(X, Y, intervals, wdwSz =[4,4], tm_pts_num=15):
    for i in range(len(intervals)-(wdwSz[0]+wdwSz[1]-1)): # how much windows do I get in this interval? -2 from the end, 
                                        # -1 from the beginning. Hence total -3.
        wdw_hlf_end = round((wdwSz[0])+i) # move the sliding window
        wdw_end = wdwSz[0]+wdwSz[1]+i 

        slct_intrv = []
        for in_i in range(i, wdw_hlf_end): # extract relevant time intervals (input window)
            slct_intrv.extend(intervals[in_i])
        in_skltns = selectTimePoints(slct_intrv, tm_pts_num)

        slct_intrv = [] # use same array, but clear it
        for out_i in range(wdw_hlf_end, wdw_end): # (output window)
            slct_intrv.extend(intervals[out_i])
        out_skltns = selectTimePoints(slct_intrv, tm_pts_num)

        X.append(in_skltns) # 15 pts * 34 jnts * 3 crdnts in each row  # easier to append to standard arrays then to numpy
        Y.append(out_skltns) 

# Method to extract relevant keypoints and remove the rest. By default extract 4 keypoints and 
# ignore the rest ([2, 7, 14, 26]) = (chest, leftArm, rightArm, head).
# Bone pairs: (2,7), (2,14), (2,26)
def filterKyPts(X, Y, slct_keys=[2, 7, 14, 26]):
    newX = X[:,:,slct_keys,:]
    newY = Y[:,:,slct_keys,:]

    return newX, newY

# Normalise skeletons using the single coordinate point as tthe origin. 
# Method created for getdataSS(...)
# input: numpy array of skeletons
# output: numpy array of skeletons
def normSkels(skeletons, debug=False):
    norm_skeletons = [] # to be filled and returned as output

    for skel in skeletons:
        keyPtNum = len(skel)
        triple = skel[2]
        norm_skeletons.append(np.subtract(skel, triple))
        if(debug and random()<0.01): # print some, but not all to check whether norm works
            print("_ _ _ _ _")
            print(skel)
            print()
            print(np.subtract(skel, triple))
            print("_________")

    return np.array(norm_skeletons)



# Method which is a copy of the code in MAIN section. Essentially outputs the X and Y created after going 
# through all the training data.
def getdata(namesList = ["skCrateLeft1.json", "skCrateLeft2.json", "skCrateLeft3.json", 
                "skCrateRight1.json", "skCrateRight2.json", "skCrateRight3.json", 
                "skCupLeft1.json", "skCupLeft2.json", "skCupLeft3.json",
                "skCupRight1.json", "skCupRight2.json", "skCupRight3.json", 
                "skFeederLeft1.json", "skFeederLeft2.json", "skFeederLeft3.json",
                "skFeederRight1.json", "skFeederRight2.json", "skFeederRight3.json"], debug = True):
    # X array:
    #   Needs to contain 34*3 data points per line because 34 features * 3 
    #   coordinates per feature. Order: features, x, time-point 1 -> features, y, time-point 1 -> 
    #   features, z, time-point 1 -> features, x, time-point 2 -> ... .
    # Y array:
    #   Similar fashion, but this time it is for output and not input.

    X = []
    Y = []
    # Read file. TODO: multiple names
    
    rlPth = "Data/SkeletonData/ShortIntervals/"
    for name in namesList:
        # 1. Read data
        fullPath = rlPth+name
        contents = pd.read_json(fullPath)

        # 2. Drop the empty body_list columns
        ## iterate and identify the drop names for columns
        dropNames = []
        for columnName, columnData in contents.items():
            if (not (columnData["body_list"])): 
                dropNames.extend([columnName])
        contents = contents.drop(columns=dropNames)

        ## compute new start and stop times as datetime
        startStamp = contents.iloc[:,0]["timestamp"]
        start = datetime.fromtimestamp(startStamp/(1000**3)) - timedelta(hours=1, minutes=0) # subtraction to convert to GMT

        stopStamp = contents.iloc[:,contents.shape[1]-1]["timestamp"]
        stop = datetime.fromtimestamp(stopStamp/(1000**3)) - timedelta(hours=1, minutes=0) # subtraction to convert to GMT

        # 3. Break skeleton measurments into chunks based on equal time periods (0.5s)
        intervals = intervalBreak(contents, start, stop, tm_wdw=0.5)

        # 4. Next take broken down data and select 15 time points adding them to create X and Y arrays
        wdw_len = 4
        tm_pts_num = 15
        #appendInstances(X, Y, intervals, wdw_len, tm_pts_num)
        appendInstances2(X, Y, intervals, [2,4], tm_pts_num)

        # check if dimensions are correct
        if (debug): 
            print(len(intervals)-3)
            print(len(X)) # this should be equal to value on the line above OR add up (if X non empty)
            print(len(X[0])) # this should be 15
            print(len(X[0][0])) # this should be 34
            print(len(X[0][0][0])) # this should be 3
            print()

            print(len(intervals)-3)
            print(len(Y)) # this should be equal to value on the line above OR add up (if Y non empty)
            print(len(Y[0])) # this should be 15
            print(len(Y[0][0])) # this should be 34
            print(len(Y[0][0][0])) # this should be 3
            print()

    # check if same dimensions everywhere + convert to numpy
    Xdata = np.array(X).astype(np.float32) # python has only floats of different length (which are 
                                            # floats and doubles essentially speaking)
    Ydata = np.array(Y).astype(np.float32)

    # Extract relevant key points of the skeleton
    Xdata, Ydata = filterKyPts(Xdata, Ydata)

    return Xdata, Ydata

# Method to get data of single skeletons instead of skeleton sequences
def getdataSS(namesList = ["skCrateLeft1.json", "skCrateLeft2.json", "skCrateLeft3.json", 
                "skCrateRight1.json", "skCrateRight2.json", "skCrateRight3.json", 
                "skCupLeft1.json", "skCupLeft2.json", "skCupLeft3.json",
                "skCupRight1.json", "skCupRight2.json", "skCupRight3.json", 
                "skFeederLeft1.json", "skFeederLeft2.json", "skFeederLeft3.json",
                "skFeederRight1.json", "skFeederRight2.json", "skFeederRight3.json"], debug = True):
    # X array:
    #   Needs to contain 34*3 data points per line because 34 features * 3 
    #   coordinates per feature. Order: features, x, time-point 1 -> features, y, time-point 1 -> 
    #   features, z, time-point 1 -> features, x, time-point 2 -> ... .
    # Y array:
    #   Similar fashion, but this time it is for output and not input.

    X = []
    Y = []
    # Read file. TODO: multiple names
    
    rlPth = "Data/SkeletonData/ShortIntervals/"
    for name in namesList:
        # 1. Read data
        fullPath = rlPth+name
        contents = pd.read_json(fullPath)

        # 2. Drop the empty body_list columns
        ## iterate and identify the drop names for columns
        dropNames = []
        for columnName, columnData in contents.items():
            if (not (columnData["body_list"])): 
                dropNames.extend([columnName])
        contents = contents.drop(columns=dropNames)

        # 3. Move skeleton measurments into a list
        skeletons = noBreak(contents)
        X.extend(skeletons)
        Y.extend(copy.deepcopy(skeletons))


        # check if dimensions are correct
        if (debug): 
            print(len(skeletons))
            print(len(X)) # this should be equal to value on the line above OR add up (if X non empty)
            print(len(X[0])) # this should be 34
            print()

            print(len(skeletons))
            print(len(Y)) # this should be equal to value on the line above OR add up (if Y non empty)
            print(len(Y[0])) # this should be 34
            print()

    # check if same dimensions everywhere + convert to numpy
    Xdata = np.array(X).astype(np.float32) # python has only floats of different length (which are 
                                            # floats and doubles essentially speaking)
    Ydata = np.array(Y).astype(np.float32)

    # normalise skeletons
    Xdata = normSkels(Xdata)
    Ydata = normSkels(Ydata)

    return Xdata, Ydata

# method similar to getData but created specifically for AMASS dataset hence has changes and assumptions 
# made for AMASS. For example: 
# - 60 fps
# - pickle file with skeleton data in a particular format must be available  
def getdataAMASS(pklPath='baselines/autoenc_basic/experiment_phase2/data.obj', debug=False, wdw=[1,2]):
    # X array:
    #   Needs to contain 4*15*3 data points per line because 4 features * 15 different time points * 3 
    #   coordinates per feature. Order: features, x, time-point 1 -> features, y, time-point 1 -> 
    #   features, z, time-point 1 -> features, x, time-point 2 -> ... .
    # Y array:
    #   Similar fashion, but this time it is for output and not input.
    X = []
    Y = []

    # 1. get data from the pkl file 
    fileObj = open(pklPath, 'rb')
    annotations = pkl.load(fileObj)
    fileObj.close()

    
    for i in range(len(annotations)):
        # 2. filter 4 relevant keypoints
        skltSeq = annotations[i]
        flt_skltSeq = skltSeq[:,:,[9, 20, 21, 15],:] # extract 4 relevant keypoints

        # 3. single frame is 1/60 (60 fps) -> do an interval break into 0.5 second intervals
        # dimensions: (actions, 0.5 intervals, skeletons, keypoints, coordinates)
        intervals = intervalBreakAMASS(flt_skltSeq)

        # 4. Next take broken down data and select 15 time points adding them to create X and Y arrays
        tm_pts_num = 15
        appendInstances2(X, Y, intervals, wdw, tm_pts_num)
    

        # check if dimensions are correct
    if (debug): 
        print(len(X)) # this should be equal to value on the line above OR add up (if X non empty)
        print(len(X[0])) # this should be 15
        print(len(X[0][0])) # this should be 34
        print(len(X[0][0][0])) # this should be 3
        print(len(X[0][0][0][0])) # this should be 3
        print()

        print(len(Y)) # this should be equal to value on the line above OR add up (if Y non empty)
        print(len(Y[0])) # this should be 15
        print(len(Y[0][0])) # this should be 34
        print(len(Y[0][0][0])) # this should be 3
        print(len(Y[0][0][0][0])) # this should be 3
        print()

    # check if same dimensions everywhere + convert to numpy
    Xdata = np.array(X).astype(np.float32) # python has only floats of different length (which are 
                                            # floats and doubles essentially speaking)
    Ydata = np.array(Y).astype(np.float32)

    # remove unnecessary axis of size 1 (i.e. those containg only a single element)
    #   there is one present from the AMASS dataset
    Xdata = np.squeeze(Xdata)
    Ydata = np.squeeze(Ydata)


    return Xdata, Ydata

# MAIN:
if __name__ == "__main__":
    getdataSS(debug=True)


