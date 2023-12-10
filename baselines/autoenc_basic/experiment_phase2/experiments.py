import pickle as pkl
import numpy as np

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

if __name__ == "__main__":
    fileObj = open('baselines/autoenc_basic/experiment_phase2/data.obj', 'rb')
    annotations = pkl.load(fileObj)
    fileObj.close()
    # extract relevant joints from AMASS: [(chest/spine, leftArm, rightArm, head)] = (9, 20, 21, 15)
    if (False):
        print(annotations[0].shape)
        print("1: ", type(annotations))
        print("2: ", type(annotations[0]))
        print("3: ", type(annotations[0][0]), annotations[0][0])
        print("4: ", type(annotations[0][0][0]), annotations[0][0][0])
        print("5: ", type(annotations[0][0][0][0]), annotations[0][0][0][0])
        print()

    fltr_annotations = []
    for i in range(len(annotations)):
        skltSeq = annotations[i]
        extJoints = skltSeq[:,:,[9, 20, 21, 15],:] # extract 4 relevant keypoints
        fltr_annotations.append(extJoints)

    # single frame is 1/60 (60 fps) -> 
    # do an interval break
    intr_annotations = [] # dimensions: (actions, 0.5 intervals, skeletons, keypoints, coordinates)
    for skltSeq in fltr_annotations:
        intervals = intervalBreak(skltSeq)
        intr_annotations.append(intervals)
    