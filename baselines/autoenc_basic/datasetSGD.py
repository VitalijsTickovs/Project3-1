import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# DESCRIPTION: 
## Output array of keypoints orderred in time
def transferKeyPts(contents):
    ## Create array
    timePoints = []
    prev_i=0

    ## Fill array
    for i in range(contents.shape[1]): # iterate through columns
        column = contents.iloc[:,i]
        key_lst = column["body_list"][0]["keypoint"]
        timePoints.append(key_lst)
    return timePoints

# Method which is a copy of the code in MAIN section. Essentially outputs the X and Y created after going 
# through all the training data.
def getdata(namesList = ["skCrateLeft1.json", "skCrateLeft2.json", "skCrateLeft3.json", 
                "skCrateRight1.json", "skCrateRight2.json", "skCrateRight3.json", 
                "skCupLeft1.json", "skCupLeft2.json", "skCupLeft3.json",
                "skCupRight1.json", "skCupRight2.json", "skCupRight3.json", 
                "skFeederLeft1.json", "skFeederLeft2.json", "skFeederLeft3.json",
                "skFeederRight1.json", "skFeederRight2.json", "skFeederRight3.json"]):
    
    rlPth = "Data/SkeletonData/"
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

        # 3. Break skeleton measurments into chunks based on equal time periods (0.5s)
        keyPtLst = transferKeyPts(contents)

        # 4. Next take broken down data and select 15 time points adding them to create X and Y arrays

        # check if dimensions are correct
        if (False): 
            print(contents.shape[1]) # number of time point
            column = contents.iloc[:,0] # ignore this is  a list of lists 
            #                           (i.e. including velocoties and dimensions)
            key_lst = column["body_list"][0]["keypoint"]
            print(len(key_lst)) # this should be 34
            print(len(key_lst[0])) # this should be 3

            print(len(keyPtLst)) # equal to column length (i.e. num of time points)
            print(len(keyPtLst[0])) # this should be 34
            print(len(keyPtLst[0][0])) # this should be 3
            print(contents.iloc[:,0]["body_list"][0]["keypoint"][0:4])
            print("_ _ _")
            print(keyPtLst[0][0:4])
            print()


    return keyPtLst



# MAIN:
if __name__ == "__main__":
    getdata()



