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
    
    rlPth = "BaselineModel/Skeleton_Dataset/Skeleton_DKE_Data/" # make sure to include '/' at the end 
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

        # 3. Break skeleton measurements into chunks based on equal time periods (0.5s)
        keyPtLst = transferKeyPts(contents)

    return keyPtLst

# MAIN:
if __name__ == "__main__":
    getdata()



