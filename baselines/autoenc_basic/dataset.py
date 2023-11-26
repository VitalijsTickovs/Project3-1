import pandas as pd
# copy of file from visual.ipynb
# objective to create the arrays of data needed to train the network:
#   - X/ features
#   - Y/ outputs

### METHODS:



### MAIN:

# Read file. TODO: multiple names
trainNames = ["skCrateLeft1.json", "skCrateLeft2.json", "skCrateLeft3.json", 
              "skCrateRight1.json", "skCrateRight2.json", "skCrateRight3.json", 
              "skCupLeft1.json", "skCupLeft2.json", "skCupLeft3.json",
              "skCupRight1.json", "skCupRight2.json", "skCupRight3.json", 
              "skFeederLeft1.json", "skFeederLeft2.json", "skFeederLeft3.json",
              "skFeederRight1.json", "skFeederRight2.json", "skFeederRight3.json"]
rlPth = "Data/SkeletonData/"
for name in trainNames:
    fullPath = rlPth+name
    contents = pd.read_json(fullPath)
