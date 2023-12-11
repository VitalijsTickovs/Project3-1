import torch
from torch import nn
import numpy as np

from main import loadTrainTestSplit
from main import loadTest
from main import preloadTrainTest
from model import ED_Network


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

# visualise how L1 loss is computed
def visualiseL1Loss():
    loss_fn = nn.L1Loss()
    tens1 = torch.from_numpy(np.array([[1,2], [3,0]]).astype(np.float32))
    tens2 = torch.from_numpy(np.array([[3,3], [3,5]]).astype(np.float32))
    result = loss_fn(tens1, tens2).item()
    print("result: ", result)

if __name__ == "__main__":
    # pre-setting procedure
    full_debug = True

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model = ED_Network().to(device)

    if (full_debug):
        print(f"Using {device} device")
        print()

        print(model)
        print()

    # 1. redo training from scratch:
    # loadTrainTestSplit(model, epochs=10, savePath='baselines/autoenc_basic/Weights/model_weights_AMASS1o2_i1o6.pth') 

    # 2. load weights and do test:
    # loadTest(model, wgtPth='baselines/autoenc_basic/Weights/model_weights_AMASS1o2_i1o4.pth', isAMASS=True, draw=True)

    # 3. load weights and continue training:
    preloadTrainTest(model, epochs = 10, isAMASS = True, wghtPth = 'baselines/autoenc_basic/Weights/model_weights_AMASS1o2_i1o2.pth')

# Training data
# i1o2: 0.5 second in 1 second out (L1 = 0.069842 | 4.74 cm)
# - 30 epochs
# i1o4: 0.5 second in 2 second out (L1 = 0.082967 | 5.64 cm)
# - 10 epochs (10 epochs/ 6.5 minutes; hence 100 ep = 1 hour) 
# i1o6: 0.5 second in 3 second out (L1 = 0.086333 | 5.86 cm)
# - 10 epochs


# AMASS Estimating distance
#   - (center chest point, head point below nose); i1o4
#   - ([0.00103837, -0.02953809, 0.2974239], [0.01126073, -0.05810481, 0.55965173]) -> 0.263977
#   - ([0.00720938, -0.0210958, 0.29842746] [0.01979808, -0.0516804, 0.5630473]) -> 0.266679
#   - ([0.0095277, -0.01204579, 0.3014936] [0.00958329, -0.04328188, 0.5633496]) -> 0.263712
#   - sk5.1 -> 0.293045 
#   - sk5.2 -> 0.289472
#   - sk5.3 -> 0.288952 
#   - sk10.1 -> 0.293662
#   - sk10.2 -> 0.294671
#   - sk10.3 -> 0.297791 
# Mean of last 6 points: 0.2929321667 
# Distance from chin to sub-naseal area: 2.9 cm (see https://en.wikipedia.org/wiki/Human_head)
# Midline neck length: 8 cm (see https://www.researchgate.net/figure/Dimensions-of-Human-Neck-mm-18_tbl2_332686697#:~:text=The%20dimensions%20of%20the%20average,19%5D%2C%20%5B20%5D%20.
#                               and https://www.researchgate.net/publication/273700390_Relationship_between_Neck_Length_Sleep_and_Cardiovascular_Risk_Factors#pf3 )
# Base of neck to middle of chest: 9 cm (approximation)
# Sum length: 19.9 cm
# Mapping/conversion rate: 67.933 per 1 (cm per unit)

