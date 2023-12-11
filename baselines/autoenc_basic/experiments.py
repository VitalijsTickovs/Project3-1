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
    loadTrainTestSplit(model, epochs=10, savePath='baselines/autoenc_basic/Weights/model_weights_AMASS1o2_i1o4.pth') 

    # 2. load weights and do test:
    # loadTest(model, wgtPth='baselines/autoenc_basic/Weights/model_weights_AMASS1o2.pth', isAMASS=True, draw=True)

    # 3. load weights and continue training:
    # preloadTrainTest(model, epochs = 10, isAMASS = True, wghtPth = 'baselines/autoenc_basic/Weights/model_weights_AMASS1o2.pth')

# Training data
# i1o2: 0.5 second in 1 second out 
# - 20 epochs
# i1o4: 0.5 second in 2 second out 
# - 10 epochs (10 epochs/ 6.5 minutes; hence 100 ep = 1 hour)
# i1o6: 0.5 second in 3 second out 
# - ??? epochs