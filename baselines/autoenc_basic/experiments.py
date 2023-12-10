import torch


from main import loadTrainTestSplit
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

    loadTrainTestSplit(model, epochs=5)
    