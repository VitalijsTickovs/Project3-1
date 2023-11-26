# Packages:
import torch
from torch import nn

# Files:
from model import ED_Network
from dataset import getdata

## MAIN
print("here")
print()
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

#X = torch.rand(1, 4, 15, device=device) # tensor with random numbers
X, Y = getdata() # get data based on json files in Data folder

Xt = torch.from_numpy(X)
Yt = torch.from_numpy(Y)

rawOut = model(Xt)
print(rawOut)