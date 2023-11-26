# Packages:
import torch
from torch import nn

# Files:
from model import ED_Network

## MAIN
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

X = torch.rand(1, 4, 15, device=device) # tensor with random numbers
rawOut = model(X)
print(rawOut)