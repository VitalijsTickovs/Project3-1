# Packages:
import torch
from torch import nn

# Files:
from model import ED_Network
from dataset import getdata

# METHODS
# Method to return an example tensor of correct shape for forward feed. Note random input values 
# are used. Differentiate between different ways to flatten.
def exFrwrdFd():
    Xt = torch.rand(15, 4, 3, device=device) # tensor with random numbers 
    print(Xt)
    m = nn.Flatten(start_dim=1)
    m2 = nn.Flatten(start_dim=0)
    print(m(Xt))
    print(m2(Xt))
    print(torch.flatten(Xt))
    return Xt


# MAIN
if __name__ == "__main__":
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


    X, Y = getdata() # get data based on json files in Data folder
    Xt = torch.from_numpy(X[0]) # convert to tensor 
                                # only works on single instance hence X[0]
    Yt = torch.from_numpy(Y)

    rawOut = model(Xt)
    print(rawOut)