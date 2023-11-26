# Packages:
import torch
from torch import nn

# Files:
from dataset import getdata
from model import ED_Network
from model import test_loop
from model import train_loop
from model import optimizationLoop
from drawSkeleton import skeletonPlot

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

def exFrwrdPrpData():
    # example forward propagation of single instance
    X, Y = getdata() # get data based on json files in Data folder
    Xt = torch.from_numpy(X[0]) # convert to tensor 
                                # only works on single instance hence X[0]
    Yt = torch.from_numpy(Y)

    rawOut = model(Xt)
    print(rawOut)

def exTestLoop():
    # example forward propagation of single instance
    X, Y = getdata() # get data based on json files in Data folder
    Xt = torch.from_numpy(X) # convert to tensor 
                                # only works on single instance hence X[0]
    Yt = torch.from_numpy(Y)

    test_loop(Xt, Yt, model, nn.L1Loss())

def exTrainLoop():
    # example forward propagation of single instance
    X, Y = getdata() # get data based on json files in Data folder
    Xt = torch.from_numpy(X) # convert to tensor 
                                # only works on single instance hence X[0]
    Yt = torch.from_numpy(Y)

    lr = 0.05
    optimizer = torch.optim.SGD(model.parameters(), lr)
    train_loop(Xt, Yt, model, nn.L1Loss(), optimizer)


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

    # example forward propagation of single instance
    X, Y = getdata() # get data based on json files in Data folder
    Xt = torch.from_numpy(X) # convert to tensor 
                                # only works on single instance hence X[0]
    Yt = torch.from_numpy(Y)

    lr = 0.05 # (lr, epochs) => (0.005; 60), (0.05, 10); 
    optimizer = torch.optim.SGD(model.parameters(), lr)
    loss_fn = nn.L1Loss()
    epochs = 10
    optimizationLoop(Xt, Yt, model, loss_fn, optimizer, epochs)
    
    # Do a test on other data
    testNames = ["skFloorLeft.json", "skFloorRight.json", 
                 "skNoneLeft1.json", "skNoneLeft2.json", "skNoneLeft3.json",
                 "skNoneRight1.json", "skNoneRight2.json", "skNoneRight3.json"]
    testX, testY = getdata(testNames)
    testXt = torch.from_numpy(testX)
    testYt = torch.from_numpy(testY)
    test_loop(testXt, testYt, model, nn.L1Loss())

    # Let's try to visualise one skeleton
    with torch.no_grad(): # don't use graident otherwise can't call numpy
        rawOut = model(testXt[0])
    arrOut = rawOut.numpy()
    arrOut = arrOut.reshape(15, 4, 3) # reshape single output into correct form

    skeletonPlot(arrOut, 3.0)
