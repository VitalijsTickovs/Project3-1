# Packages:
import torch
from torch import nn
import time

# Files:
from dataset import getdata
from dataset import getdataAMASS
from model import ED_Network
from model import test_loop
from model import train_loop
from model import optimizationLoop
from drawSkeleton import skeletonPlot
from drawSkeleton import cmprtvSkltPlt
from drawSkeleton import cmprtvSeqPlt

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

# method which loads the data and trains the model from scratch. Weights are saved at the end.
def loadTrainTest(model):
    X, Y = getdata() # get data based on json files in Data folder
    Xt = torch.from_numpy(X) # convert to tensor 
                                # only works on single instance hence X[0]
    Yt = torch.from_numpy(Y)

    lr = 0.05 # (lr, epochs) => (0.005; 60), (0.05, 10); 
    optimizer = torch.optim.SGD(model.parameters(), lr)
    loss_fn = nn.L1Loss()
    epochs = 60
    optimizationLoop(Xt, Yt, model, loss_fn, optimizer, epochs)
    
    # Do a test on other data
    testNames = ["skFloorLeft.json", "skFloorRight.json", 
                 "skNoneLeft1.json", "skNoneLeft2.json", "skNoneLeft3.json",
                 "skNoneRight1.json", "skNoneRight2.json", "skNoneRight3.json"]
    testX, testY = getdata(testNames)
    testXt = torch.from_numpy(testX)
    testYt = torch.from_numpy(testY)
    test_loop(testXt, testYt, model, nn.L1Loss())

    # save model weights
    torch.save(model.state_dict(), 'baselines/autoenc_basic/Weights/model_weights.pth')
    
    drawSkltns(testXt, testYt)

def loadTrainTestSplit(model, epochs = 60, isAMASS = True, savePath='baselines/autoenc_basic/Weights/model_weights_AMASS1o2.pth'):
    if (isAMASS):
        X, Y = getdataAMASS(wdw=[1,6]) # get data based on json files in Data folder
    
    halfEnd = int(len(X)//2)
    end = len(X)
    X_train = X[0:(halfEnd-1)]
    Y_train = Y[0:(halfEnd-1)]
    X_test = X[halfEnd:end]
    Y_test = Y[halfEnd:end]

    Xt = torch.from_numpy(X_train) # convert to tensor 
                                # only works on single instance hence X[0]
    Yt = torch.from_numpy(Y_train)

    lr = 0.05 # (lr, epochs) => (0.005; 60), (0.05, 10); 
    optimizer = torch.optim.SGD(model.parameters(), lr)
    loss_fn = nn.L1Loss()
    optimizationLoop(Xt, Yt, model, loss_fn, optimizer, epochs)
    
    # Do a test on other data
    testXt = torch.from_numpy(X_test)
    testYt = torch.from_numpy(Y_test)
    test_loop(testXt, testYt, model, nn.L1Loss())

    # save model weights
    torch.save(model.state_dict(), 'baselines/autoenc_basic/Weights/model_weights_AMASS1o2_i1o6.pth')

def preloadTrainTest(model, epochs = 60, isAMASS = True, wghtPth = 'baselines/autoenc_basic/Weights/model_weights_AMASS1o2.pth'):
    if (isAMASS):
        X, Y = getdataAMASS(wdw=[1,2]) # get data based on json files in Data folder
    
    halfEnd = int(len(X)//2) # divide int train and test (currently by half)
    end = len(X)
    X_train = X[0:(halfEnd-1)]
    Y_train = Y[0:(halfEnd-1)]
    X_test = X[halfEnd:end]
    Y_test = Y[halfEnd:end]

    Xt = torch.from_numpy(X_train) # convert to tensor 
                                # only works on single instance hence X[0]
    Yt = torch.from_numpy(Y_train)

    # load model weights
    model.load_state_dict(torch.load(wghtPth))

    lr = 0.05 # (lr, epochs) => (0.005; 60), (0.05, 10); 
    optimizer = torch.optim.SGD(model.parameters(), lr)
    loss_fn = nn.L1Loss()
    optimizationLoop(Xt, Yt, model, loss_fn, optimizer, epochs)
    
    # Do a test on other data
    testXt = torch.from_numpy(X_test)
    testYt = torch.from_numpy(Y_test)
    test_loop(testXt, testYt, model, nn.L1Loss())

    # save model weights
    torch.save(model.state_dict(), wghtPth)


# method which loads the saved weights and tests the model
def loadTest(model, wgtPth='baselines/autoenc_basic/Weights/model_weights.pth', isAMASS = False, draw=False):
    if (not isAMASS):
        testNames = ["skFloorLeft.json", "skFloorRight.json", 
                    "skNoneLeft1.json", "skNoneLeft2.json", "skNoneLeft3.json",
                    "skNoneRight1.json", "skNoneRight2.json", "skNoneRight3.json"]
        testX, testY = getdata(testNames, False)
    else:
        X, Y = getdataAMASS()
        halfEnd = int(len(X)//2) # divide int train and test (currently by half)
        end = len(X)
        testX = X[halfEnd:end]
        testY = Y[halfEnd:end]

    model.load_state_dict(torch.load(wgtPth)) # load weights model
    testXt = torch.from_numpy(testX)
    testYt = torch.from_numpy(testY)
    test_loop(testXt, testYt, model, nn.L1Loss())

    #if (draw): drawSkltns(testXt, testYt, model)
    #if (draw): drawComprtvSkltns(testXt, testYt, model)
    if (draw): drawCmprtvSeqPlt(testXt, testYt, model)


def drawCmprtvSeqPlt(testXt, testYt, model, sk_id = 10):
    with torch.no_grad(): # don't use graident otherwise can't call numpy
        rawOut = model(testXt[sk_id])
    arrOut = rawOut.numpy()
    arrOut = arrOut.reshape(15, 4, 3)
    cmprtvSeqPlt(arrOut, testYt[sk_id], 5.0)


# draw predicted and real skeleton in a single plot
def drawComprtvSkltns(testXt, testYt, model):
    with torch.no_grad(): # don't use graident otherwise can't call numpy
        rawOut = model(testXt[0])
    arrOut = rawOut.numpy()
    arrOut = arrOut.reshape(15, 4, 3)
    cmprtvSkltPlt(arrOut, testYt[0])

# method for drawing skeleton sequences
def drawSkltns(testXt, testYt, model):
    # Let's try to visualise one skeleton prediction
    with torch.no_grad(): # don't use graident otherwise can't call numpy
        rawOut = model(testXt[0])
    arrOut = rawOut.numpy()
    arrOut = arrOut.reshape(15, 4, 3) # reshape single output into correct form
    skeletonPlot(arrOut, 5.0, "prediction")

    rawOut = testYt[0]
    arrOut = rawOut.numpy()
    arrOut = arrOut.reshape(15, 4, 3) # reshape single output into correct form
    skeletonPlot(arrOut, 5.0, "real")

# method for computing average forward propgation time (last measurment: 0.9092473983764648 ms)
def avgFrwdPropTm(model, n = 100):
    sum = 0
    model.load_state_dict(torch.load('baselines/autoenc_basic/Weights/model_weights.pth'))
    for i in range(n):
        X = torch.rand(3, 4, 15, device=device) # tensor with random numbers
        print(i)
        start_time = time.time()
        rawOut = model(X)
        sum += (time.time() - start_time)
    rslt = sum/n
    print(rslt, "seconds or", rslt*1000, "ms")
    return rslt

# MAIN
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

    # space for execution of a method below
    tmp = avgFrwdPropTm(model)