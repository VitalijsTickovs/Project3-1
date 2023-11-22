# Class containing different procedures combining methods from model.py

# Packages
import torch
from torch import nn

# Local files
from model import ED_Network
from model import train_loop
from model import test_loop



# example forward run of the model
def testForwardRun(full_debug):
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

# example train test loop
def trainTest(DataX, DataY, model, learning_rate= 1e-3):
    lossFunc = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(DataX[0], DataY[0], model, lossFunc, optimizer)
        test_loop(DataX[0], DataY[0], model, lossFunc)
    print("Done!")