import torch
from torch import nn

import sys
import numpy as np

# Layer structure:
#   in -> encoder_1 -> encoder_2 -> code -> decoder_1 -> decoder_2 -> out
# - 2 layers for decoding and encoding

## METHODS & CLASSES
# TODO: Site pytorch documentation
class ED_Network(nn.Module): # inherit from nn.Module
    def __init__(self): # initialise layers
        super().__init__()
        self.flatten = nn.Flatten(start_dim=0) # by default retains 1 dimension (start_dim=1) because wants to keep batches
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(34*3, 50), # 34 features * 3 coordinates per feature
                                 #   features: left arm, right arm, head, spine point (ignoring the object coordinates for now)
            nn.ReLU(), # keep ReLU for now (lower computation) althoguh tempted to add LeakyReLU (dying neuron problem resolved)
            nn.Linear(50, 40), # encoder_1 -> encoder_2
            nn.ReLU(),
            nn.Linear(40, 30),
            nn.ReLU(),
            nn.Linear(30, 40), # encoder_2 -> code
            nn.ReLU(),
            nn.Linear(40, 50),
            nn.ReLU(),
            nn.Linear(50, 34*3), # decoder_2 -> output
        )

    def forward(self, x):
        x = self.flatten(x)
        rawOut = self.linear_relu_stack(x)
        return rawOut

    def expFlat(self, data):
        return self.flatten(data)


# Parameters:
#   features - 2d array input skeleton [X skeletons][34 keypoints][3 coordinates]
#   output - 2d array output skeleton (same as input) 
def train_loop(features, outputs, model, loss_fn, optimizer):
    size = len(features) # number of instances
    model.train() # model in training mode

    iter = 0
    for mat in features:
        # Compute prediction and loss
        pred = model(mat)
        loss = loss_fn(pred, torch.flatten(outputs[iter]))

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if iter % 100 == 0:
            loss = loss.item()
            #print(f"loss: {loss:>7f}  [{iter:>5d}/{size:>5d}]") # show loss every 100th instance
        iter+=1

# features and outputs is a numpy array
def test_loop(features, outputs, model, loss_fn, showSample=-1):
    model.eval() # evaluation mode
    av_loss = 0

    with torch.no_grad(): # ensure less computation overhead by avoiding gradient computations
        for i, mat in enumerate(features):
            pred = model(mat)
            av_loss += loss_fn(pred, torch.flatten(outputs[i])).item() # flatten output to ensure they are the same

    av_loss /= (i+1)
    print(f"avereage MAE for all samples: {av_loss:>8f} \n")

    if showSample!=-1:
        torch.set_printoptions(sci_mode=False)
        pred = model(features[showSample])
        print(torch.reshape(pred,[34,3]))
        print("\n")
        print(outputs[showSample])

# similar to original test loop, but made specially for P&C to calculate percentage wise error
def test_loop2(features, outputs, model, loss_fn, showSample=-1):
    model.eval()
    av_loss = 0

    with torch.no_grad(): # ensure less computation overhead by avoiding gradient computations
        for i, mat in enumerate(features):
            pred = model(mat)
            num = np.subtract(pred.numpy(), torch.flatten(outputs[i]).numpy()) # flatten output to ensure they are the same
            perc_mat = np.divide(np.abs(num), np.add(np.abs(torch.flatten(outputs[i]).numpy()), 0.0000000001))
            loss = np.mean(perc_mat)
            av_loss += loss

    av_loss /= (i+1)
    print(f"avereage relative MAE for all samples: {av_loss:>8f} \n")

    if showSample!=-1:
        torch.set_printoptions(sci_mode=False)
        pred = model(features[showSample])
        print(torch.reshape(pred,[34,3]))
        print("\n")
        print(outputs[showSample])


def optimizationLoop(features, outputs, model, loss_fn, optimizer, epochs=10):
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(features, outputs, model, loss_fn, optimizer)
        test_loop(features, outputs, model, loss_fn)


## MAIN


