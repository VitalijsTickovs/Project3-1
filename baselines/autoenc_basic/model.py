import torch
from torch import nn

# Layer structure:
#   in -> encoder_1 -> encoder_2 -> code -> decoder_1 -> decoder_2 -> out
# - 2 layers for decoding and encoding

## METHODS & CLASSES
# TODO: Site pytorch documentation
class ED_Network(nn.Module): # inherit from nn.Module
    def __init__(self): # initialise layers
        super().__init__()
        self.flatten = nn.Flatten() # flatten variable of self
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4*15*3, 50), # 4 features * 15 different time points * 3 coordinates per feature
                                 #   features: left arm, right arm, head, spine point (ignoring the object coordinates for now)
            nn.ReLU(), # keep ReLU for now (lower computation) althoguh tempted to add LeakyReLU (dying neuron problem resolved)
            nn.Linear(50, 40), # encoder_1 -> encoder_2
            nn.ReLU(),
            nn.Linear(40, 30), # encoder_2 -> code
            nn.ReLU(),
            nn.Linear(30, 40), # code -> decoder_1
            nn.ReLU(),
            nn.Linear(40, 50), # decoder_1 -> decoder_2
            nn.ReLU(),
            nn.Linear(50, 4*15*3), # decoder_2 -> output
        )

    def forward(self, x):
        x = self.flatten(x)
        rawOut = self.linear_relu_stack(x)
        return rawOut


# Parameters:
#   features - 2d array input motion [X moves][15 time pts][4 keypoints] e.g. { {{a1,b1,c1,d1}, {a2,b2,c2,d2}...},  ...}
#   output - 2d array output motion 
def train_loop(features, outputs, model, loss_fn, optimizer):
    size = len(features) # number of instances
    model.train() # model in training mode

    iter = 0
    for mat in features:
        # Compute prediction and loss
        pred = model(mat)
        loss = loss_fn(pred, outputs[iter])

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if iter % 100 == 0:
            loss = loss.item()
            print(f"loss: {loss:>7f}  [{iter:>5d}/{size:>5d}]")
        iter+=1

def test_loop(features, outputs, model, loss_fn):
    model.eval() # evaluation mode
    size = len(features)
    av_loss = 0

    iter = 0
    with torch.no_grad(): # ensure less computation overhead by avoiding gradient computations
        for mat in features:
            pred = model(mat)
            av_loss += loss_fn(pred, outputs[iter]).item()
            iter+=1

    av_loss /= iter
    print(f"Avg loss: {av_loss:>8f} \n")

## MAIN


