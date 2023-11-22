import torch
from torch import nn
from torch.utils.data import DataLoader

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
            nn.Linear(4*15, 50), # 4 features * 15 different time points
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
            nn.Linear(50, 60), # decoder_2 -> output
        )

    def forward(self, x):
        x = self.flatten(x)
        rawOut = self.linear_relu_stack(x)
        return rawOut


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