import torch, torch.nn as nn, torch.nn.functional as F


class ColorMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        # 2 hidden layers of width 64 (Linear + ReLU), then Linear -> 3 and a Sigmoid (RGB in [0, 1])
        self.hidden_dim = 64
        self.net = nn.Sequential(nn.Linear(in_dim, self.hidden_dim), 
                                nn.ReLU(),
                                nn.Linear(self.hidden_dim, self.hidden_dim), 
                                nn.ReLU(),
                                nn.Linear(self.hidden_dim, 3),
                                nn.Sigmoid())

    def forward(self, x):
        return self.net(x)