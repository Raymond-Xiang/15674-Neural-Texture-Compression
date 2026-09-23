import torch, torch.nn as nn, torch.nn.functional as F
from feature_grid import FeatureGrid
from decoder_mlp import ColorMLP

class NeuralTexture(nn.Module):
    def __init__(self):
        super().__init__()
        self.grid = FeatureGrid()
        self.mlp  = ColorMLP(self.grid.out_dim)
    def forward(self, uv):
        return self.mlp(self.grid(uv))
