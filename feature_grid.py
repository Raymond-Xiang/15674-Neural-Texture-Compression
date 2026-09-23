import torch, torch.nn as nn, torch.nn.functional as F

class FeatureGrid(nn.Module):
    def __init__(self, resolutions=(16, 32, 64, 128), feat_dim=2):
        super().__init__()
        # one learnable grid per resolution, each shaped (1, feat_dim, R, R)
        # these are the learnable feature maps for my image
        self.grids = nn.ParameterList([
            nn.Parameter(torch.randn(1, feat_dim, R, R) * 0.01)
            for R in resolutions
        ])
        self.out_dim = feat_dim * len(resolutions)

    def forward(self, uv):                     # uv: (N, 2) in [0, 1], N = batch size
        # bilinear-sample each grid at uv (see F.grid_sample, which wants
        # coords in [-1, 1]) and concatenate the features across resolutions

        # uv: (N, 2) in [0, 1]
        batch_size = uv.shape[0]
        
        # 1. Scale coordinates from [0, 1] to [-1, 1]
        # PyTorch's F.grid_sample expects coordinates in the [-1, 1] range.
        uv_scaled = uv * 2.0 - 1.0
        
        # 2. Reshape UVs to match grid_sample's expected input format.
        # grid_sample expects coordinates as (Batch, H_out, W_out, 2).
        # Since our self.grids have Batch=1, we reshape our N points into a (1, 1, N, 2) tensor.
        uv_grid = uv_scaled.view(1, 1, batch_size, 2)
        
        sampled_features = []
        
        for grid in self.grids:
            # grid is (1, feat_dim, R, R)
            # F.grid_sample output will be (1, feat_dim, 1, N)
            sampled = F.grid_sample(
                grid, 
                uv_grid, 
                mode="bilinear", 
                padding_mode="border", 
                align_corners=False
            )
            
            # Reshape the output from (1, feat_dim, 1, N) -> (feat_dim, N) -> (N, feat_dim)
            feat_dim = grid.size(1)
            sampled = sampled.view(feat_dim, batch_size).t() 
            
            sampled_features.append(sampled)
            
        # 3. Concatenate across the resolution dimension
        # Resulting shape: (N, out_dim) where out_dim = feat_dim * len(resolutions)
        return torch.cat(sampled_features, dim=-1)
