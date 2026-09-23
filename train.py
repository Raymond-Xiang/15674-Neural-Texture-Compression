import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from PIL import Image
from neural_texture import NeuralTexture

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# ASSUME: only overfitting to a single image first
image = torch.tensor((np.asarray(Image.open("gradient.png")) / 255.0).astype(np.float32))
H, W, _ = image.shape

# Create 1D arrays of texel-center coordinates in [0, 1]
u_coords = (torch.arange(W, dtype=torch.float32) + 0.5) / W
v_coords = (torch.arange(H, dtype=torch.float32) + 0.5) / H

# Create a 2D grid of coordinates using meshgrid
# indexing='ij' means first output varies along H (y/v), second along W (x/u)
grid_v, grid_u = torch.meshgrid(v_coords, u_coords, indexing='ij')

# Stack u and v, then flatten into a list of N=H*W coordinates
coords = torch.stack([grid_u, grid_v], dim=-1).view(-1, 2)  # Shape: (N, 2)

# Flatten the target image to match
target = image.view(-1, 3)  # Shape: (N, 3)


device = get_device()
model = NeuralTexture().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-2)

coords = coords.to(device)
target = target.to(device)


N_total = coords.shape[0] #i.e. H*W
batch_size = 16384

for step in range(2000):
    # Randomly sample 'batch_size' indices from the full dataset
    indices = torch.randint(0, N_total, (batch_size,), device=device)
    
    # Get the minibatch of coordinates and target colors
    batch_coords = coords[indices]
    batch_target = target[indices]
    
    # Forward pass: predict colors for these UVs
    batch_preds = model(batch_coords)
    
    # Compute the Mean Squared Error (MSE) loss
    loss = F.mse_loss(batch_preds, batch_target)
    
    # Backpropagation
    opt.zero_grad()
    loss.backward()
    opt.step()
    
    # Logging progress (optional, every 200 steps)
    if step % 200 == 0:
        psnr = -10 * torch.log10(loss).item()
        print(f"Step {step:4d} | Loss: {loss.item():.5f} | PSNR: {psnr:.2f} dB")