import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os

from neural_texture import NeuralTexture

def get_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def calc_psnr(mse):
    if mse < 1e-10: return 100.0
    return -10.0 * np.log10(mse)

def get_model_size_kb(model):
    """Calculates float32 model size."""
    return (sum(p.numel() for p in model.parameters()) * 4) / 1024.0

def get_quantized_size_kb(model, quantize_mlp=False):
    """Calculates size assuming 8-bit grids and optionally 8-bit MLP."""
    grid_params = sum(p.numel() for p in model.grid.grids)
    mlp_params = sum(p.numel() for p in model.mlp.parameters())
    
    # 1 byte per quantized value + 8 bytes for the (lo, scale) float32 pair per grid
    grid_bytes = (grid_params * 1) + (len(model.grid.grids) * 8)
    
    if quantize_mlp:
        mlp_bytes = (mlp_params * 1) + (len(list(model.mlp.parameters())) * 8)
    else:
        mlp_bytes = mlp_params * 4  # Kept as float32
        
    return (grid_bytes + mlp_bytes) / 1024.0

# ==========================================
# 1. QUANTIZATION FUNCTIONS
# ==========================================
def quantize_uint8(x):
    """Maps float tensor to 256 evenly spaced levels."""
    lo, hi = x.min(), x.max()
    scale = (hi - lo) / 255.0 
    
    # Avoid division by zero if all values are identical
    if scale == 0: 
        scale = 1.0
        
    q = torch.round((x - lo) / scale).clamp(0, 255)
    x_hat = lo + q * scale
    return q, lo, scale, x_hat

def quantize_model(model, quantize_mlp=False):
    """Replaces model float32 parameters with their dequantized approximations."""
    with torch.no_grad():
        for grid in model.grid.grids:
            q, lo, scale, x_hat = quantize_uint8(grid.data)
            grid.data.copy_(x_hat)

        if quantize_mlp:
            for p in model.mlp.parameters():
                q, lo, scale, x_hat = quantize_uint8(p.data)
                p.data.copy_(x_hat)

@torch.no_grad()
def reconstruct_full_image(model, coords, H, W, device):
    model.eval()
    chunk_size = 65536
    preds = []
    for i in range(0, coords.shape[0], chunk_size):
        preds.append(model(coords[i:i+chunk_size].to(device)))
    
    full_recon = torch.cat(preds, dim=0).view(H, W, 3)
    return torch.clamp(full_recon, 0.0, 1.0).cpu().numpy()

# ==========================================
# 2. MAIN LOOP (Medium Model Only)
# ==========================================
textures = ["bricks.png", "clouds.png", "gradient.png"]
cfg = {"res": (16, 32, 64), "feat": 2}
device = get_device()

for tex_name in textures:
    if not os.path.exists(tex_name): continue
        
    print(f"\n{'='*40}\nProcessing {tex_name}...\n{'='*40}")
    
    image_np = (np.asarray(Image.open(tex_name).convert("RGB")) / 255.0).astype(np.float32)
    image_tensor = torch.tensor(image_np)
    H, W, _ = image_tensor.shape
    raw_size_kb = (H * W * 3) / 1024.0

    u_coords = (torch.arange(W, dtype=torch.float32) + 0.5) / W
    v_coords = (torch.arange(H, dtype=torch.float32) + 0.5) / H
    grid_v, grid_u = torch.meshgrid(v_coords, u_coords, indexing='ij')
    coords_full = torch.stack([grid_u, grid_v], dim=-1).view(-1, 2)
    target_full = image_tensor.view(-1, 3)
    
    N_total, batch_size = coords_full.shape[0], 16384

    model = NeuralTexture(cfg["res"], cfg["feat"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    coords_gpu, target_gpu = coords_full.to(device), target_full.to(device)
    
    # 1. Train Float32 Model
    model.train()
    for step in range(2000):
        indices = torch.randint(0, N_total, (batch_size,), device=device)
        loss = F.mse_loss(model(coords_gpu[indices]), target_gpu[indices])
        opt.zero_grad()
        loss.backward()
        opt.step()

    # 2. Evaluate Float32
    recon_float32 = reconstruct_full_image(model, coords_full, H, W, device)
    psnr_f32 = calc_psnr(np.mean((recon_float32 - image_np) ** 2))
    size_f32 = get_model_size_kb(model)
    
    # 3. Apply Quantization & Evaluate UINT8
    quantize_model(model, quantize_mlp=False)
    recon_uint8 = reconstruct_full_image(model, coords_full, H, W, device)
    psnr_u8 = calc_psnr(np.mean((recon_uint8 - image_np) ** 2))
    size_u8 = get_quantized_size_kb(model, quantize_mlp=False)

    print(f"--- Results for {tex_name} ---")
    print(f"[Float32] PSNR: {psnr_f32:.2f} dB | Size: {size_f32:.1f} KB | Ratio: {raw_size_kb/size_f32:.1f}x")
    print(f"[UINT8]   PSNR: {psnr_u8:.2f} dB | Size: {size_u8:.1f} KB | Ratio: {raw_size_kb/size_u8:.1f}x")
    print(f"-> Quality Drop: {psnr_f32 - psnr_u8:.2f} dB")