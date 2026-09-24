import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# Assuming NeuralTexture is defined in neural_texture.py
from neural_texture import NeuralTexture

def get_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def calc_psnr(mse):
    # Avoid log(0)
    if mse < 1e-10: return 100.0
    return -10.0 * np.log10(mse)

def get_model_size_kb(model):
    """Calculates the size of the neural network in Kilobytes (assuming 32-bit floats)."""
    total_params = sum(p.numel() for p in model.parameters())
    return (total_params * 4) / 1024.0

@torch.no_grad()
def reconstruct_full_image(model, coords, H, W, device):
    """Runs the full coordinate grid through the model to generate the final image."""
    model.eval()
    # Process in chunks to avoid out-of-memory errors on smaller GPUs
    chunk_size = 65536
    preds = []
    for i in range(0, coords.shape[0], chunk_size):
        chunk = coords[i:i+chunk_size].to(device)
        preds.append(model(chunk))
    
    # Concatenate, reshape to (H, W, 3), clip to [0, 1], and convert to numpy
    full_recon = torch.cat(preds, dim=0).view(H, W, 3)
    full_recon = torch.clamp(full_recon, 0.0, 1.0).cpu().numpy()
    return full_recon

# ==========================================
# 1. SETUP EXPERIMENT
# ==========================================
configs = {
    "Small":  {"res": (64,), "feat": 2},
    "Medium": {"res": (16, 32, 64), "feat": 2},
    "Large":  {"res": (16, 32, 64, 128), "feat": 4}
}
# textures = ["bricks.png", "clouds.png", "gradient.png"]
textures = ["stone.jpg", "wood.png", "canvas.jpg"]
device = get_device()

psnr_histories = {}
results_stats = {}

# ==========================================
# 2. MAIN TRAINING & EVALUATION LOOP
# ==========================================
for tex_name in textures:
    if not os.path.exists(tex_name):
        print(f"Skipping {tex_name}, file not found.")
        continue
        
    print(f"\n{'='*40}\nProcessing {tex_name}...\n{'='*40}")
    
    # Load original image
    image_np = (np.asarray(Image.open(tex_name).convert("RGB")) / 255.0).astype(np.float32)
    image_tensor = torch.tensor(image_np)
    H, W, _ = image_tensor.shape
    raw_size_kb = (H * W * 3) / 1024.0

    # Create coordinate grid once per image
    u_coords = (torch.arange(W, dtype=torch.float32) + 0.5) / W
    v_coords = (torch.arange(H, dtype=torch.float32) + 0.5) / H
    
    # Create a 2D grid of coordinates using meshgrid
    # indexing='ij' means first output varies along H (y/v), second along W (x/u)
    grid_v, grid_u = torch.meshgrid(v_coords, u_coords, indexing='ij')
    
    coords_full = torch.stack([grid_u, grid_v], dim=-1).view(-1, 2) # (N, 2)
    target_full = image_tensor.view(-1, 3) #(N, 3)
    
    N_total = coords_full.shape[0]
    batch_size = 16384

    # Dictionaries to hold reconstructions for the side-by-side plot
    reconstructions = {}
    
    for cfg_name, cfg in configs.items():
        print(f"\n--- Training {cfg_name} ---")
        model = NeuralTexture(cfg["res"], cfg["feat"]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)

        coords_gpu = coords_full.to(device)
        target_gpu = target_full.to(device)

        current_run_psnr_list = []
        
        # Training loop
        model.train()
        for step in range(2000):
            indices = torch.randint(0, N_total, (batch_size,), device=device)
            batch_coords = coords_gpu[indices]
            batch_target = target_gpu[indices]
            
            batch_preds = model(batch_coords)
            loss = F.mse_loss(batch_preds, batch_target)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            # Log PSNR every 10 steps for smooth plotting
            if step % 10 == 0:
                step_psnr = calc_psnr(loss.item())
                current_run_psnr_list.append((step, step_psnr))
                
            if step % 500 == 0 or step == 1999:
                print(f"Step {step:4d} | Loss: {loss.item():.5f} | PSNR: {calc_psnr(loss.item()):.2f} dB")

        # Save history
        psnr_histories[f"{tex_name}_{cfg_name}"] = current_run_psnr_list

        # Reconstruct full image
        recon_img = reconstruct_full_image(model, coords_full, H, W, device)
        reconstructions[cfg_name] = recon_img
        
        # Calculate final stats
        final_mse = np.mean((recon_img - image_np) ** 2)
        final_psnr = calc_psnr(final_mse)
        model_size_kb = get_model_size_kb(model)
        comp_ratio = raw_size_kb / model_size_kb
        
        results_stats[f"{tex_name}_{cfg_name}"] = {
            "psnr": final_psnr,
            "size_kb": model_size_kb,
            "ratio": comp_ratio
        }

    # ==========================================
    # 3. PLOT SIDE-BY-SIDE RECONSTRUCTIONS
    # ==========================================
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original
    axes[0].imshow(image_np)
    axes[0].set_title(f"Original\nSize: {raw_size_kb:.1f} KB")
    axes[0].axis("off")
    
    # Neural Configurations
    for i, cfg_name in enumerate(["Small", "Medium", "Large"]):
        ax = axes[i+1]
        ax.imshow(reconstructions[cfg_name])
        stats = results_stats[f"{tex_name}_{cfg_name}"]
        ax.set_title(f"{cfg_name}\nPSNR: {stats['psnr']:.2f} dB\nSize: {stats['size_kb']:.1f} KB ({stats['ratio']:.1f}x)")
        ax.axis("off")
        
    plt.tight_layout()
    out_path = f"{tex_name.split('.')[0]}_neural_comparison.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved reconstruction comparison to {out_path}")

# ==========================================
# 4. PLOT PSNR HISTORY CURVES
# ==========================================
fig, axes = plt.subplots(1, len(textures), figsize=(18, 5))

for i, tex_name in enumerate(textures):
    if not os.path.exists(tex_name): continue
    ax = axes[i]
    
    for cfg_name in ["Small", "Medium", "Large"]:
        history = psnr_histories[f"{tex_name}_{cfg_name}"]
        steps, psnrs = zip(*history)
        ax.plot(steps, psnrs, label=cfg_name)
        
    ax.set_title(f"Training PSNR: {tex_name}")
    ax.set_xlabel("Steps")
    ax.set_ylabel("PSNR (dB)")
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.savefig("psnr_training_curves.png", dpi=150)
plt.close(fig)
print("\nSaved PSNR training curves to psnr_training_curves.png")