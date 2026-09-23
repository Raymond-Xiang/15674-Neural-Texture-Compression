import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class TextureSampler:
    
    def __init__(self, image_path: str):
        self.my_texture = (np.asarray(Image.open(image_path)) / 255).astype(np.float32)

    def sample(self, u, v):
        H = self.my_texture.shape[0]
        W = self.my_texture.shape[1]
        i = W * u - 0.5
        j = H * v - 0.5

        # 1. make sure there are no negatives, and no overbound values
        i = np.clip(i, 0.0, W - 1.0)
        j = np.clip(j, 0.0, H - 1.0)

        # 2. find one side of adjacent texels
        i0 = np.floor(i).astype(np.int32)
        j0 = np.floor(j).astype(np.int32)

        # and the other side
        i1 = np.minimum(i0 + 1, W - 1)
        j1 = np.minimum(j0 + 1, H - 1)

        # 3. compute weight for interpolation (s, t)
        u_ratio = i - i0
        u_ratio = u_ratio.reshape(-1, 1)
        v_ratio = j - j0
        v_ratio = v_ratio.reshape(-1, 1)

        # 4. find those texels
        c00 = self.my_texture[j0, i0]
        c10 = self.my_texture[j0, i1]
        c01 = self.my_texture[j1, i0]
        c11 = self.my_texture[j1, i1]

        # 5. do the bilinear interpolation
        color_top = c00 * (1 - u_ratio) + c10 * u_ratio
        color_bottom = c01 * (1 - u_ratio) + c11 * u_ratio
        final_color = color_top * (1 - v_ratio) + color_bottom * v_ratio

        return final_color


class S3TCSampler:
    
    def __init__(self, image_path: str):
        image = (np.asarray(Image.open(image_path)) / 255.0).astype(np.float32)
        self.H, self.W = image.shape[:2]
        
        # only store Endpoints and Indices!
        self.endpoints, self.indices = self.compress(image)

    def compress(self, image):
        """
        (H, W, 3) -> endpoints  (H/4, W/4, 2, 3)
                  -> indices (H/4, W/4, 16)
        """
        H, W = self.H, self.W
        
        # 1. Cut the image into 4*4 blocks
        # (H/4, W/4, 16, 3)
        blocks = image.reshape(H // 4, 4, W // 4, 4, 3).transpose(0, 2, 1, 3, 4).reshape(H // 4, W // 4, 16, 3)

        # 2. Compute endpoints, c0, c1
        # I chose to use directly the max min values in each channel of (R,G,B)
        c0 = blocks.min(axis=2)  # shape: (H/4, W/4, 3)
        c1 = blocks.max(axis=2)  # shape: (H/4, W/4, 3)

        # 3. Quantize to 16 bits (rgb565 format)
        levels = np.array([31.0, 63.0, 31.0])
        c0 = np.round(c0 * levels) / levels
        c1 = np.round(c1 * levels) / levels

        # 4. Construct the other 2 colors, our palette should have 4 colors in total
        palette = np.zeros((H // 4, W // 4, 4, 3))
        palette[:, :, 0, :] = c0
        palette[:, :, 1, :] = c1
        palette[:, :, 2, :] = (2 * c0 + c1) / 3.0
        palette[:, :, 3, :] = (c0 + 2 * c1) / 3.0

        # 5. for our 16-pixel blocks, compute the color (from the palette) closest to the actual color
        # blocks: (H/4, W/4, 16, 1, 3)
        # palette: (H/4, W/4, 1, 4, 3)
         
        # This basically calculates the euclidean distance between actual color to C0,1,2,3
        # and then finds the argmin, recall we should calculate...
        # index(p) = argmin_j ||p − c_j||²
        diff = blocks[:, :, :, None, :] - palette[:, :, None, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)  # shape: (H/4, W/4, 16, 4)
        indices = np.argmin(dist_sq, axis=-1) # shape: (H/4, W/4, 16)

        endpoints = np.stack([c0, c1], axis=2) # shape: (H/4, W/4, 2, 3)
        
        return endpoints, indices

    def get_texel(self, j, i):
        # which block do I live in?
        block_j = j // 4
        block_i = i // 4
        pixel_idx = (j % 4) * 4 + (i % 4)

        # read endpoints, and compute c2,c3
        c0 = self.endpoints[block_j, block_i, 0]
        c1 = self.endpoints[block_j, block_i, 1]
        c2 = (2 * c0 + c1) / 3.0
        c3 = (c0 + 2 * c1) / 3.0
        
        idx = self.indices[block_j, block_i, pixel_idx]

        # find out which color I should paint it
        if np.isscalar(idx):
            palette = np.array([c0, c1, c2, c3])
            return palette[idx]
        else:
            palette = np.stack([c0, c1, c2, c3], axis=1) # shape: (N, 4, 3)
            batch_indices = np.arange(idx.shape[0])
            return palette[batch_indices, idx]           # shape: (N, 3)

    def sample(self, u, v):
        H, W = self.H, self.W
        i = W * u - 0.5
        j = H * v - 0.5

        i = np.clip(i, 0.0, W - 1.0)
        j = np.clip(j, 0.0, H - 1.0)

        i0 = np.floor(i).astype(np.int32)
        j0 = np.floor(j).astype(np.int32)
        i1 = np.minimum(i0 + 1, W - 1)
        j1 = np.minimum(j0 + 1, H - 1)

        # NOTE: I am not looking at the pixel themselves for color, I go look at endpoints and indices to deduce
        # the 4 adjacent pixel colors
        c00 = self.get_texel(j0, i0)
        c10 = self.get_texel(j0, i1)
        c01 = self.get_texel(j1, i0)
        c11 = self.get_texel(j1, i1)

        u_ratio = (i - i0)[..., None]
        v_ratio = (j - j0)[..., None]

        color_top = c00 * (1 - u_ratio) + c10 * u_ratio
        color_bottom = c01 * (1 - u_ratio) + c11 * u_ratio
        final_color = color_top * (1 - v_ratio) + color_bottom * v_ratio

        return final_color

def reconstruct_full_image(sampler, H, W):
    """Sample every texel center via sampler.sample(u, v) -- same convention as P1,
    so P1 ground truth, P2 S3TC, and later the neural model are all compared
    on identical grid points."""
    us = (np.arange(W) + 0.5) / W
    vs = (np.arange(H) + 0.5) / H
    uu, vv = np.meshgrid(us, vs)  # each (H, W)
    colors = sampler.sample(uu.reshape(-1), vv.reshape(-1))  # (H*W, 3)
    return colors.reshape(H, W, 3)
 
 
def psnr(reconstruction, original):
    mse = np.mean((reconstruction - original) ** 2)
    return -10.0 * np.log10(mse)  # MAX=1.0 for normalized images
 
 
def compression_stats(H, W):
    raw_bytes = H * W * 3
    s3tc_bytes = (W * H / 16) * 8  # 8 bytes per 4x4 block == 4 bits/texel
    return raw_bytes, s3tc_bytes, raw_bytes / s3tc_bytes
 
 
def run_one(image_path, out_prefix):
    original = (
        np.asarray(Image.open(image_path).convert("RGB")) / 255.0
    ).astype(np.float32)
    H, W = original.shape[:2]
 
    s3tc = S3TCSampler(image_path)
    recon = reconstruct_full_image(s3tc, H, W)
 
    p = psnr(recon, original)
    raw_bytes, s3tc_bytes, ratio = compression_stats(H, W)
 
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(np.clip(recon, 0, 1))
    axes[1].set_title(f"S3TC reconstruction\nPSNR={p:.2f} dB, ratio={ratio:.1f}x")
    axes[1].axis("off")
    plt.tight_layout()
    out_path = f"{out_prefix}_s3tc_comparison.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
 
    return {
        "texture": out_prefix,
        "psnr_db": p,
        "raw_bytes": raw_bytes,
        "s3tc_bytes": int(s3tc_bytes),
        "compression_ratio": ratio,
        "comparison_image": out_path,
    }
 
 
if __name__ == "__main__":
    textures = ["gradient.png", "bricks.png", "clouds.png"]
    results = []
    for path in textures:
        name = path.rsplit(".", 1)[0]
        result = run_one(path, name)
        results.append(result)
        print(
            f"{result['texture']:>10s}  PSNR={result['psnr_db']:6.2f} dB  "
            f"compression={result['compression_ratio']:5.1f}x  "
            f"({result['raw_bytes']} -> {result['s3tc_bytes']} bytes)  "
            f"-> {result['comparison_image']}"
        )