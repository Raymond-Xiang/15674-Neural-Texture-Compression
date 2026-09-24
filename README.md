# Neural Texture Compression (15-674 A1)

Compresses a single RGB texture into a small multi-resolution feature grid +
tiny MLP decoder, and compares it against a classical S3TC (DXT1) block-compression
baseline. See the write-up PDF for the full analysis; this README explains how
to reproduce the numbers and figures it references.

## Layout

| File | Problem | What it contains |
|---|---|---|
| `s3tc.py` | P1, P2 | Bilinear `TextureSampler` (ground truth) and the `S3TCSampler` block-compression baseline |
| `feature_grid.py` | P3 | `FeatureGrid` — multi-resolution learnable grids, `F.grid_sample` lookup |
| `decoder_mlp.py` | P4 | `ColorMLP` — 2×64 hidden layers + Sigmoid output |
| `neural_texture.py` | P5 | `NeuralTexture` — wraps `FeatureGrid` + `ColorMLP` into one model |
| `train.py` | P5, P6 | Training loop; fits one `NeuralTexture` to one image and reports PSNR / compression ratio |
| `train_quantized.py` | P7 | Post-training 8-bit quantization of a fitted model's grid (and optionally MLP) weights |


## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install torch torchvision numpy pillow matplotlib
```

## Reproducing each part

None of the scripts take command-line arguments. Each has a `texture_names`
list (and, for `train.py` / `train_quantized.py`, a model-size setting) near
the bottom of the file — edit that, then run the script directly. There's no
saved-checkpoint mechanism: every run trains its model from scratch.

**P1 + P2 — bilinear sampler and S3TC baseline**

```bash
python s3tc.py
```

Edit `texture_names` at the bottom of the file to the texture(s) you want
first. Runs the S3TC baseline on each, prints PSNR and compression ratio
against the raw texture, and saves a side-by-side original-vs-reconstruction
PNG per texture.

**P3–P5 — fit the neural model to one texture**

```bash
python train.py
```

Edit `texture_names` and the model-size setting at the bottom of the file
first. Trains a `NeuralTexture` (grid + MLP) on each listed texture for 2000
steps (batch size 16384, Adam, lr=1e-2) from scratch, then prints final PSNR
and compression ratio.

**P6 — full results grid (3 textures × 3 sizes)**

Set `texture_names` to all three provided textures, then run `train.py` once
per model size (`small`, `medium`, `large`), changing the size setting
between runs — nine runs total. There's currently no single script that
sweeps all nine sizes/textures automatically; running `train.py` nine times
with different settings is the current way to reproduce the P6 table.

**P7 — quantization**

```bash
python train_quantized.py
```

Independent of `train.py` — it trains a fresh `NeuralTexture` from scratch and then quantizes the
stored grid values from float32 to 8-bit.
Edit `texture_names` and the size setting at the bottom of the file first.

**P8 — your own textures**

Add three self-sourced textures to `data/` (see the `.gitignore` note
above), then run `train.py` on them the same way as P6, once per model size.

## Results summary

| Texture | Method | PSNR (dB) | Compression ratio |
|---|---|---|---|
| Gradient | S3TC | 43.03 | 6.0x |
| Gradient | Neural (Medium) | 58.53 | 12.6x |
| Bricks | S3TC | 41.39 | 6.0x |
| Bricks | Neural (Large) | 40.88 | 2.1x |
| Clouds | S3TC | 35.95 | 6.0x |
| Clouds | Neural (Small) | 39.49 | 15.4x |

Full 3×3 table, training curves, and per-texture analysis are in the write-up PDF.
