# Image Generator — Autoregressive Pixel Generation (Inference)

Autoregressive pixel-level image generator with quantized weights. Generates 32×32 grayscale digit images conditioned on a class label (0–9).

Supports two weight formats:
- **Ternary** `{-1, 0, +1}` — 2-bit packed (default)
- **Binary** `{-1, +1}` — 1-bit packed (smaller, near-zero sparsity makes binary a close substitute)

## Quick Start

```bash
# Ternary mode (default)
python run.py

# Binary mode
python run.py --mode binary
```

Generates 16 sample images (one per digit 0–9, plus extras) and saves them as PNGs to `benchmark/samples/`.

## Options

```bash
python run.py \
    --mode ternary \
    --img-size 32 \
    --num-samples 16 \
    --temperature 0.3 \
    --class-id -1 \
    --out-dir ./benchmark/samples \
    --device cpu
```

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `ternary` | Weight format: `ternary` or `binary` |
| `--img-size` | `32` | Output image size (pixels) |
| `--num-samples` | `16` | Number of images to generate |
| `--temperature` | `0.3` | Noise level for generation diversity |
| `--class-id` | `-1` | Digit class 0–9, or -1 for all digits |
| `--out-dir` | `./benchmark/samples` | Output directory for PNGs |
| `--device` | `cpu` | `cpu` or `cuda` |

## Model

| Property | Value |
|----------|-------|
| Architecture | Transformer (pre-LN) |
| Layers | 3 |
| Heads | 4 |
| Embedding dim | 192 |
| Max image size | 32 |
| Position encoding | 2D RoPE (separate x/y axes) |
| Output | Sigmoid per pixel (regression, [0,1]) |
| Class conditioning | Digit label embedding (0–9) |
| Quantized params | Attention + MLP linear weights |
| Start token | 256 |

## Weights

| File | Mode | Size |
|------|------|------|
| `weights/ckpt.pt` | Ternary (2-bit packed) | 1.3 MB |
| `weights/ckpt_binary.pt` | Binary (1-bit packed) | 1.0 MB |

## How It Works

The model generates images pixel-by-pixel in raster order. At each position, it predicts the next pixel value as a continuous [0,1] output via sigmoid. Optional temperature noise adds diversity. Predicted values are discretized to 0–255 for output.
