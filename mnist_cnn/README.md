# MNIST CNN — Quantized Classifier (Inference)

Small CNN with quantized convolutional weights for MNIST digit classification. This folder provides inference and accuracy benchmarking only. Ships in both **ternary** `{-1,0,+1}` (2-bit) and **binary** `{-1,+1}` (1-bit) weight formats.

## Quick Start

```bash
python run.py                    # ternary (default)
python run.py --mode binary      # binary weights
```

Loads the selected quantized model, evaluates it on the MNIST test set, and prints test accuracy.

## Options

```bash
python run.py --mode ternary --device cpu --batch-size 256 --data-dir ./data
```

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `ternary` | `ternary` ({-1,0,+1}, 2-bit) or `binary` ({-1,+1}, 1-bit) |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--batch-size` | `256` | Evaluation batch size |
| `--data-dir` | `./data` | MNIST data directory (auto-downloads if missing) |

## Architecture

| Layer | Type | Shape | Weights |
|-------|------|-------|---------|
| conv1 | TernaryConv2d | 1→32, k3, s1, p1 | quantized |
| bn1 | TernaryBatchNorm2d | 32 | gamma=quantized, beta=fp32 |
| conv2 | TernaryConv2d | 32→64, k3, s2, p1 | quantized |
| bn2 | TernaryBatchNorm2d | 64 | gamma=quantized, beta=fp32 |
| conv3 | TernaryConv2d | 64→128, k3, s2, p1 | quantized |
| bn3 | TernaryBatchNorm2d | 128 | gamma=quantized, beta=fp32 |
| pool | AdaptiveAvgPool2d | →1×1 | — |
| fc | Linear | 128→10 | fp32 |

## Weights

| File | Model | Size |
|------|-------|------|
| `weights/ternary.pt` | Ternary CNN (2-bit packed) | 49 KB |
| `weights/binary.pt` | Binary CNN (1-bit packed) | 32 KB |

## Benchmark Results

| Model | Test Accuracy |
|-------|---------------|
| Ternary CNN ({-1,0,+1}) | 46.97% |
| Binary CNN ({-1,+1}) | 19.68% |

The ternary model uses 2-bit weights for all convolutions and BN gamma. The binary variant snaps zeros to +1, giving 1-bit storage at a further accuracy trade-off (this model has ~6.5% zeros, so the drop is more noticeable than in the LLM/image generator where sparsity is <0.5%).
