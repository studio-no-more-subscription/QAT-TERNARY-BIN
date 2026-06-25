# LLM 100M — Quantized Language Model (Inference)

NanoGPT-style transformer with quantized weights trained on TinyStories + WikiText-2. This folder provides inference and text generation only. Ships in both **ternary** `{-1,0,+1}` (2-bit) and **binary** `{-1,+1}` (1-bit) weight formats.

## Quick Start

```bash
python run.py                          # ternary (default)
python run.py --mode binary            # binary weights
```

Generates text from the default prompt `"Once upon a time,"` and several additional prompts.

## Options

```bash
python run.py \
    --mode ternary \
    --prompt "Once upon a time," \
    --max-tokens 200 \
    --temperature 1.0 \
    --top-p 0.9 \
    --device cpu
```

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `ternary` | `ternary` ({-1,0,+1}, 2-bit) or `binary` ({-1,+1}, 1-bit) |
| `--prompt` | `"Once upon a time,"` | Text prompt to condition generation |
| `--max-tokens` | `200` | Number of new tokens to generate |
| `--temperature` | `1.0` | Sampling temperature |
| `--top-p` | `0.9` | Nucleus sampling threshold |
| `--device` | `cpu` | `cpu` or `cuda` |

## Model

| Property | Value |
|----------|-------|
| Architecture | NanoGPT (pre-LN transformer) |
| Layers | 12 |
| Heads | 8 |
| Embedding dim | 512 |
| Block size | 512 |
| Vocab size | 50,257 (GPT-2 tokenizer) |
| Total params | ~100M |
| Quantized params | Linear weights (QKV, proj, MLP, LM head) |
| Position encoding | RoPE |
| Norm | RMSNorm |
| Weight tying | Embedding ↔ LM head |

## Weights

| File | Format | Size |
|------|--------|------|
| `weights/ckpt.pt.zip` | Ternary (2-bit packed) | 54 MB |
| `weights/ckpt_binary.pt.zip` | Binary (1-bit packed) | 51 MB |

Each zip contains:
- `ckpt.pt` — model state dict with packed quantized weights + fp32 non-quantized params
- `embedding_fp16.npy` — tied embedding (50,257 × 512) stored as fp16 to reduce size
