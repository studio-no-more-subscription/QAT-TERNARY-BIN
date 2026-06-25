# Ternary & Binary Weight Inference Models

Pretrained models with quantized weights, packaged for inference and benchmarking. Each model ships in **two** weight formats:

- **Ternary** `{-1, 0, +1}` — 1.58-bit (BitNet-style), 2-bit packed (4 values/byte)
- **Binary** `{-1, +1}` — 1-bit, packed (8 values/byte), ~2× smaller quantized weights

Since the ternary weights have very low sparsity (<0.5% zeros for LLM), the binary variant gives nearly identical results at half the quantized-weight storage.

Each subfolder is **self-contained**: one `run.py` file + weights + benchmark artifacts. No external project code required.

## Models

| Model | Folder | Weights (ternary / binary) | Task |
|-------|--------|----------------------------|------|
| LLM 100M(Will be uploaded as soon as possible.) | [`llm/`](llm/) | 54 MB / 51 MB (zipped) | Text generation (TinyStories-style) |
| MNIST CNN | [`mnist_cnn/`](mnist_cnn/) | 49 KB / 32 KB | Digit classification |
| Image Generator(currently not working and not tested) | [`image_generator/`](image_generator/) | 1.3 MB / 1.0 MB | Autoregressive pixel image generation |

## Quick Start

```bash
# LLM — generate text (ternary or binary)
cd llm && python run.py --mode ternary --prompt "Once upon a time," --max-tokens 200
cd llm && python run.py --mode binary  --prompt "Once upon a time," --max-tokens 200

# MNIST CNN — evaluate test accuracy
cd mnist_cnn && python run.py --mode ternary
cd mnist_cnn && python run.py --mode binary

# Image Generator — generate sample digit images
cd image_generator && python run.py --mode ternary --num-samples 16
cd image_generator && python run.py --mode binary  --num-samples 16
```

`--mode` defaults to `ternary`. Both modes use the same model architecture; only the weight file differs.

## Requirements

- Python 3.10+
- `torch`, `numpy`
- LLM also needs: `tiktoken`
- MNIST CNN also needs: `torchvision`
- Image Generator also needs: `Pillow`

## Folder Structure

```
github/
├── README.md
├── llm/
│   ├── run.py
│   ├── weights/
│   │   ├── ckpt.pt.zip          ← ternary (2-bit packed)
│   │   └── ckpt_binary.pt.zip   ← binary  (1-bit packed)
├── mnist_cnn/
│   ├── run.py
│   ├── weights/
│   │   ├── ternary.pt           ← ternary (2-bit packed)
│   │   └── binary.pt            ← binary  (1-bit packed)
│   └── benchmark/
└── image_generator/
    ├── run.py
    ├── weights/
    │   ├── ckpt.pt              ← ternary (2-bit packed)
    │   └── ckpt_binary.pt       ← binary  (1-bit packed)
    └── benchmark/
        └── samples/
```

## Weight Format

**Ternary** (2-bit packed, 4 values/byte):
- `{-1, 0, +1}` → 2-bit codes `{0b00, 0b01, 0b10}`

**Binary** (1-bit packed, 8 values/byte):
- `{-1, +1}` → 1-bit codes `{0, 1}` (zeros snapped to +1)

Unpacking logic for both formats is inlined in each `run.py`. Non-quantized parameters (embeddings, norms, scales, biases, final classifiers) are stored as standard fp32 tensors.

## Acknowledgements & License

This project is a quantized version of the original `mnist-cnn` by Han Hao.

Copyright (c) 2017 Han Hao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
