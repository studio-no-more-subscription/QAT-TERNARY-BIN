#!/usr/bin/env python
"""Self-contained inference runner for PixelImageGenerator.

Usage:
    python run.py [--device cpu] [--img-size 32] [--num-samples 16]
                  [--temperature 0.3] [--class-id -1] [--out-dir ./benchmark/samples]

Requirements: torch, numpy, Pillow
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from PIL import Image


# ============================================================================
# Section 1 — Unpack Packing Codec (2-bit ternary + 1-bit binary)
# ============================================================================

_DECODE_LUT = np.array([-1, 0, 1, 0], dtype=np.int8)
_PACKED_KEY = "_packed_ternary"
_PACKED_BINARY_KEY = "_packed_binary"


def unpack_ternary_values(data, shape, n_total, dtype=torch.float32, device="cpu"):
    """Decode packed 2-bit bytes back to a ternary tensor of *shape*."""
    if n_total == 0:
        return torch.empty(shape, dtype=dtype, device=device)
    arr = np.frombuffer(data, dtype=np.uint8)
    codes = np.empty(arr.shape[0] * 4, dtype=np.uint8)
    codes[0::4] = arr & 0b11
    codes[1::4] = (arr >> 2) & 0b11
    codes[2::4] = (arr >> 4) & 0b11
    codes[3::4] = (arr >> 6) & 0b11
    vals = _DECODE_LUT[codes]
    vals = vals[:n_total]
    t = torch.from_numpy(vals.copy()).to(dtype).reshape(shape)
    return t.to(device)


def is_packed(value):
    return isinstance(value, dict) and value.get(_PACKED_KEY) is True


def unpack_ternary(packed, device="cpu"):
    return unpack_ternary_values(
        packed["data"], packed["shape"], packed["n_total"], device=device
    )


def unpack_binary_values(data, shape, n_total, dtype=torch.float32, device="cpu"):
    """Decode packed 1-bit bytes back to a binary {-1, +1} tensor."""
    if n_total == 0:
        return torch.empty(shape, dtype=dtype, device=device)
    arr = np.frombuffer(data, dtype=np.uint8)
    bits = np.unpackbits(arr, bitorder="little")
    vals = bits.astype(np.int8) * 2 - 1
    vals = vals[:n_total]
    t = torch.from_numpy(vals.copy()).to(dtype).reshape(shape)
    return t.to(device)


def is_packed_binary(value):
    return isinstance(value, dict) and value.get(_PACKED_BINARY_KEY) is True


def unpack_binary(packed, device="cpu"):
    return unpack_binary_values(
        packed["data"], packed["shape"], packed["n_total"], device=device
    )


def unpack_state_dict(state_dict, device="cpu"):
    """Return a copy with packed entries decoded; non-packed pass through."""
    unpacked = {}
    for key, val in state_dict.items():
        if is_packed(val):
            unpacked[key] = unpack_ternary(val, device=device)
        elif is_packed_binary(val):
            unpacked[key] = unpack_binary(val, device=device)
        else:
            unpacked[key] = val
    return unpacked


# ============================================================================
# Section 2 — Shared Components (TernaryLinear, RMSNorm, MLP)
# ============================================================================


class TernaryLinear(nn.Module):
    """Linear layer with ternary {-1, 0, +1} weights."""
    is_ternary = True

    def __init__(self, in_features, out_features, bias=True, weight_dtype=torch.float32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fan_in = in_features
        self.inv_sqrt_fan_in = in_features ** -0.5
        w = torch.empty(out_features, in_features)
        bound = (3.0 / max(in_features, 1)) ** 0.5
        w.uniform_(-bound, bound)
        self.weight = nn.Parameter(torch.sign(w).to(weight_dtype), requires_grad=True)
        self.scale = nn.Parameter(torch.ones(out_features), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=True)
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        y = F.linear(x, self.weight.to(x.dtype), self.bias)
        y = y * self.scale
        return y * self.inv_sqrt_fan_in


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


class MLP(nn.Module):
    def __init__(self, n_embd, dropout=0.0, weight_dtype=torch.float32):
        super().__init__()
        self.c_fc = TernaryLinear(n_embd, 4 * n_embd, bias=False, weight_dtype=weight_dtype)
        self.c_proj = TernaryLinear(4 * n_embd, n_embd, bias=False, weight_dtype=weight_dtype)
        self.dropout = nn.Identity()

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x, approximate="tanh")
        x = self.dropout(x)
        x = self.c_proj(x)
        return self.dropout(x)


# ============================================================================
# Section 3 — 2D RoPE + CausalSelfAttention2D
# ============================================================================


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class CausalSelfAttention2D(nn.Module):
    """Self-attention with 2D rotary position embeddings (split head_dim for x/y)."""

    def __init__(self, n_embd, n_head, max_img_size=64, dropout=0.0, weight_dtype=torch.float32):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.half_dim = self.head_dim // 2

        self.c_attn = TernaryLinear(n_embd, 3 * n_embd, bias=False, weight_dtype=weight_dtype)
        self.c_proj = TernaryLinear(n_embd, n_embd, bias=False, weight_dtype=weight_dtype)

        theta = 1.0 / (10000.0 ** (torch.arange(0, self.half_dim, 2, dtype=torch.float32) / self.half_dim))
        positions = torch.arange(max_img_size, dtype=torch.float32)
        freqs = torch.outer(positions, theta)
        cos_raw = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
        sin_raw = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
        self.register_buffer("rope_cos", cos_raw, persistent=False)
        self.register_buffer("rope_sin", sin_raw, persistent=False)

    def forward(self, x, coord_x, coord_y):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        cos_x = self.rope_cos[coord_x].unsqueeze(1).to(q.dtype)
        sin_x = self.rope_sin[coord_x].unsqueeze(1).to(q.dtype)
        cos_y = self.rope_cos[coord_y].unsqueeze(1).to(q.dtype)
        sin_y = self.rope_sin[coord_y].unsqueeze(1).to(q.dtype)

        q_x, q_y = q.split(self.half_dim, dim=-1)
        k_x, k_y = k.split(self.half_dim, dim=-1)
        q_x = q_x * cos_x + _rotate_half(q_x) * sin_x
        q_y = q_y * cos_y + _rotate_half(q_y) * sin_y
        k_x = k_x * cos_x + _rotate_half(k_x) * sin_x
        k_y = k_y * cos_y + _rotate_half(k_y) * sin_y
        q = torch.cat([q_x, q_y], dim=-1)
        k = torch.cat([k_x, k_y], dim=-1)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


# ============================================================================
# Section 4 — Transformer Block (Pre-LN + DeepNet residual scale)
# ============================================================================


class Block2D(nn.Module):
    def __init__(self, n_embd, n_head, max_img_size=64, dropout=0.0, weight_dtype=torch.float32):
        super().__init__()
        self.ln_1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention2D(n_embd, n_head, max_img_size, dropout, weight_dtype)
        self.ln_2 = RMSNorm(n_embd)
        self.mlp = MLP(n_embd, dropout, weight_dtype=weight_dtype)
        self.attn_alpha = nn.Parameter(torch.zeros(1))
        self.mlp_alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x, coord_x, coord_y):
        x = x + (1.0 + torch.tanh(self.attn_alpha)) * self.attn(self.ln_1(x), coord_x, coord_y)
        x = x + (1.0 + torch.tanh(self.mlp_alpha)) * self.mlp(self.ln_2(x))
        return x


# ============================================================================
# Section 5 — PixelImageGenerator (inference only)
# ============================================================================

START_TOKEN = 256


class PixelImageGenerator(nn.Module):
    """Autoregressive pixel-level image generator — regression variant.

    Predicts each pixel as a continuous [0, 1] value via sigmoid.
    """

    def __init__(self, n_embd=192, n_head=4, n_layer=3, max_img_size=32,
                 dropout=0.0, weight_dtype=torch.float32):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.max_img_size = max_img_size

        self.pixel_embed = nn.Embedding(257, n_embd)
        self.class_embed = nn.Embedding(10, n_embd)
        self.blocks = nn.ModuleList([
            Block2D(n_embd, n_head, max_img_size, dropout, weight_dtype)
            for _ in range(n_layer)
        ])
        self.ln_f = RMSNorm(n_embd)
        self.output = nn.Linear(n_embd, 1, bias=True)

    def forward(self, pixel_values, coords, class_ids=None):
        """Inference forward pass.

        Args:
            pixel_values: [B, T] — pixel tokens (0-255) + START_TOKEN.
            coords: [B, T, 2] — (y, x) coordinates.
            class_ids: [B] — digit label 0-9 for class conditioning.

        Returns:
            pred: [B, T] predicted pixel values in [0, 1].
        """
        x = self.pixel_embed(pixel_values.long())

        if class_ids is not None:
            cls_emb = self.class_embed(class_ids).unsqueeze(1)
            x = x + cls_emb

        coord_x = coords[..., 0]
        coord_y = coords[..., 1]

        for block in self.blocks:
            x = block(x, coord_x, coord_y)
        x = self.ln_f(x)
        pred = torch.sigmoid(self.output(x).squeeze(-1))
        return pred


# ============================================================================
# Section 6 — Data Helpers
# ============================================================================


def make_pixel_grid(h, w):
    """Return (h*w, 2) tensor of (y, x) coordinate pairs."""
    ys = torch.arange(h).unsqueeze(1).expand(-1, w).reshape(-1)
    xs = torch.arange(w).unsqueeze(0).expand(h, -1).reshape(-1)
    return torch.stack([ys, xs], dim=-1)


# ============================================================================
# Section 7 — Autoregressive Generation
# ============================================================================


@torch.no_grad()
def generate(model, img_size, temperature=0.3, device="cpu", class_id=0):
    """Generate one grayscale image by autoregressive pixel prediction."""
    model.eval()
    num_pixels = img_size * img_size
    grid = make_pixel_grid(img_size, img_size).to(device)

    if class_id >= 0:
        class_ids = torch.tensor([class_id], device=device)
    else:
        class_ids = torch.tensor([torch.randint(0, 10, (1,)).item()], device=device)

    generated = torch.full((1, num_pixels + 1), START_TOKEN, dtype=torch.long, device=device)

    for pos in range(1, num_pixels + 1):
        coords = grid[:pos].unsqueeze(0)
        preds = model(generated[:, :pos], coords, class_ids=class_ids)
        next_val = preds[:, -1]

        if temperature > 0:
            noise = torch.randn_like(next_val) * temperature * 0.1
            next_val = (next_val + noise).clamp(0, 1)

        pixel_token = (next_val * 255).round().clamp(0, 255).long()
        generated[:, pos] = pixel_token

    image = generated[0, 1:].cpu().numpy().astype(np.uint8).reshape(img_size, img_size)
    return image


def save_images(images, out_dir, prefix="gen"):
    """Save a list of 2D numpy arrays as individual PNGs and a grid."""
    os.makedirs(out_dir, exist_ok=True)
    for i, img in enumerate(images):
        pil = Image.fromarray(img, mode="L")
        path = os.path.join(out_dir, f"{prefix}_{i:02d}.png")
        pil.save(path)
        print(f"  saved {path}")
    grid_img = make_image_grid(images)
    grid_path = os.path.join(out_dir, f"{prefix}_grid.png")
    grid_img.save(grid_path)
    print(f"  saved grid {grid_path}")


def make_image_grid(images, cols=5):
    """Arrange a list of 2D arrays into a grid PIL Image."""
    n = len(images)
    rows = (n + cols - 1) // cols
    h, w = images[0].shape
    grid_arr = np.zeros((rows * h + rows - 1, cols * w + cols - 1), dtype=np.uint8)
    for i, img in enumerate(images):
        r, c = i // cols, i % cols
        grid_arr[r * (h + 1):r * (h + 1) + h, c * (w + 1):c * (w + 1) + w] = img
    return Image.fromarray(grid_arr, mode="L")


# ============================================================================
# Section 8 — Model Loading
# ============================================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(_SCRIPT_DIR, "weights")


def load_model(device="cpu", mode="ternary"):
    """Load model from packed checkpoint, return (model, step).

    Args:
        mode: "ternary" (2-bit, {-1,0,+1}) or "binary" (1-bit, {-1,+1}).
    """
    fname = "ckpt.pt" if mode == "ternary" else "ckpt_binary.pt"
    ckpt_path = os.path.join(WEIGHTS_DIR, fname)
    print(f"Loading checkpoint from {ckpt_path} ...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = unpack_state_dict(ckpt["model"], device=device)
    model = PixelImageGenerator(n_embd=192, n_head=4, n_layer=3, max_img_size=32)
    model.load_state_dict(sd)
    model.eval()
    model.to(device)
    return model, ckpt.get("step", "?")


# ============================================================================
# Section 9 — CLI and Main
# ============================================================================


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def main():
    ap = argparse.ArgumentParser(description="Pixel Image Generator — Inference Runner")
    ap.add_argument("--device", default="cpu", help="Device: cpu or cuda")
    ap.add_argument("--mode", choices=["ternary", "binary"], default="ternary",
                    help="Weight format: ternary ({-1,0,+1}, 2-bit) or binary ({-1,+1}, 1-bit)")
    ap.add_argument("--img-size", type=int, default=32, help="Image size in pixels (e.g. 32)")
    ap.add_argument("--num-samples", type=int, default=16, help="Number of images to generate")
    ap.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature (0=greedy)")
    ap.add_argument("--class-id", type=int, default=-1,
                    help="Digit class 0-9, or -1 for one per digit (cycles if num_samples > 10)")
    ap.add_argument("--out-dir", default=None,
                    help="Output directory for saved images (default: ./benchmark/samples)")
    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    # Load model
    model, step = load_model(device, mode=args.mode)
    total_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Checkpoint step: {step}")
    print(f"Device: {device}")
    print(f"Image size: {args.img_size}x{args.img_size}")
    print(f"Temperature: {args.temperature}")

    # Determine which digits to generate
    if args.class_id >= 0:
        class_ids_to_gen = [args.class_id] * args.num_samples
    else:
        num_digits = 10
        total = max(num_digits, args.num_samples)
        class_ids_to_gen = [i % num_digits for i in range(total)]

    # Generate images
    all_images = []
    for idx, cid in enumerate(class_ids_to_gen):
        cls_label = f"class_{cid}"
        if idx % 5 == 0:
            print(f"  generating image {idx + 1}/{len(class_ids_to_gen)} ...")
        img = generate(model, args.img_size, args.temperature, device, class_id=cid)
        all_images.append(img)

    # Save
    out_dir = args.out_dir or os.path.join(_SCRIPT_DIR, "benchmark", "samples")
    save_images(all_images, out_dir)
    print(f"Done! Generated {len(all_images)} image(s) in {out_dir}")


if __name__ == "__main__":
    main()
