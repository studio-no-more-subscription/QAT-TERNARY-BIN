#!/usr/bin/env python3
"""Inference runner for ternary-weight LLM.

Usage:
    python run.py                          # default prompt
    python run.py --prompt "Hello world"   # custom prompt
    python run.py --device cuda            # GPU inference
    python run.py --max-tokens 100         # shorter generation

Requires: torch, numpy, tiktoken
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import time
import zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ===========================================================================
# Packing Codec — unpack only
# ===========================================================================
# 2-bit ternary decoder: 2-bit code -> ternary value (0b11 -> 0 safe fallback)
_DECODE_LUT = np.array([-1, 0, 1, 0], dtype=np.int8)

_PACKED_KEY = "_packed_ternary"
_PACKED_BINARY_KEY = "_packed_binary"


def unpack_ternary_values(
    data: bytes,
    shape,
    n_total: int,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
) -> Tensor:
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


def unpack_ternary(packed: dict, device: str | torch.device = "cpu") -> Tensor:
    """Reconstruct a ternary tensor from a packed dict."""
    return unpack_ternary_values(
        packed["data"], packed["shape"], packed["n_total"], device=device
    )


def is_packed(value) -> bool:
    """True if *value* is a packed-ternary dict."""
    return isinstance(value, dict) and value.get(_PACKED_KEY) is True


def unpack_binary_values(
    data: bytes,
    shape,
    n_total: int,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
) -> Tensor:
    """Decode packed 1-bit bytes back to a binary {-1, +1} tensor of *shape*."""
    if n_total == 0:
        return torch.empty(shape, dtype=dtype, device=device)
    arr = np.frombuffer(data, dtype=np.uint8)
    bits = np.unpackbits(arr, bitorder="little")
    vals = bits.astype(np.int8) * 2 - 1  # 0->-1, 1->+1
    vals = vals[:n_total]
    t = torch.from_numpy(vals.copy()).to(dtype).reshape(shape)
    return t.to(device)


def unpack_binary(packed: dict, device: str | torch.device = "cpu") -> Tensor:
    """Reconstruct a binary tensor from a 1-bit packed dict."""
    return unpack_binary_values(
        packed["data"], packed["shape"], packed["n_total"], device=device
    )


def is_packed_binary(value) -> bool:
    """True if *value* is a packed-binary (1-bit) dict."""
    return isinstance(value, dict) and value.get(_PACKED_BINARY_KEY) is True


def unpack_state_dict(
    state_dict: dict, device: str | torch.device = "cpu"
) -> dict:
    """Return a copy of *state_dict* with packed-ternary/-binary entries decoded.

    Non-packed entries pass through unchanged.
    """
    unpacked: dict = {}
    for key, val in state_dict.items():
        if is_packed(val):
            unpacked[key] = unpack_ternary(val, device=device)
        elif is_packed_binary(val):
            unpacked[key] = unpack_binary(val, device=device)
        else:
            unpacked[key] = val
    return unpacked


# ===========================================================================
# Model Architecture
# ===========================================================================


class TernaryLinear(nn.Module):
    """Linear layer with ternary weights in {-1, 0, +1} and per-channel scale.

    Forward: y = scale * F.linear(x, weight, bias) / sqrt(fan_in)
    The fan-in division keeps activation magnitudes stable across layer sizes.
    """

    is_ternary = True

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 weight_dtype: torch.dtype = torch.float32):
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

    def forward(self, x: Tensor) -> Tensor:
        y = F.linear(x, self.weight.to(x.dtype), self.bias)
        y = y * self.scale
        return y * self.inv_sqrt_fan_in


# ---------------------------------------------------------------------------
# RoPE (Rotary Position Embedding)
# ---------------------------------------------------------------------------

def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _build_rope_cache(block_size: int, head_dim: int, device, dtype=torch.float32) -> tuple[Tensor, Tensor]:
    """Precompute RoPE cos/sin tables of shape [1, 1, block_size, head_dim]."""
    theta = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    positions = torch.arange(block_size, device=device, dtype=dtype)
    freqs = torch.outer(positions, theta)
    cos = freqs.cos().unsqueeze(0).unsqueeze(0)
    sin = freqs.sin().unsqueeze(0).unsqueeze(0)
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    return cos, sin


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


# ---------------------------------------------------------------------------
# Causal Self-Attention
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.0,
                 weight_dtype: torch.dtype = torch.float32):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.c_attn = TernaryLinear(n_embd, 3 * n_embd, bias=False, weight_dtype=weight_dtype)
        self.c_proj = TernaryLinear(n_embd, n_embd, bias=False, weight_dtype=weight_dtype)
        self.attn_dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.resid_dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        cos, sin = _build_rope_cache(block_size, self.head_dim, device="cpu", dtype=torch.float32)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        cos = self.cos_cached[:, :, :T, :].to(q.dtype)
        sin = self.sin_cached[:, :, :T, :].to(q.dtype)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float = 0.0,
                 weight_dtype: torch.dtype = torch.float32):
        super().__init__()
        self.c_fc = TernaryLinear(n_embd, 4 * n_embd, bias=False, weight_dtype=weight_dtype)
        self.c_proj = TernaryLinear(4 * n_embd, n_embd, bias=False, weight_dtype=weight_dtype)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.c_fc(x)
        x = F.gelu(x, approximate="tanh")
        x = self.dropout(x)
        x = self.c_proj(x)
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.0,
                 weight_dtype: torch.dtype = torch.float32):
        super().__init__()
        self.ln_1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout, weight_dtype=weight_dtype)
        self.ln_2 = RMSNorm(n_embd)
        self.mlp = MLP(n_embd, dropout, weight_dtype=weight_dtype)
        self.attn_alpha = nn.Parameter(torch.zeros(1))
        self.mlp_alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor) -> Tensor:
        x = x + (1.0 + torch.tanh(self.attn_alpha)) * self.attn(self.ln_1(x))
        x = x + (1.0 + torch.tanh(self.mlp_alpha)) * self.mlp(self.ln_2(x))
        return x


# ---------------------------------------------------------------------------
# Ternary LLM Model (inference only)
# ---------------------------------------------------------------------------

class TernaryLLM(nn.Module):
    def __init__(
        self,
        n_layer: int = 4,
        n_head: int = 4,
        n_embd: int = 256,
        block_size: int = 512,
        vocab_size: int = 50257,
        dropout: float = 0.0,
        tie_weights: bool = True,
        weight_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.block_size = block_size
        self.vocab_size = vocab_size

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(vocab_size, n_embd),
            drop=nn.Dropout(dropout) if dropout else nn.Identity(),
            h=nn.ModuleList([Block(n_embd, n_head, block_size, dropout, weight_dtype=weight_dtype)
                             for _ in range(n_layer)]),
            ln_f=RMSNorm(n_embd),
        ))
        self.tie_weights = tie_weights
        self.lm_head = TernaryLinear(n_embd, vocab_size, bias=False, weight_dtype=weight_dtype)
        if tie_weights:
            self.lm_head.weight = self.transformer.wte.weight
            self.lm_head.is_ternary = False

    def forward(self, idx: Tensor) -> tuple[Tensor, None]:
        """Run inference. Returns (logits, None)."""
        B, T = idx.shape
        assert T <= self.block_size

        x = self.transformer.wte(idx)
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x[:, -1:])  # last position only
        return logits, None

    @torch.no_grad()
    def generate(
        self, idx: Tensor, max_new_tokens: int,
        temperature: float = 1.0, top_p: float = 0.9,
    ) -> Tensor:
        """Top-p (nucleus) sampling with NaN guard."""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.shape[1] <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :].float() / max(temperature, 1e-8)
            logits = torch.nan_to_num(logits, nan=-float("inf"), posinf=-float("inf"), neginf=-float("inf"))
            if top_p < 1.0:
                sorted_logits, sorted_idx = logits.sort(descending=True)
                cum_probs = sorted_logits.softmax(-1).cumsum(-1)
                sorted_idx_to_remove = cum_probs > top_p
                sorted_idx_to_remove[..., 1:] = sorted_idx_to_remove[..., :-1].clone()
                sorted_idx_to_remove[..., 0] = False
                idx_to_remove = sorted_idx_to_remove.scatter(-1, sorted_idx, sorted_idx_to_remove)
                logits[idx_to_remove] = -float("inf")
            probs = logits.softmax(-1)
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            if probs.sum() < 1e-8:
                probs = torch.ones_like(probs) / probs.numel()
            idx_next = probs.multinomial(1)
            idx = torch.cat([idx, idx_next], dim=-1)
        return idx


# ===========================================================================
# Weight Loading
# ===========================================================================

def load_model(device="cpu", mode="ternary"):
    """Load the pretrained model from the packaged zip checkpoint.

    Args:
        mode: "ternary" (2-bit, {-1,0,+1}) or "binary" (1-bit, {-1,+1}).

    The zip contains:
      - ckpt.pt: torch checkpoint with state dict (packed weights + fp32 tensors)
      - embedding_fp16.npy: tied embedding / LM head weights

    Returns (model, step_number).
    """
    zip_name = "ckpt.pt.zip" if mode == "ternary" else "ckpt_binary.pt.zip"
    weights_zip = os.path.join(os.path.dirname(__file__), "weights", zip_name)

    tmpdir = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(weights_zip, "r") as z:
            z.extractall(tmpdir)

        ckpt = torch.load(os.path.join(tmpdir, "ckpt.pt"), map_location=device, weights_only=False)
        sd = ckpt["model"]

        emb_np = np.load(os.path.join(tmpdir, "embedding_fp16.npy"))
        emb_tensor = torch.from_numpy(emb_np).float().to(device)

        # Restore tied embedding weights (removed from checkpoint to save space)
        sd["transformer.wte.weight"] = emb_tensor
        sd["lm_head.weight"] = emb_tensor

        # Unpack ternary / binary compressed weights
        sd = unpack_state_dict(sd, device=device)

        model = TernaryLLM(
            n_layer=12, n_head=8, n_embd=512,
            block_size=512, vocab_size=50257, tie_weights=True,
        )
        model.load_state_dict(sd)
        model.eval()
        model.to(device)

        return model, ckpt.get("step", "?")
    finally:
        shutil.rmtree(tmpdir)


# ===========================================================================
# Inference & Benchmark
# ===========================================================================

def count_params(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters())


def generate_text(model, enc, prompt: str, max_tokens: int, temperature: float, top_p: float, device: str):
    """Generate text from a prompt string."""
    tokens = enc.encode_ordinary(prompt)
    idx = torch.tensor([tokens], dtype=torch.long, device=device)
    generated = model.generate(idx, max_new_tokens=max_tokens, temperature=temperature, top_p=top_p)
    output_ids = generated[0].cpu().tolist()
    return enc.decode(output_ids)


def main():
    parser = argparse.ArgumentParser(description="Inference runner for ternary LLM")
    parser.add_argument("--device", default="cpu", help="Device (cpu, cuda, etc.)")
    parser.add_argument("--mode", choices=["ternary", "binary"], default="ternary",
                        help="Weight format: ternary ({-1,0,+1}, 2-bit) or binary ({-1,+1}, 1-bit)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling threshold")
    parser.add_argument("--max-tokens", type=int, default=200, help="Max tokens to generate")
    parser.add_argument("--prompt", default="Once upon a time,", help="Generation prompt")
    args = parser.parse_args()

    print(f"Loading {args.mode} model on {args.device}...")
    t0 = time.time()
    model, step = load_model(args.device, mode=args.mode)
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.2f}s (step {step})")

    total = count_params(model)
    print(f"Total parameters: {total:,}")
    print()

    # Tokenizer
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")

    # =======================================================================
    # Main prompt
    # =======================================================================
    prompt = args.prompt
    print(f"{'=' * 60}")
    print(f"Prompt: {prompt!r}")
    print(f"{'=' * 60}")
    t0 = time.time()
    text = generate_text(model, enc, prompt, args.max_tokens, args.temperature, args.top_p, args.device)
    gen_time = time.time() - t0
    print(text)
    print()
    print(f"[{args.max_tokens} tokens in {gen_time:.2f}s ({args.max_tokens / gen_time:.1f} tok/s)]")
    print()

    # =======================================================================
    # Benchmark: a few additional prompts
    # =======================================================================
    sample_prompts = [
        "The future of artificial intelligence is",
        "In the beginning,",
        "The meaning of life is",
    ]

    for p in sample_prompts:
        t0 = time.time()
        t = generate_text(model, enc, p, 100, args.temperature, args.top_p, args.device)
        elapsed = time.time() - t0
        print(f"{'─' * 50}")
        print(f"Prompt: {p!r}  ({100 / elapsed:.1f} tok/s)")
        print(f"{'─' * 50}")
        print(t)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
