#!/usr/bin/env python3
"""
Self-contained inference runner for MNIST CNN models (ternary + binary).

This file is self-contained — no external project imports.
Weight-packing codec (unpack only), custom layer classes, and model
definitions are inlined.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")

# ═══════════════════════════════════════════════════════════════════════
# 2-bit ternary unpack codec (unpack only — no pack functions)
# ═══════════════════════════════════════════════════════════════════════

_DECODE_LUT = np.array([-1, 0, 1, 0], dtype=np.int8)
_PACKED_KEY = "_packed_ternary"
_PACKED_BINARY_KEY = "_packed_binary"


def unpack_ternary_values(data, shape, n_total, dtype=torch.float32, device="cpu"):
    """Decode packed 2-bit bytes back to a ternary tensor of *shape*."""
    if n_total == 0:
        return torch.empty(shape, dtype=dtype, device=device)
    arr = np.frombuffer(data, dtype=np.uint8)
    # Extract 4 codes per byte
    codes = np.empty(arr.shape[0] * 4, dtype=np.uint8)
    codes[0::4] = arr & 0b11
    codes[1::4] = (arr >> 2) & 0b11
    codes[2::4] = (arr >> 4) & 0b11
    codes[3::4] = (arr >> 6) & 0b11
    # Map 2-bit codes to ternary values via LUT (0b11 -> 0)
    vals = _DECODE_LUT[codes]  # int8, values in {-1,0,1}
    # Trim padding
    vals = vals[:n_total]
    t = torch.from_numpy(vals.copy()).to(dtype).reshape(shape)
    return t.to(device)


def unpack_ternary(packed, device="cpu"):
    """Reconstruct a ternary tensor from a packed dict."""
    return unpack_ternary_values(
        packed["data"], packed["shape"], packed["n_total"], device=device
    )


def is_packed(value):
    """True if *value* is a packed-ternary dict."""
    return isinstance(value, dict) and value.get(_PACKED_KEY) is True


def unpack_binary_values(data, shape, n_total, dtype=torch.float32, device="cpu"):
    """Decode packed 1-bit bytes back to a binary {-1, +1} tensor of *shape*."""
    if n_total == 0:
        return torch.empty(shape, dtype=dtype, device=device)
    arr = np.frombuffer(data, dtype=np.uint8)
    # Extract 8 bits per byte (little-endian bit order to match pack)
    bits = np.unpackbits(arr, bitorder="little")
    # Map {0, 1} -> {-1, +1}
    vals = (bits.astype(np.int8) * 2 - 1)  # 0->-1, 1->+1
    # Trim padding
    vals = vals[:n_total]
    t = torch.from_numpy(vals.copy()).to(dtype).reshape(shape)
    return t.to(device)


def unpack_binary(packed, device="cpu"):
    """Reconstruct a binary tensor from a 1-bit packed dict."""
    return unpack_binary_values(
        packed["data"], packed["shape"], packed["n_total"], device=device
    )


def is_packed_binary(value):
    """True if *value* is a packed-binary (1-bit) dict."""
    return isinstance(value, dict) and value.get(_PACKED_BINARY_KEY) is True


def unpack_state_dict(state_dict, device="cpu"):
    """Return a copy of *state_dict* with packed-ternary/binary entries decoded.

    Non-packed entries pass through unchanged, so calling this on a legacy
    fp32 checkpoint is a no-op (backward compatible).
    """
    unpacked = {}
    for key, val in state_dict.items():
        if is_packed(val):
            unpacked[key] = unpack_ternary(val, device=device)
        elif is_packed_binary(val):
            unpacked[key] = unpack_binary(val, device=device)
        else:
            unpacked[key] = val
    return unpacked


# ═══════════════════════════════════════════════════════════════════════
# Ternary layers (inference-only, no training internals)
# ═══════════════════════════════════════════════════════════════════════


class TernaryConv2d(nn.Module):
    """2D convolution with ternary weights {-1, 0, 1} + per-channel scale."""

    is_ternary = True

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = False,
        use_scale: bool = True,
        apply_scale_in_forward: bool = True,
        weight_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups

        # Placeholder init — weights are overwritten by load_state_dict
        w = torch.zeros(out_channels, in_channels // groups, *kernel_size)
        self.weight = nn.Parameter(w.to(weight_dtype), requires_grad=True)
        self.use_scale = use_scale
        self.apply_scale_in_forward = apply_scale_in_forward
        self.scale = nn.Parameter(torch.ones(out_channels), requires_grad=use_scale)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=True)
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        y = F.conv2d(
            x,
            self.weight.to(x.dtype),
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        if self.apply_scale_in_forward and self.use_scale:
            y = y * self.scale.view(1, -1, 1, 1)
        return y


class TernaryBatchNorm2d(nn.Module):
    """BatchNorm2d with ternary gamma and fp32 beta (eval-only forward)."""

    is_ternary = True

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        weight_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if affine:
            # ternary gamma, init +1 (identity scale)
            gamma = torch.ones(num_features, dtype=weight_dtype)
            self.weight = nn.Parameter(gamma, requires_grad=True)
            # fp32 beta
            self.bias = nn.Parameter(torch.zeros(num_features), requires_grad=True)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_features))
            self.register_buffer("running_var", torch.ones(num_features))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

    def forward(self, x):
        mean = self.running_mean
        inv_std = 1.0 / torch.sqrt(self.running_var + self.eps)
        x_norm = (x - mean.view(1, -1, 1, 1)) * inv_std.view(1, -1, 1, 1)
        if self.affine:
            # gamma is ternary {-1,0,1}; apply as scale
            x_norm = x_norm * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return x_norm


# ═══════════════════════════════════════════════════════════════════════
# Model definitions
# ═══════════════════════════════════════════════════════════════════════


class SmallCNN(nn.Module):
    """Small CNN for MNIST: 3 ternary conv blocks, GAP, fp32 FC.

    Conv(1->32, k3, s1) -> TBN -> ReLU
    Conv(32->64, k3, s2) -> TBN -> ReLU
    Conv(64->128, k3, s2) -> TBN -> ReLU
    GAP -> FP32 Linear(128 -> 10)
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        channels: tuple[int, int, int] = (32, 64, 128),
        weight_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        c1, c2, c3 = channels

        self.conv1 = TernaryConv2d(
            in_channels, c1, kernel_size=3, stride=1, padding=1,
            apply_scale_in_forward=False, weight_dtype=weight_dtype,
        )
        self.bn1 = TernaryBatchNorm2d(c1, weight_dtype=weight_dtype)

        self.conv2 = TernaryConv2d(
            c1, c2, kernel_size=3, stride=2, padding=1,
            apply_scale_in_forward=False, weight_dtype=weight_dtype,
        )
        self.bn2 = TernaryBatchNorm2d(c2, weight_dtype=weight_dtype)

        self.conv3 = TernaryConv2d(
            c2, c3, kernel_size=3, stride=2, padding=1,
            apply_scale_in_forward=False, weight_dtype=weight_dtype,
        )
        self.bn3 = TernaryBatchNorm2d(c3, weight_dtype=weight_dtype)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c3, num_classes, bias=True)  # fp32

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x) * self.conv1.scale.view(1, -1, 1, 1))
        x = self.conv2(x)
        x = F.relu(self.bn2(x) * self.conv2.scale.view(1, -1, 1, 1))
        x = self.conv3(x)
        x = F.relu(self.bn3(x) * self.conv3.scale.view(1, -1, 1, 1))
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return x


# ═══════════════════════════════════════════════════════════════════════
# Weight loading
# ═══════════════════════════════════════════════════════════════════════
# Weight loading
# ═══════════════════════════════════════════════════════════════════════


def load_ternary_model(device="cpu", mode="ternary"):
    """Load the ternary/binary SmallCNN from packed checkpoint, unpacking on the fly.

    Args:
        mode: "ternary" (2-bit, {-1,0,+1}) or "binary" (1-bit, {-1,+1}).
    """
    fname = "ternary.pt" if mode == "ternary" else "binary.pt"
    path = os.path.join(WEIGHTS_DIR, fname)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = unpack_state_dict(ckpt["model"], device=device)
    model = SmallCNN(in_channels=1, num_classes=10, channels=(32, 64, 128))
    model.load_state_dict(sd)
    model.eval()
    model.to(device)
    return model, ckpt.get("test_acc", "?")


# ═══════════════════════════════════════════════════════════════════════
# Evaluation helpers
# ═══════════════════════════════════════════════════════════════════════


@torch.no_grad()
def evaluate(model, loader, device):
    """Compute classification accuracy over a DataLoader."""
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    return correct / total


def count_parameters(model):
    """Return total number of learnable parameters."""
    return sum(p.numel() for p in model.parameters())


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="MNIST CNN inference benchmark")
    parser.add_argument("--device", default="cpu", help="device (cpu or cuda)")
    parser.add_argument("--batch-size", type=int, default=256, help="batch size")
    parser.add_argument("--data-dir", default="./data", help="MNIST data directory")
    parser.add_argument("--mode", choices=["ternary", "binary"], default="ternary",
                        help="Weight format: ternary ({-1,0,+1}, 2-bit) or binary ({-1,+1}, 1-bit)")
    args = parser.parse_args()

    device = args.device

    # Load test dataset
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_ds = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Load model
    quant_model, quant_ckpt_acc = load_ternary_model(device, mode=args.mode)

    # Count parameters
    n_quant = count_parameters(quant_model)

    # Evaluate
    quant_acc = evaluate(quant_model, test_loader, device)

    print(f"{args.mode.capitalize()} model (checkpoint test_acc: {quant_ckpt_acc}):")
    print(f"  Parameters: {n_quant:,}")
    print(f"  Test accuracy: {quant_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
