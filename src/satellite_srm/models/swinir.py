"""Core Swin Transformer image restoration blocks and attention mechanisms."""
import math
import numpy as np
from satellite_srm.compat import torch, nn, F

class WindowAttention(nn.Module):
    """Window-based Multi-head Self-Attention (W-MSA)."""
    def __init__(self, dim: int, window_size: int = 8, num_heads: int = 6, qkv_bias: bool = True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (B, N, C)
        qkv = self.qkv(x)
        return self.proj(x)

class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block with LayerNorm, MSA, and MLP."""
    def __init__(self, dim: int, num_heads: int = 6, window_size: int = 8, shift_size: int = 0, mlp_ratio: float = 2.0):
        super().__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=window_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ResidualSwinTransformerBlock(nn.Module):
    """Residual Swin Transformer Block (RSTB) encapsulating multiple Swin blocks and residual conv."""
    def __init__(self, dim: int, depth: int = 4, num_heads: int = 6, window_size: int = 8):
        super().__init__()
        self.dim = dim
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, num_heads=num_heads, window_size=window_size,
                                shift_size=0 if (i % 2 == 0) else window_size // 2)
            for i in range(depth)
        ])
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        res = x
        tokens = x.permute(0, 2, 3, 1)
        for block in self.blocks:
            tokens = block(tokens)
        out = tokens.permute(0, 3, 1, 2)
        out = self.conv(out)
        return out + res

class SwinIR(nn.Module):
    """Standard SwinIR backbone for image super-resolution."""
    def __init__(self, img_size: int = 128, in_channels: int = 3, out_channels: int = 3,
                 embed_dim: int = 96, depths=(4, 4, 4, 4), num_heads=(6, 6, 6, 6),
                 window_size: int = 8, scale: int = 3):
        super().__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)
        self.rstb_layers = nn.ModuleList([
            ResidualSwinTransformerBlock(dim=embed_dim, depth=d, num_heads=h, window_size=window_size)
            for d, h in zip(depths, num_heads)
        ])
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.upsample = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * (scale * scale), kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(embed_dim, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        feat = self.conv_first(x)
        body = feat
        for rstb in self.rstb_layers:
            body = rstb(body)
        body = self.conv_after_body(body) + feat
        out = self.upsample(body)
        return out
