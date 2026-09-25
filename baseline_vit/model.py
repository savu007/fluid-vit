"""Plain global-attention 3D ViT; predicts one normalized velocity increment.

This is a baseline inspired by residual forecasting, not a SwinV2 reproduction.
Attention operates on spatial tokens from a single observed frame.
"""

import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    """Keep the attention kernel separate for future implementation experiments."""

    def __init__(self, width, heads, dropout):
        super().__init__()
        self.heads = heads
        self.dropout = dropout
        self.qkv = nn.Linear(width, 3 * width)
        self.projection = nn.Linear(width, width)

    def forward(self, tokens):
        batch, length, width = tokens.shape
        qkv = self.qkv(tokens).reshape(batch, length, 3, self.heads, width // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=False
        )
        return self.projection(attended.transpose(1, 2).reshape(batch, length, width))


class Block(nn.Module):
    def __init__(self, width, heads, mlp_ratio, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = Attention(width, heads, dropout)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, width * mlp_ratio), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(width * mlp_ratio, width), nn.Dropout(dropout),
        )

    def forward(self, tokens):
        tokens = tokens + self.attention(self.norm1(tokens))
        return tokens + self.mlp(self.norm2(tokens))


class BaselineViT(nn.Module):
    def __init__(self, grid_size=64, in_channels=3, patch_size=8, embed_dim=192,
                 depth=6, num_heads=6, mlp_ratio=4, dropout=0.0):
        super().__init__()
        if grid_size < 1 or patch_size < 1 or grid_size % patch_size:
            raise ValueError("Positive grid_size must be divisible by positive patch_size")
        if num_heads < 1 or embed_dim < 1 or embed_dim % num_heads:
            raise ValueError("Positive embed_dim must be divisible by positive num_heads")
        if in_channels < 1 or depth < 1 or mlp_ratio < 1 or not 0 <= dropout < 1:
            raise ValueError("Invalid model dimensions or dropout")
        self.grid_size = grid_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.token_side = grid_size // patch_size
        self.patch_embed = nn.Conv3d(in_channels, embed_dim, patch_size, stride=patch_size)
        self.position = nn.Parameter(torch.zeros(1, self.token_side ** 3, embed_dim))
        nn.init.normal_(self.position, std=0.02)
        self.time_embed = nn.Sequential(nn.Linear(1, embed_dim), nn.LayerNorm(embed_dim))
        self.blocks = nn.Sequential(*[
            Block(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, in_channels * patch_size ** 3)
        # Begin at the persistence predictor, then learn departures from it.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def unpatchify(self, patches):
        """Inverse layout: tokens in x,y,z order; patch values in C,px,py,pz order."""
        batch = patches.shape[0]
        n, p, c = self.token_side, self.patch_size, self.in_channels
        blocks = patches.reshape(batch, n, n, n, c, p, p, p)
        return blocks.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(
            batch, c, self.grid_size, self.grid_size, self.grid_size
        )

    def forward(self, x, delta_t):
        """x: normalized [B,C,X,Y,Z]; delta_t: [B], divided by train reference dt."""
        if x.ndim != 5 or tuple(x.shape[1:]) != (self.in_channels,) + (self.grid_size,) * 3:
            raise ValueError("Input must have shape [B, in_channels, grid_size, grid_size, grid_size]")
        if delta_t.shape not in ((x.shape[0],), (x.shape[0], 1)):
            raise ValueError("delta_t must contain one interval per batch element")
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        time = self.time_embed(delta_t.reshape(-1, 1).to(device=x.device, dtype=x.dtype))
        tokens = tokens + self.position + time[:, None, :]
        delta = self.unpatchify(self.head(self.norm(self.blocks(tokens))))
        return x + delta
