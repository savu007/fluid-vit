"""Yalamanchi-inspired 3D residual SwinV2-UNet.

Original local implementation, not an exact author-code reproduction.
SwinV2 semantic reference: microsoft/Swin-Transformer/models/swin_transformer_v2.py.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F


def space_to_depth(x, factor):
    """Group children in local-x, local-y, local-z, channel order."""
    b, nx, ny, nz, c = x.shape
    if factor < 1 or any(n % factor for n in (nx, ny, nz)):
        raise ValueError("Spatial dimensions must be divisible by factor")
    return (x.reshape(b, nx // factor, factor, ny // factor, factor,
                      nz // factor, factor, c)
            .permute(0, 1, 3, 5, 2, 4, 6, 7)
            .reshape(b, nx // factor, ny // factor, nz // factor, c * factor**3))

def depth_to_space(x, factor):
    b, nx, ny, nz, packed = x.shape
    if factor < 1 or packed % factor**3:
        raise ValueError("Feature dimension must be divisible by factor cubed")
    c = packed // factor**3
    return (x.reshape(b, nx, ny, nz, factor, factor, factor, c)
            .permute(0, 1, 4, 2, 5, 3, 6, 7)
            .reshape(b, nx * factor, ny * factor, nz * factor, c))

def partition(x, window):
    b, nx, ny, nz, c = x.shape
    packed = space_to_depth(x, window)
    return packed.reshape(b * (nx // window) * (ny // window) * (nz // window),
                          window**3, c)

def reverse(windows, window, side):
    if side % window:
        raise ValueError("Grid must be divisible by window")
    n = side // window
    if windows.shape[0] % n**3 or windows.shape[1] != window**3:
        raise ValueError("Invalid window shape")
    b = windows.shape[0] // n**3
    return depth_to_space(windows.reshape(b, n, n, n, -1), window)

def shifted_mask(side, window, shift):
    """Finite-domain reference mask. Periodic attention must NOT use this mask."""
    labels = torch.zeros(1, side, side, side, 1)
    if not shift:
        return torch.zeros((side // window)**3, window**3, window**3)
    spans = (slice(0, -window), slice(-window, -shift), slice(-shift, None))
    number = 0
    for a in spans:
        for b in spans:
            for c in spans:
                labels[:, a, b, c, :] = number
                number += 1
    groups = partition(labels, window).squeeze(-1)
    different = groups[:, :, None] != groups[:, None, :]
    return torch.zeros_like(different, dtype=torch.float32).masked_fill(different, -torch.inf)

class WindowAttention3D(nn.Module):
    def __init__(self, width, heads, window, *, periodic_full=False, dropout=0.0):
        super().__init__()
        if width % heads:
            raise ValueError("Width must be divisible by heads")
        self.width, self.heads, self.window = width, heads, window
        self.dropout = dropout
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(width))
        self.v_bias = nn.Parameter(torch.zeros(width))
        self.logit_scale = nn.Parameter(torch.full((heads, 1, 1), math.log(10.0)))
        self.position_mlp = nn.Sequential(nn.Linear(3, 512), nn.ReLU(), nn.Linear(512, heads, bias=False))
        self.projection = nn.Linear(width, width)
        self.output_dropout = nn.Dropout(dropout)

        points = torch.stack(torch.meshgrid(*[torch.arange(window)] * 3, indexing="ij"), -1)
        points = points.reshape(-1, 3)
        offsets = points[:, None, :] - points[None, :, :]
        if periodic_full:
            # Minimum image on the full-domain bottleneck; even ties use -side/2.
            offsets = (offsets + window // 2) % window - window // 2
        unique, inverse = torch.unique(offsets.reshape(-1, 3), dim=0, return_inverse=True)
        coordinates = unique.float() / max(window - 1, 1) * 8
        coordinates = coordinates.sign() * torch.log2(coordinates.abs() + 1) / math.log2(8)
        self.register_buffer("relative_coordinates", coordinates)
        self.register_buffer("relative_index", inverse.reshape(window**3, window**3))

    def relative_bias(self):
        table = 16 * torch.sigmoid(self.position_mlp(self.relative_coordinates))
        return table[self.relative_index].permute(2, 0, 1)

    def forward(self, x, mask=None):
        batches, length, width = x.shape
        bias = torch.cat((self.q_bias, torch.zeros_like(self.q_bias), self.v_bias))
        qkv = F.linear(x, self.qkv.weight, bias).reshape(batches, length, 3, self.heads, width // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        scores = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-1, -2)
        scores = scores * self.logit_scale.clamp(max=math.log(100.0)).exp()
        scores = scores + self.relative_bias().unsqueeze(0)
        if mask is not None:
            windows = mask.shape[0]
            scores = scores.reshape(-1, windows, self.heads, length, length)
            scores = scores + mask[None, :, None, :, :]
            scores = scores.reshape(batches, self.heads, length, length)
        weights = F.dropout(scores.softmax(-1), p=self.dropout, training=self.training)
        mixed = (weights @ v).transpose(1, 2).reshape(batches, length, width)
        return self.output_dropout(self.projection(mixed))

class SwinBlock3D(nn.Module):
    def __init__(self, width, heads, side, window, shifted, mlp_ratio, dropout, boundary):
        super().__init__()
        self.side = side
        self.window = min(window, side)
        if side % self.window:
            raise ValueError("Each token grid must be divisible by its effective window")
        self.shift = self.window // 2 if shifted and side > self.window else 0
        self.attention = WindowAttention3D(
            width, heads, self.window, periodic_full=boundary == "periodic" and side == self.window,
            dropout=dropout)
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, width * mlp_ratio), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(width * mlp_ratio, width), nn.Dropout(dropout))
        mask = shifted_mask(side, self.window, self.shift) if boundary == "finite" and self.shift else None
        self.register_buffer("mask", mask)

    def forward(self, x):
        moved = torch.roll(x, (-self.shift,) * 3, (1, 2, 3)) if self.shift else x
        attended = self.attention(partition(moved, self.window), self.mask)
        attended = reverse(attended, self.window, self.side)
        if self.shift:
            attended = torch.roll(attended, (self.shift,) * 3, (1, 2, 3))
        # SwinV2 residual POST-normalization (not the old ViT's pre-norm).
        x = x + self.norm1(attended)
        return x + self.norm2(self.mlp(x))

class Merge3D(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.reduction = nn.Linear(8 * width, 2 * width, bias=False)
        self.norm = nn.LayerNorm(2 * width)

    def forward(self, x):
        return self.norm(self.reduction(space_to_depth(x, 2)))

class Expand3D(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.expansion = nn.Linear(width, 8 * (width // 2), bias=False)
        self.norm = nn.LayerNorm(width // 2)

    def forward(self, x):
        return self.norm(depth_to_space(self.expansion(x), 2))

class ConvSkip3D(nn.Module):
    def __init__(self, width, boundary):
        super().__init__()
        self.depthwise = nn.Conv3d(width, width, 3, padding=1, groups=width,
                                   padding_mode="circular" if boundary == "periodic" else "zeros")
        self.norm = nn.LayerNorm(width)
        self.pointwise = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(),
                                       nn.Linear(4 * width, width))
        self.scale = nn.Parameter(torch.full((width,), 1e-6))

    def forward(self, x):
        local = self.depthwise(x.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        return x + self.scale * self.pointwise(self.norm(local))

ARCHITECTURE = "yalamanchi_swinv2_3d_v1"

def validate_model(grid_size, in_channels, patch_size, embed_dim, depths, num_heads,
                   window_size, mlp_ratio, dropout, boundary):
    for key, value in dict(grid_size=grid_size, in_channels=in_channels, patch_size=patch_size,
                           embed_dim=embed_dim, window_size=window_size, mlp_ratio=mlp_ratio).items():
        if type(value) is not int or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if grid_size % (4 * patch_size):
        raise ValueError("grid_size must be divisible by four times patch_size (two merges)")
    if not isinstance(depths, (list, tuple)) or len(depths) != 5 or any(type(v) is not int or v < 1 for v in depths):
        raise ValueError("depths must contain five positive integers")
    if not isinstance(num_heads, (list, tuple)) or len(num_heads) != 3:
        raise ValueError("num_heads must contain three positive integers")
    for level, heads in enumerate(num_heads):
        if type(heads) is not int or heads < 1 or (embed_dim * 2**level) % heads:
            raise ValueError("Each stage width must be divisible by its positive head count")
        side = grid_size // patch_size // 2**level
        if side % min(side, window_size):
            raise ValueError("Stage grid must be divisible by effective window size")
    if type(dropout) not in (int, float) or not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError("dropout must be finite and in [0, 1)")
    if boundary not in ("periodic", "finite"):
        raise ValueError("boundary must be periodic or finite")

class YalamanchiViT(nn.Module):
    def __init__(self, grid_size=64, in_channels=3, patch_size=4, embed_dim=96,
                 depths=(2, 2, 2, 2, 2), num_heads=(3, 6, 12), window_size=4,
                 mlp_ratio=4, dropout=0.0, boundary="periodic"):
        super().__init__()
        validate_model(grid_size, in_channels, patch_size, embed_dim, depths, num_heads,
                       window_size, mlp_ratio, dropout, boundary)
        self.grid_size, self.in_channels, self.patch_size = grid_size, in_channels, patch_size
        self.patch_embed = nn.Conv3d(in_channels, embed_dim, patch_size, stride=patch_size)
        self.patch_norm = nn.LayerNorm(embed_dim)
        self.time_embed = nn.Sequential(nn.Linear(1, embed_dim), nn.LayerNorm(embed_dim))
        n = grid_size // patch_size

        def stage(level, count):
            return nn.Sequential(*[
                SwinBlock3D(embed_dim * 2**level, num_heads[level], n // 2**level,
                            window_size, i % 2 == 1, mlp_ratio, dropout, boundary)
                for i in range(count)
            ])

        self.encoder0, self.encoder1 = stage(0, depths[0]), stage(1, depths[1])
        self.skip0, self.skip1 = ConvSkip3D(embed_dim, boundary), ConvSkip3D(2 * embed_dim, boundary)
        self.merge0, self.merge1 = Merge3D(embed_dim), Merge3D(2 * embed_dim)
        self.bottleneck = stage(2, depths[2])
        self.expand1, self.expand0 = Expand3D(4 * embed_dim), Expand3D(2 * embed_dim)
        self.decoder1, self.decoder0 = stage(1, depths[3]), stage(0, depths[4])
        self.norm = nn.LayerNorm(embed_dim)
        self.unpatch = nn.Linear(embed_dim, embed_dim * patch_size**3)
        self.head = nn.Conv3d(embed_dim, in_channels, 3, padding=1,
                              padding_mode="circular" if boundary == "periodic" else "zeros")
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, delta_t):
        if x.ndim != 5 or tuple(x.shape[1:]) != (self.in_channels,) + (self.grid_size,) * 3:
            raise ValueError("Input must be [B, channels, grid, grid, grid]")
        if delta_t.shape not in ((x.shape[0],), (x.shape[0], 1)):
            raise ValueError("delta_t must have one value per example")
        tokens = self.patch_norm(self.patch_embed(x).permute(0, 2, 3, 4, 1))
        time = self.time_embed(delta_t.reshape(-1, 1).to(device=x.device, dtype=x.dtype))
        tokens = tokens + time[:, None, None, None, :]
        skip0 = self.skip0(self.encoder0(tokens))
        skip1 = self.skip1(self.encoder1(self.merge0(skip0)))
        coarse = self.bottleneck(self.merge1(skip1))
        decoded = self.decoder1(self.expand1(coarse) + skip1)
        decoded = self.decoder0(self.expand0(decoded) + skip0)
        volume = depth_to_space(self.unpatch(self.norm(decoded)), self.patch_size)
        return x + self.head(volume.permute(0, 4, 1, 2, 3))
