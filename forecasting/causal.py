"""Explicit temporal-attention adapter for isolated diagnostics and ordered MATEY.

Time-major tokens are required, including within each local spatial group.
Frame-local normalization is an explicit experimental change, not upstream MATEY.
"""
from types import MethodType
import torch
from torch.nn import functional as F


def temporal_mask(length, frames, device=None):
    if frames < 1 or length % frames:
        raise ValueError('Token sequence must contain equal-sized temporal slices')
    times = torch.arange(length, device=device) // (length // frames)
    # SDPA True means allowed. All spatial keys at the same time remain visible.
    return times[None, :] <= times[:, None]


def install_temporal_attention(block, *, frames=4, causal=True, frame_local_norm=False):
    """Change only this block instance. Supports unpadded single-device attention."""
    block.register_buffer('_temporal_allowed', torch.empty(0, dtype=torch.bool), persistent=False)
    def forward(self, x, sequence_parallel_group=None, leadtime=None, bcs=None,
                mask_padding=None, t_pos_area=None, local_att=False):
        if sequence_parallel_group is not None or mask_padding is not None:
            raise ValueError('Diagnostic supports unpadded single-device inputs only')
        batch, width, length = x.shape
        if frames < 1 or length % frames:
            raise ValueError('Invalid time-major token geometry')
        mask = None
        if causal:
            if self._temporal_allowed.shape != (length, length) or self._temporal_allowed.device != x.device:
                self._temporal_allowed = temporal_mask(length, frames, x.device)
            mask = self._temporal_allowed

        def norm(layer, values):
            if not frame_local_norm:
                return layer(values)
            spatial = length // frames
            values = values.reshape(batch, width, frames, spatial).transpose(1, 2)
            values = layer(values.reshape(batch * frames, width, spatial))
            return values.reshape(batch, frames, width, spatial).transpose(1, 2).reshape(batch, width, length)

        normalized = norm(self.norm1, x).transpose(1, 2)
        qkv = self.input_head(normalized).reshape(batch, length, self.num_heads, -1).transpose(1, 2)
        q, k, v = qkv.tensor_split(3, dim=-1)
        q, k = self.qnorm(q), self.knorm(k)
        attended = F.scaled_dot_product_attention(
            q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16),
            attn_mask=mask, dropout_p=0., is_causal=False).to(qkv.dtype)
        attended = attended.transpose(1, 2).reshape(batch, length, width).transpose(1, 2)
        attended = self.output_head(norm(self.norm2, attended).transpose(1, 2)).transpose(1, 2)
        if leadtime is not None:
            attended = attended + leadtime[:, :, None]
        x = x + self.drop_path(attended * self.gamma_att[None, :, None])
        update = self.mlp(x.transpose(1, 2)).transpose(1, 2)
        return x + self.drop_path(self.gamma_mlp[None, :, None] * norm(self.mlp_norm, update))
    block.forward = MethodType(forward, block)
    return block


# Backward-compatible name used by the earlier block diagnostic.
install_block_test = install_temporal_attention
