"""Explicit local execution backends; no changes to attention scores or weights.

The model-only export can run without timm or FlashAttention. SDPA is an
explicit alternative backend, not a claim of bitwise equivalence to FlashAttention.
The caller's upstream bfloat16 Q/K/V casts are preserved.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import torch
from torch import nn
from torch.nn import functional as F

_BACKEND = ContextVar('matey_attention_backend', default='flash')


@contextmanager
def attention_backend(name):
    if name not in ('sdpa', 'flash'):
        raise ValueError('Attention backend must be sdpa or flash')
    token = _BACKEND.set(name)
    try:
        yield
    finally:
        _BACKEND.reset(token)


class DropPath(nn.Module):
    """Per-example stochastic depth with the usual keep-probability rescaling."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)
        if not 0 <= self.drop_prob < 1:
            raise ValueError('Invalid stochastic-depth probability')

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        keep = 1 - self.drop_prob
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


def flash_attn_func(q, k, v):
    if _BACKEND.get() == 'flash':
        from flash_attn import flash_attn_func as upstream
        return upstream(q, k, v)
    return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                         v.transpose(1, 2), dropout_p=0.0,
                                         is_causal=False).transpose(1, 2)


def ringX_attn_func(*args, **kwargs):
    # Lazy import: local attention does not require the external flash-attn package.
    from .models.ringX_attn import ringX_varlen_attn_func
    return ringX_varlen_attn_func(*args, **kwargs)
