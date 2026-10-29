"""Ordered-time MATEY adaptation; vendored modules remain unchanged."""
from types import FunctionType, MethodType
import torch
from .causal import install_temporal_attention
from .looped import FinestStageForecast, checkpoint_blocks

ARCHITECTURE = 'matey_prefix_normalization_temporal_attention_v1'


def prefix_normalize(x, sequence_parallel_group=None):
    """[T,B,C,D,H,W]: frame i uses spatial statistics of frames 0..i.

    Preserve upstream detached sample-standard-deviation statistics and floor.
    The last slice's statistics equal the original full-history normalization.
    """
    if sequence_parallel_group is not None or x.ndim != 6:
        raise ValueError('Prefix normalization requires unpartitioned six-dimensional input')
    means, deviations = [], []
    with torch.no_grad():
        for stop in range(1, x.shape[0]+1):
            std, mean = torch.std_mean(x[:stop], dim=(0,-3,-2,-1), keepdim=True)
            means.append(mean); deviations.append(std.clamp_min(1e-4))
        mean, std = torch.cat(means, dim=0), torch.cat(deviations, dim=0)
    return (x-mean)/std, mean, std


def install_prefix_normalization(core):
    """Rebind this instance's forward with one explicit preprocessing override.

    A private globals mapping changes only the imported normalization function.
    Recursive hierarchy calls resolve self.forward and inherit this override.
    Other model instances and the vendored module's globals are untouched.
    """
    original = core.forward.__func__
    namespace = dict(original.__globals__)
    namespace['normalize_spatiotemporal_persample'] = prefix_normalize
    forward = FunctionType(original.__code__, namespace, original.__name__,
                           original.__defaults__, original.__closure__)
    forward.__kwdefaults__ = original.__kwdefaults__
    core.forward = MethodType(forward, core)


class OrderedMateyForecast(FinestStageForecast):
    def __init__(self, config, dt_reference, *, causal, checkpointing=True):
        if config['model']['attention_backend'] != 'sdpa':
            raise ValueError('Ordered-time experiment requires the explicit SDPA backend')
        super().__init__('matey', config, dt_reference, looped=False, checkpointing=False)
        self.temporal_causal = bool(causal)
        install_prefix_normalization(self.model.core)
        blocks = [block for stage in self.model.core.module_blocks.values() for block in stage]
        for block in blocks:
            install_temporal_attention(block.spatialtemporal, frames=config['model']['history'],
                               causal=causal, frame_local_norm=True)
        if checkpointing:
            checkpoint_blocks(blocks, self, self.model.backend)
