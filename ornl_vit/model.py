"""Thin wrapper around the authors' TurbT; full-volume four-frame input."""
import math
import torch
from torch import nn

ARCHITECTURE = 'ornl_matey_turbt_3d_history4_v1'
UPSTREAM_COMMIT = '631a4139295d4f51e573d474cff7d6586f3aa310'


def validate_model(grid_size, in_channels, history, patch_size, embed_dim, num_heads,
                   processor_blocks, nlevels, filtersize, upsampling, drop_path,
                   attention_backend, dt_reference, dt_rtol, dt_atol):
    for name, value in locals().copy().items():
        if name in ('grid_size', 'in_channels', 'history', 'patch_size', 'embed_dim', 'num_heads',
                    'processor_blocks', 'nlevels', 'filtersize'):
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
    if in_channels != 3 or history != 4:
        raise ValueError('This adaptation requires three velocity channels and four frames')
    if nlevels < 2 or filtersize < 2 or processor_blocks % nlevels:
        raise ValueError('Require at least two levels and equal blocks per level')
    if patch_size & (patch_size - 1):
        raise ValueError('The upstream patch stem requires power-of-two patch sizes')
    if grid_size % (patch_size * filtersize ** (nlevels - 1)):
        raise ValueError('Grid must divide exactly through filters and patching')
    if grid_size // (patch_size * filtersize ** (nlevels - 1)) < 2:
        raise ValueError('Coarsest token grid must have at least two tokens per axis')
    if embed_dim % 12 or embed_dim % num_heads:
        raise ValueError('Width must be divisible by 12 and by the head count')
    if upsampling not in ('learned', 'fixed', 'linear'):
        raise ValueError('Unknown upstream upsampling mode')
    if attention_backend not in ('sdpa', 'flash'):
        raise ValueError('Backend must be sdpa or flash')
    if not isinstance(drop_path, (int, float)) or not math.isfinite(drop_path) or not 0 <= drop_path < 1:
        raise ValueError('Invalid drop_path')
    for name, value in [('dt_reference', dt_reference), ('dt_rtol', dt_rtol), ('dt_atol', dt_atol)]:
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'Invalid {name}')
    if dt_reference == 0:
        raise ValueError('dt_reference must be positive')


class MateyForecast(nn.Module):
    def __init__(self, *, grid_size, in_channels, history, patch_size, embed_dim, num_heads,
                 processor_blocks, nlevels, filtersize, upsampling, drop_path,
                 attention_backend, dt_reference, dt_rtol, dt_atol):
        super().__init__()
        validate_model(**{k: v for k, v in locals().copy().items() if k not in ('self', '__class__')})
        from .matey.models.turbt import TurbT
        self.grid_size, self.history, self.in_channels = grid_size, history, in_channels
        self.backend, self.nlevels = attention_backend, nlevels
        self.dt_reference, self.dt_rtol, self.dt_atol = dt_reference, dt_rtol, dt_atol
        self.core = TurbT(
            tokenizer_heads=[dict(head_name='tk-3D', patch_size=[[patch_size] * 3])],
            n_states=in_channels, n_steps=history, embed_dim=embed_dim, num_heads=num_heads,
            processor_blocks=processor_blocks, drop_path=drop_path, leadtime=True,
            bias_type='PositionAreaBias', hierarchical=dict(nlevels=nlevels, filtersize=filtersize,
                fixedupsample=upsampling == 'fixed', linearupsample=upsampling == 'linear'))

    def forward(self, history, history_times, target_time):
        from .matey.compat import attention_backend
        from .matey.utils.forward_options import ForwardOptionsBase
        if history.ndim != 6 or tuple(history.shape[1:]) != (4, 3) + (self.grid_size,) * 3:
            raise ValueError('Expected [B,4,3,grid,grid,grid] physical-valued input')
        if history_times.shape != (history.shape[0], 4) or target_time.shape != (history.shape[0],):
            raise ValueError('Incorrect history/target time shapes')
        times = torch.cat((history_times, target_time[:, None]), dim=1)
        if not torch.isfinite(history).all() or not torch.isfinite(times).all():
            raise ValueError('Fields and times must be finite')
        intervals = times[:, 1:] - times[:, :-1]
        expected = torch.full_like(intervals, self.dt_reference)
        if not torch.all(intervals > 0) or not torch.allclose(intervals, expected, rtol=self.dt_rtol, atol=self.dt_atol):
            raise ValueError('History must use consecutive saved times within configured tolerance')
        leadtime = (intervals[:, -1:] / self.dt_reference).to(device=history.device, dtype=history.dtype)
        labels = torch.arange(3, device=history.device).expand(history.shape[0], -1)
        # TurbT mutates opts during recursive hierarchy evaluation: create fresh each call.
        opts = ForwardOptionsBase(imod=self.nlevels - 1, imod_bottom=0,
                                  tkhead_name='tk-3D', leadtime=leadtime)
        with attention_backend(self.backend):
            prediction = self.core(history.transpose(0, 1), labels, None, opts)
        if prediction.shape != history[:, -1].shape:
            raise ValueError('Upstream model returned an unexpected shape')
        # TurbT restores physical values internally. It predicts the next field;
        # do not add the last input as another residual or normalize a second time.
        return prediction
