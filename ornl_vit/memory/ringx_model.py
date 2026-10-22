"""Upstream spatial partitioning: local normalization/boundaries are intentional.

One context group processes ONE sample. Fine/middle attention stays local;
coarse attention uses the authors' broadcast/reduce RingX-A implementation.
"""
import copy
import torch
import torch.distributed as dist

from ..matey.compat import attention_backend
from ..matey.utils.distributed_utils import getblocksplitstat
from ..matey.utils.forward_options import ForwardOptionsBase
from .model import CheckpointedMatey


class RingXMatey(CheckpointedMatey):
    def __init__(self, *, group, **kwargs):
        super().__init__(**kwargs)
        self.group = group
        self.rank, self.world = dist.get_rank(group), dist.get_world_size(group)
        self.blockdict = getblocksplitstat(self.rank, self.world, *([self.grid_size] * 3))
        self.local_shape = tuple(self.blockdict['Ind_dim'])
        divisor = kwargs['patch_size'] * kwargs['filtersize'] ** (self.nlevels - 1)
        if any(self.grid_size % n for n in self.blockdict['nproc_blocks']):
            raise ValueError('Grid must divide evenly among spatial partitions')
        if any(n % divisor or n // divisor < 2 for n in self.local_shape):
            raise ValueError('Every local coarsest token axis must contain at least two tokens')

    def partition(self, fields):
        starts = [int(i) for i in self.blockdict['Ind_start']]
        return fields[(...,) + tuple(slice(i, i + n) for i, n in zip(starts, self.local_shape))].contiguous()

    @torch.no_grad()
    def assemble(self, fields):
        pieces = [torch.empty_like(fields) for _ in range(self.world)]
        dist.all_gather(pieces, fields.contiguous(), group=self.group)
        result = fields.new_empty(fields.shape[:-3] + (self.grid_size,) * 3)
        for rank, piece in enumerate(pieces):
            info = getblocksplitstat(rank, self.world, *([self.grid_size] * 3))
            slices = tuple(slice(int(i), int(i) + n) for i, n in zip(info['Ind_start'], self.local_shape))
            result[(...,) + slices] = piece
        return result

    def forward(self, history, history_times, target_time):
        if history.ndim != 6 or tuple(history.shape[1:]) != (4, 3) + self.local_shape:
            raise ValueError(f'Expected local [B,4,3,{self.local_shape}] history')
        if history_times.shape != (history.shape[0], 4) or target_time.shape != (history.shape[0],):
            raise ValueError('Incorrect history/target time shapes')
        times = torch.cat((history_times, target_time[:, None]), dim=1)
        intervals = times[:, 1:] - times[:, :-1]
        if not torch.isfinite(history).all() or not torch.isfinite(times).all():
            raise ValueError('Fields and times must be finite')
        if not torch.all(intervals > 0) or not torch.allclose(
                intervals, torch.full_like(intervals, self.dt_reference),
                rtol=self.dt_rtol, atol=self.dt_atol):
            raise ValueError('History must use consecutive saved times')
        leadtime = (intervals[:, -1:] / self.dt_reference).to(history)
        labels = torch.arange(3, device=history.device).expand(history.shape[0], -1)
        opts = ForwardOptionsBase(imod=self.nlevels - 1, imod_bottom=0,
            tkhead_name='tk-3D', leadtime=leadtime, sequence_parallel_group=self.group,
            blockdict=copy.deepcopy(self.blockdict))
        with attention_backend(self.backend):
            prediction = self.core(history.transpose(0, 1), labels, None, opts)
        if prediction.shape != history[:, -1].shape:
            raise ValueError('Unexpected local output shape')
        return prediction


def average_gradients(model, group):
    """Average local-loss parameter gradients after RingX's activation backward.

    Each equal-volume rank computes a local mean loss. Summing parameter
    contributions then dividing by context-group size gives global-mean loss
    gradients. Explicit post-backward reduction avoids interleaving DDP and
    checkpoint-replayed RingX collectives. Unused parameters remain unused.
    """
    world = dist.get_world_size(group)
    params = list(model.parameters())
    present = torch.tensor([p.grad is not None for p in params],
                           device=params[0].device, dtype=torch.int32)
    dist.all_reduce(present, group=group)
    if torch.any((present != 0) & (present != world)):
        raise RuntimeError('Ranks disagree about used parameters')
    active = [p for p in params if p.grad is not None]
    flat = torch.cat([p.grad.reshape(-1) for p in active])
    dist.all_reduce(flat, group=group)
    flat.div_(world)
    start = 0
    for p in active:
        p.grad.copy_(flat[start:start + p.numel()].view_as(p))
        start += p.numel()
