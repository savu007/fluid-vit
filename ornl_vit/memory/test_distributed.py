"""Run with torchrun --standalone --nproc-per-node=4 -m ...test_distributed.

CPU/Gloo checks trainer integration with an explicitly substituted SDPA oracle;
it DOES NOT test the authors' CUDA RingX kernel. CUDA/NCCL tests actual RingX.
"""
import argparse
import json
import os
from pathlib import Path
import tempfile
from contextlib import nullcontext
from datetime import timedelta
from unittest.mock import patch

import torch
import numpy as np
import torch.distributed as dist
from torch.nn import functional as F

from ..matey.compat import ringX_attn_func
from .model import pushforward_five
from .ringx_model import RingXMatey, average_gradients


def kernel_check(device):
    if device.type != 'cuda':
        raise ValueError('The authors\' RingX/FlashAttention kernel requires CUDA')
    rank, world = dist.get_rank(), dist.get_world_size()
    dtype = torch.bfloat16
    # Unequal sequence lengths exercise the actual varlen RingX entry point.
    lengths = [5 + r for r in range(world)]
    total = sum(lengths); start = sum(lengths[:rank]); end = start + lengths[rank]
    gen = torch.Generator(device=device).manual_seed(123)
    full = [torch.randn(1, total, 3, 64, generator=gen, device=device, dtype=dtype,
                        requires_grad=True) for _ in range(3)]
    dout = torch.randn(full[0].shape, generator=gen, device=device, dtype=dtype)
    local = [x[:, start:end].detach().clone().requires_grad_() for x in full]
    expected = F.scaled_dot_product_attention(*(x.transpose(1, 2) for x in full)).transpose(1, 2)
    actual = ringX_attn_func(*local, group=dist.group.WORLD)
    (actual * dout[:, start:end]).sum().backward()
    (expected * dout).sum().backward()
    atol, rtol = 0.04, 0.08
    torch.testing.assert_close(actual, expected[:, start:end], atol=atol, rtol=rtol)
    for x, y in zip(local, full):
        torch.testing.assert_close(x.grad, y.grad[:, start:end], atol=atol, rtol=rtol)
    dist.barrier()
    if rank == 0:
        print(f'RingX output and Q/K/V gradients agree with full attention ({device.type}, {world} ranks)', flush=True)


def cpu_attention_oracle(q, k, v, causal=False, group=None):
    """Test-only global attention with autograd-aware gathering; never used by jobs."""
    from torch.distributed.nn.functional import all_gather
    k = torch.cat(all_gather(k.contiguous(), group=group), dim=1)
    v = torch.cat(all_gather(v.contiguous(), group=group), dim=1)
    return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                         v.transpose(1, 2)).transpose(1, 2)


def model_check(device):
    rank, world = dist.get_rank(), dist.get_world_size()
    kwargs = dict(grid_size=32, in_channels=3, history=4, patch_size=2,
        embed_dim=24, num_heads=3, processor_blocks=6, nlevels=3,
        filtersize=2, upsampling='learned', drop_path=.2, attention_backend='sdpa',
        dt_reference=1., dt_rtol=.003, dt_atol=1e-10, group=dist.group.WORLD)
    torch.manual_seed(55)
    plain = RingXMatey(**kwargs, activation_checkpointing=False).to(device).train()
    checked = RingXMatey(**kwargs, activation_checkpointing=True).to(device).train()
    checked.load_state_dict(plain.state_dict())
    gen = torch.Generator(device=device).manual_seed(91)
    history = torch.randn(1, 4, 3, 32, 32, 32, generator=gen, device=device)
    times = torch.arange(9, device=device, dtype=torch.float64)[None]
    local = plain.partition(history)
    torch.testing.assert_close(plain.assemble(local), history, rtol=0, atol=0)
    results = []
    for model in (plain, checked):
        torch.manual_seed(71 + rank)
        pred = pushforward_five(model, local, times)
        pred.square().mean().backward()
        average_gradients(model, dist.group.WORLD)
        results.append(pred.detach())
    torch.testing.assert_close(*results, atol=1e-5, rtol=1e-5)
    for (name, p), (_, q) in zip(plain.named_parameters(), checked.named_parameters()):
        if p.grad is None:
            assert q.grad is None, name
        else:
            torch.testing.assert_close(p.grad, q.grad, atol=1e-5, rtol=2e-4, msg=name)
            assert torch.isfinite(q.grad).all(), name
    # All ranks must apply identical updates despite different spatial inputs.
    torch.optim.AdamW(checked.parameters(), lr=1e-4).step()
    flat = torch.cat([p.detach().flatten() for p in checked.parameters()])
    reference = flat.clone(); dist.broadcast(reference, src=0)
    torch.testing.assert_close(flat, reference, rtol=0, atol=0)
    dist.barrier()
    if rank == 0:
        print(f'Partition/reassembly, stochastic checkpoint replay, gradients and synchronized update pass ({world} ranks)', flush=True)


def pipeline_check(device, objective='pushforward_final_normalized_mse'):
    """Small full-stack distributed test with one fresh optimizer update."""
    from baseline_vit.dataset import CacheIndex, training_statistics
    from ..config import load_config
    from .run import Execution, smoke, train
    rank = dist.get_rank()
    temporary = tempfile.TemporaryDirectory() if rank == 0 else None
    paths = [temporary.name if rank == 0 else None]
    dist.broadcast_object_list(paths, src=0)
    root = Path(paths[0]); cache = root / 'cache'; output = root / 'run'
    try:
        if rank == 0:
            cache.mkdir(); output.mkdir(); times = {}; rng = np.random.default_rng(24)
            for fid in (1, 3, 5):
                times[str(fid)] = (np.arange(10) + fid * 10).astype(float).tolist()
                np.save(cache / f'fluid_{fid}.npy', rng.normal(size=(10, 3, 32, 32, 32)).astype('float32'))
            (cache / 'manifest.json').write_text(json.dumps(dict(channels=['u', 'v', 'w'],
                target_grid=[32]*3, var_dims=['time', 'x', 'y', 'z'], file_ids=[1, 3, 5], time_coordinates=times)))
        dist.barrier()
        cfg = load_config(Path(__file__).parents[1] / 'config.yaml')
        cfg['model'].update(grid_size=32, embed_dim=24, processor_blocks=3, drop_path=0.)
        cfg['data'].update(cache_dir=str(cache), train=[1], val=[3], test=[5], gaps=[])
        cfg['training'].update(device=str(device.type), epochs=1, max_train_windows=1,
                              output_dir=str(output), objective=objective)
        index = CacheIndex(cfg)
        objects = [training_statistics(index) if rank == 0 else None]
        dist.broadcast_object_list(objects, src=0); stats = objects[0]
        execution = Execution('ringx', device)
        settings = dict(smoke_updates=2, wall_budget_seconds=300)
        smoke(cfg, settings, index, stats, output, execution)
        train(cfg, settings, index, stats, output, execution)
        if rank == 0:
            for filename in ('acceptance.json', 'best.pt', 'last.pt', 'test.json',
                             'energy-data.npz', 'energy-evolution.png', 'energy-spectra.png', 'complete.json'):
                assert (output / filename).is_file(), filename
            print('Distributed acceptance/train/reload/full-field energy pipeline passes', flush=True)
        dist.barrier()
    finally:
        if temporary is not None: temporary.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--kernel-only', action='store_true')
    parser.add_argument('--pipeline', action='store_true')
    parser.add_argument('--rollout', action='store_true', help='Use full five-step rollout in the pipeline test')
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device('cuda', int(os.environ['LOCAL_RANK'])) if args.device == 'cuda' else torch.device('cpu')
    if device.type == 'cuda': torch.cuda.set_device(device)
    dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo', timeout=timedelta(minutes=5))
    try:
        if device.type == 'cuda':
            kernel_check(device)
        elif args.kernel_only:
            raise ValueError('--kernel-only requires --device cuda')
        context = patch('ornl_vit.matey.models.attention_modules.ringX_attn_func',
                        cpu_attention_oracle) if device.type == 'cpu' else nullcontext()
        with context:
            if not args.kernel_only: model_check(device)
            if args.pipeline: pipeline_check(device, 'rollout_normalized_mse' if args.rollout else 'pushforward_final_normalized_mse')
        if device.type == 'cpu' and dist.get_rank() == 0:
            print('CPU integration only: actual RingX CUDA kernel NOT exercised', flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
