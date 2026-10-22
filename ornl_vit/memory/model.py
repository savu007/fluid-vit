"""Activation checkpointing without changing the original model's state-dict keys."""
import torch
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

from ..matey.compat import attention_backend
from ..model import MateyForecast


def checkpoint_blocks(model):
    """Recompute blocks only when autograd is enabled; preserve stochastic-depth RNG.

    Capture the backend inside the recomputation closure: backward happens after
    MateyForecast.forward has exited its attention_backend context manager.
    """
    for blocks in model.core.module_blocks.values():
        for block in blocks:
            original = block.forward

            def wrapped(*args, _forward=original, **kwargs):
                def compute(*inputs):
                    with attention_backend(model.backend):
                        return _forward(*inputs, **kwargs)
                if model.training and torch.is_grad_enabled():
                    # Complete replay also keeps distributed collectives in order.
                    with set_checkpoint_early_stop(False):
                        return checkpoint(compute, *args, use_reentrant=False,
                                          preserve_rng_state=True)
                return compute(*args)

            block.forward = wrapped


class CheckpointedMatey(MateyForecast):
    def __init__(self, *, activation_checkpointing=True, **kwargs):
        super().__init__(**kwargs)
        if activation_checkpointing:
            checkpoint_blocks(self)


def pushforward_five(model, history, times, memory_callback=None):
    """Return only lead five; the first four predictions have no autograd graph.

    Train/eval mode is left to the caller. Warmup predictions therefore retain
    training-time stochastic depth, matching the upstream pushforward approach.
    """
    if history.ndim != 6 or history.shape[1:3] != (4, 3):
        raise ValueError('Expected [B,4,3,D,H,W] history')
    if times.shape != (history.shape[0], 9):
        raise ValueError('Expected nine recorded times per sample')
    state = history
    with torch.no_grad():
        for step in range(4):
            prediction = model(state, times[:, step:step + 4], times[:, step + 4])
            if not torch.isfinite(prediction).all():
                raise FloatingPointError(f'Nonfinite pushforward at lead {step + 1}')
            state = torch.cat((state[:, 1:], prediction[:, None]), dim=1)
            if memory_callback is not None:
                memory_callback(step + 1)
    prediction = model(state.detach(), times[:, 4:8], times[:, 8])
    if not torch.isfinite(prediction).all():
        raise FloatingPointError('Nonfinite pushforward at lead five')
    if memory_callback is not None:
        memory_callback(5)
    return prediction
