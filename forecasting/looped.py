"""Finest-stage forecasting adaptation of Looped-DiT; no diffusion sampler.

Production factories and vendored source remain unchanged. Geometry and block
applications stay fixed. The looped treatment combines parameter sharing,
headwise sigmoid gates, and an auxiliary decode of the first loop.
"""
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

ARCHITECTURE = 'finest_stage_looped_forecast_v1'


def tie_parameters(destination, source):
    """Share learned tensors while retaining position-specific shift/drop-path.

    Modules remain distinct: Swin masks/shifts and MATEY stochastic-depth rates
    belong to applications, not shared weights. Optimizers deduplicate tensors.
    """
    if dict(destination.named_parameters()).keys() != dict(source.named_parameters()).keys():
        raise ValueError('Cannot tie blocks with different learned tensors')
    for name, parameter in source.named_parameters():
        parent, _, leaf = name.rpartition('.')
        setattr(destination.get_submodule(parent) if parent else destination, leaf, parameter)


def gated_window(self, x, mask=None):
    import math
    batches, length, width = x.shape
    bias = torch.cat((self.q_bias, torch.zeros_like(self.q_bias), self.v_bias))
    qkv = F.linear(x, self.qkv.weight, bias).reshape(batches, length, 3, self.heads, width // self.heads)
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    scores = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-1, -2)
    scores = scores * self.logit_scale.clamp(max=math.log(100.)).exp()
    scores = scores + self.relative_bias().unsqueeze(0)
    if mask is not None:
        windows = mask.shape[0]
        scores = scores.reshape(-1, windows, self.heads, length, length)
        scores = (scores + mask[None, :, None]).reshape(batches, self.heads, length, length)
    weights = F.dropout(scores.softmax(-1), p=self.dropout, training=self.training)
    gates = self.loop_gate(F.layer_norm(x, (width,))).sigmoid().transpose(1, 2)[..., None]
    mixed = ((weights @ v) * gates).transpose(1, 2).reshape(batches, length, width)
    return self.output_dropout(self.projection(mixed))


def gated_matey(self, x, sequence_parallel_group=None, leadtime=None, bcs=None,
                mask_padding=None, t_pos_area=None, local_att=False):
    from ornl_vit.matey.compat import flash_attn_func
    if sequence_parallel_group is not None or mask_padding is not None:
        raise ValueError('Looped experiment supports unpadded single-GPU volumes only')
    batch, width, length = x.shape
    original = x
    normalized = self.norm1(x).transpose(1, 2)
    # Preserve upstream interleaved, per-head Q/K/V packing and BF16 attention.
    qkv = self.input_head(normalized).reshape(batch, length, self.num_heads, -1).transpose(1, 2)
    q, k, v = qkv.tensor_split(3, dim=-1)
    q, k = self.qnorm(q), self.knorm(k)
    dtype = qkv.dtype
    q, k, v = [item.to(torch.bfloat16).transpose(1, 2) for item in (q, k, v)]
    attended = flash_attn_func(q, k, v).to(dtype)
    gates = self.loop_gate(normalized).sigmoid()[..., None]
    attended = (attended * gates).reshape(batch, length, width).transpose(1, 2)
    attended = self.output_head(self.norm2(attended).transpose(1, 2)).transpose(1, 2)
    if leadtime is not None:
        attended = attended + leadtime[:, :, None]
    x = original + self.drop_path(attended * self.gamma_att[None, :, None])
    update = self.mlp(x.transpose(1, 2)).transpose(1, 2)
    return x + self.drop_path(self.gamma_mlp[None, :, None] * self.mlp_norm(update))


def add_gate(attention, width, heads, method):
    attention.loop_gate = nn.Linear(width, heads)
    nn.init.zeros_(attention.loop_gate.weight)
    nn.init.zeros_(attention.loop_gate.bias)  # initial sigmoid gate = 0.5
    attention.forward = MethodType(method, attention)


def checkpoint_blocks(blocks, model, backend):
    """Capture each application separately; tied weights still accumulate grads."""
    import contextlib
    for block in blocks:
        original = block.forward

        def wrapped(*args, _forward=original, **kwargs):
            def compute(*inputs):
                if backend is None:
                    context = contextlib.nullcontext()
                else:
                    from ornl_vit.matey.compat import attention_backend
                    context = attention_backend(backend)
                with context:
                    return _forward(*inputs, **kwargs)
            if model.training and torch.is_grad_enabled():
                with set_checkpoint_early_stop(False):
                    return checkpoint(compute, *args, use_reentrant=False, preserve_rng_state=True)
            return compute(*args)
        block.forward = wrapped


class FinestStageForecast(nn.Module):
    def __init__(self, name, config, dt_reference, *, looped, checkpointing=True):
        super().__init__()
        self.name, self.looped = name, looped
        self._collect = False
        self._mid = self._candidate = self._base = None
        if name == 'matey':
            from ornl_vit.model import MateyForecast
            self.model = MateyForecast(**config['model'], dt_reference=dt_reference,
                dt_rtol=config['data']['dt_rtol'], dt_atol=config['data']['dt_atol'])
            core = self.model.core
            finest = str(core.nhlevels - 1)
            blocks = core.module_blocks[finest]
            if len(blocks) != 4:
                raise ValueError('MATEY experiment requires four blocks at the finest level')
            if looped:
                for block in blocks:
                    add_gate(block.spatialtemporal, config['model']['embed_dim'],
                             config['model']['num_heads'], gated_matey)
                tie_parameters(blocks[2], blocks[0]); tie_parameters(blocks[3], blocks[1])
                blocks[1].register_forward_hook(self._capture_mid)
                original_decode = core.get_spatiotemporalfromsequence
                original_up = core.upsampeldata
                original_short = core.sequence_factor_short

                def factor_short(tokens, level, *args, **kwargs):
                    result = original_short(tokens, level, *args, **kwargs)
                    if self._collect and level == core.nhlevels - 1:
                        self._nfact = result[1]
                    return result

                def decoded(tokens, *args, **kwargs):
                    level = kwargs.get('ilevel', args[4] if len(args) > 4 else 0)
                    result = original_decode(tokens, *args, **kwargs)
                    if self._collect and level == core.nhlevels - 1:
                        # Restore the exact same local-group layout as upstream.
                        shape = args[2]  # [D,H,W] after patch_ids, patch_ids_ref
                        mid = core.sequence_factor_long(self._mid, level, 'tk-3D',
                                                       [4, *shape], nfact=self._nfact)
                        batch, width, total = mid.shape
                        mid = mid.reshape(batch, width, 4, total // 4).permute(2, 0, 1, 3)
                        self._candidate = original_decode(mid, *args, **kwargs)[-1]
                    return result

                def upsample(data, level):
                    result = original_up(data, level)
                    if self._collect and level == core.nhlevels - 1:
                        # Last call is the upsampled coarser prediction (the base).
                        self._base = result
                    return result
                core.get_spatiotemporalfromsequence = decoded
                core.upsampeldata = upsample
                core.sequence_factor_short = factor_short
            all_blocks = [block for stage in core.module_blocks.values() for block in stage]
            backend = self.model.backend
        elif name == 'yalamanchi':
            from yalamanchi_vit.model import YalamanchiViT
            self.model = YalamanchiViT(**config['model'])
            blocks = self.model.decoder0
            if len(blocks) != 2:
                raise ValueError('Yalamanchi experiment requires two finest decoder blocks')
            if looped:
                for block in blocks:
                    add_gate(block.attention, block.attention.width, block.attention.heads, gated_window)
                tie_parameters(blocks[1], blocks[0])
                blocks[0].register_forward_hook(self._capture_mid)
            all_blocks = [block for name in ('encoder0','encoder1','bottleneck','decoder1','decoder0')
                          for block in getattr(self.model, name)]
            backend = None
        else:
            raise ValueError('Unknown forecasting architecture')
        if checkpointing:
            checkpoint_blocks(all_blocks, self, backend)

    def _capture_mid(self, module, inputs, output):
        if self._collect:
            self._mid = output

    def forward(self, *args):
        return self.model(*args)

    def supervised(self, *args):
        """Return first-loop auxiliary and final; recurse only on the final."""
        if not self.looped:
            final = self(*args)
            return final, final
        self._collect = True
        try:
            final = self(*args)
            if self.name == 'yalamanchi':
                from yalamanchi_vit.model import depth_to_space
                volume = depth_to_space(self.model.unpatch(self.model.norm(self._mid)),
                                        self.model.patch_size)
                auxiliary = args[0] + self.model.head(volume.permute(0, 4, 1, 2, 3))
            else:
                from ornl_vit.matey.data_utils.shared_utils import normalize_spatiotemporal_persample
                core = self.model.core
                _, mean, std = normalize_spatiotemporal_persample(args[0].transpose(0, 1))
                level = core.nhlevels - 1
                filtered = core.filterdata(self._candidate[None])[0][-1]
                base = self._base
                removed = core.upsampeldata(filtered, level)
                auxiliary = (self._candidate - removed + base) * std[-1] + mean[-1]
            return auxiliary, final
        finally:
            self._collect = False
            self._mid = self._candidate = self._base = None


def restore_weights(model, weights):
    """Reject inconsistent values for aliases before loading tied parameters."""
    aliases = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        previous = aliases.setdefault(id(parameter), name)
        if previous != name and (previous not in weights or name not in weights
                                 or not torch.equal(weights[previous], weights[name])):
            raise ValueError('Checkpoint contains inconsistent shared parameter aliases')
    model.load_state_dict(weights, strict=True)
