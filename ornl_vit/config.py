"""Explicit, validated settings for the approved MATEY adaptation."""
import math
from pathlib import Path
import yaml
from .model import validate_model


def load_config(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    sections = {
        'data': {'cache_dir', 'channels', 'train', 'val', 'test', 'gaps', 'dt_rtol', 'dt_atol'},
        'model': {'grid_size', 'in_channels', 'history', 'patch_size', 'embed_dim', 'num_heads',
                  'processor_blocks', 'nlevels', 'filtersize', 'upsampling', 'drop_path', 'attention_backend'},
        'training': {'output_dir', 'device', 'epochs', 'batch_size', 'num_workers', 'learning_rate',
                     'weight_decay', 'grad_clip', 'max_train_windows', 'objective', 'horizon'},
    }
    if not isinstance(cfg, dict) or set(cfg) != {'seed', *sections}:
        raise ValueError('Expected seed, data, model and training sections')
    for section, keys in sections.items():
        if not isinstance(cfg[section], dict) or set(cfg[section]) != keys:
            raise ValueError(f'{section} must contain exactly {sorted(keys)}')
    if type(cfg['seed']) is not int or cfg['seed'] < 0:
        raise ValueError('seed must be a nonnegative integer')
    data, model, training = cfg['data'], cfg['model'], cfg['training']
    validate_model(**model, dt_reference=1.0, dt_rtol=data['dt_rtol'], dt_atol=data['dt_atol'])
    if data['channels'] != ['u', 'v', 'w']:
        raise ValueError('Channel order must be [u,v,w]')
    seen = set()
    for split in ('train', 'val', 'test', 'gaps'):
        ids = data[split]
        if not isinstance(ids, list) or any(type(i) is not int for i in ids):
            raise ValueError(f'{split} must contain integer IDs')
        if split != 'gaps' and not ids:
            raise ValueError(f'{split} must not be empty')
        if len(ids) != len(set(ids)) or seen.intersection(ids):
            raise ValueError('Split IDs must be unique and disjoint')
        seen.update(ids)
    for name in ('learning_rate', 'weight_decay', 'grad_clip'):
        value = training[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (name != 'weight_decay' and value == 0):
            raise ValueError(f'Invalid {name}')
    for name in ('epochs', 'batch_size', 'num_workers'):
        value = training[name]
        if type(value) is not int or value < (0 if name == 'num_workers' else 1):
            raise ValueError(f'Invalid {name}')
    limit = training['max_train_windows']
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError('max_train_windows must be a positive integer or null')
    if training['objective'] != 'rollout_normalized_mse' or type(training['horizon']) is not int or training['horizon'] != 5:
        raise ValueError('Approved objective is five-step rollout_normalized_mse')
    if training['device'] not in ('cpu', 'cuda'):
        raise ValueError('device must be cpu or cuda')
    for mapping, key in ((data, 'cache_dir'), (training, 'output_dir')):
        value = mapping[key]
        if not isinstance(value, str) or not value:
            raise ValueError(f'Invalid {key}')
        p = Path(value).expanduser()
        mapping[key] = str(p if p.is_absolute() else (path.parent / p).resolve())
    return cfg
