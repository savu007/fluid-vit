"""Five-step training with baseline normalization and MATEY's native forward pass."""
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from baseline_vit.dataset import CacheIndex, training_statistics
from baseline_vit.engine import select_device
from forecasting.checkpoint import check_cache
from .dataset import HistoryDataset
from .model import ARCHITECTURE, UPSTREAM_COMMIT, MateyForecast
from .rollout import forecast_five


def make_model(cfg, dt_reference, device):
    return MateyForecast(**cfg['model'], dt_reference=dt_reference,
                         dt_rtol=cfg['data']['dt_rtol'], dt_atol=cfg['data']['dt_atol']).to(device)


def loss_scale(stats, device):
    std = torch.tensor(stats['std'], device=device, dtype=torch.float32)
    if std.shape != (3,) or not torch.isfinite(std).all() or not torch.all(std > 0):
        raise ValueError('Need three finite positive training standard deviations')
    return std.reshape(1, 1, 3, 1, 1, 1)


def normalized_mse(prediction, target, stats):
    # Equivalent to baseline MSE between normalized fields: channel means cancel.
    return ((prediction - target) / loss_scale(stats, prediction.device)).square().mean()


@torch.no_grad()
def evaluate(model, loader, stats, device, save_first=None):
    model.eval()
    std = loss_scale(stats, device).double()
    totals = np.zeros((5, 6), dtype=np.float64)
    count = 0
    for history, targets, times in loader:
        history, targets, times = history.to(device), targets.to(device), times.to(device)
        prediction = forecast_five(model, history, times)
        error = prediction.double() - targets.double()
        for lead in range(5):
            values = [error[:, lead].square().sum(), (error / std)[:, lead].square().sum(),
                      error[:, lead].abs().sum(),
                      (history[:, -1].double() - targets[:, lead].double()).square().sum(),
                      prediction[:, lead].double().square().sum(), targets[:, lead].double().square().sum()]
            totals[lead] += [v.item() for v in values]
        if count == 0 and save_first is not None:
            np.savez_compressed(save_first, history=history[0].cpu().numpy(),
                                actual=targets[0].cpu().numpy(), predicted=prediction[0].cpu().numpy(),
                                times=times[0].cpu().numpy())
        count += targets[:, 0].numel()
    if count == 0:
        raise ValueError('No evaluation windows')
    metrics = []
    for lead, (error, norm, absolute, persistence, predicted, actual) in enumerate(totals.tolist(), 1):
        metrics.append(dict(lead_steps=lead, physical_mse=error / count, normalized_mse=norm / count,
            physical_mae=absolute / count,
            relative_l2=float(np.sqrt(error / actual)) if actual else None,
            persistence_mse=persistence / count,
            mse_skill_vs_persistence=1 - error / persistence if persistence else None,
            predicted_kinetic_energy=1.5 * predicted / count,
            target_kinetic_energy=1.5 * actual / count, samples=len(loader.dataset)))
    return dict(horizon=5, samples=len(loader.dataset), horizons=metrics,
                mean_normalized_mse=float(totals[:, 1].mean() / count))


def train(cfg):
    opt = cfg['training']
    device = select_device(opt['device'])
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    index = CacheIndex(cfg)
    training = HistoryDataset(index, 'train', max_windows=opt['max_train_windows'])
    validation = HistoryDataset(index, 'val')
    output = Path(opt['output_dir'])
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.json').write_text(json.dumps(cfg, indent=2))
    stats = training_statistics(index)
    (output / 'normalization.json').write_text(json.dumps(stats, indent=2))
    model = make_model(cfg, index.dt_reference, device)
    environment = dict(architecture=ARCHITECTURE, upstream_commit=UPSTREAM_COMMIT,
        torch=str(torch.__version__), numpy=np.__version__, device=str(device),
        cuda_device=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        parameters=sum(p.numel() for p in model.parameters()),
        attention_backend=cfg['model']['attention_backend'],
        training_windows=len(training), validation_windows=len(validation))
    (output / 'environment.json').write_text(json.dumps(environment, indent=2))
    print(json.dumps(environment), flush=True)
    common = dict(batch_size=opt['batch_size'], num_workers=opt['num_workers'], pin_memory=device.type == 'cuda')
    train_loader = DataLoader(training, shuffle=True, generator=torch.Generator().manual_seed(cfg['seed']), **common)
    val_loader = DataLoader(validation, shuffle=False, **common)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt['learning_rate'], weight_decay=opt['weight_decay'])
    best = float('inf')
    for epoch in range(1, opt['epochs'] + 1):
        if device.type == 'cuda':
            torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        model.train(); total, examples = 0., 0
        for history, targets, times in train_loader:
            history, targets, times = history.to(device), targets.to(device), times.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = forecast_five(model, history, times)
            loss = normalized_mse(prediction, targets, stats)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt['grad_clip'], error_if_nonfinite=True)
            optimizer.step()
            total += loss.item() * history.shape[0]; examples += history.shape[0]
        report = evaluate(model, val_loader, stats, device)
        if device.type == 'cuda': torch.cuda.synchronize(device)
        record = dict(epoch=epoch, objective=opt['objective'], history=4, horizon=5,
            training_loss=total / examples, validation=report,
            optimizer_updates=epoch * len(train_loader), seconds=time.perf_counter() - started,
            peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None)
        with (output / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
        print(json.dumps(record, allow_nan=False), flush=True)
        improved = report['mean_normalized_mse'] < best
        saved = dict(format_version=1, architecture=ARCHITECTURE, model=model.state_dict(), config=cfg,
                     normalization=stats, epoch=epoch, validation=report, dt_reference=index.dt_reference,
                     cache_fingerprint=index.fingerprint, environment=environment)
        for name in (['best.pt'] if improved else []) + ['last.pt']:
            temporary = output / (name + '.tmp')
            torch.save(saved, temporary); temporary.replace(output / name)
        if improved: best = report['mean_normalized_mse']
    print(f'Complete; best five-step validation normalized MSE={best:g}', flush=True)


def restore(path, device):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('format_version') != 1 or saved.get('architecture') != ARCHITECTURE:
        raise ValueError('Not a supported four-frame MATEY checkpoint')
    index = CacheIndex(saved['config']); check_cache(saved, index)
    model = make_model(saved['config'], saved['dt_reference'], device)
    model.load_state_dict(saved['model']); model.eval()
    return model, saved, index
