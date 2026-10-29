"""Profile, train and evaluate the isolated finest-stage looping experiment."""
import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from baseline_vit.dataset import CacheIndex, training_statistics
from baseline_vit.engine import normalization_tensors
from ornl_vit.config import load_config as matey_config
from ornl_vit.dataset import HistoryDataset
from ornl_vit.engine import evaluate as evaluate_matey
from yalamanchi_vit.config import load_config as yala_config
from .dataset import SequenceDataset
from .energy import energy_spectrum, plot_energy
from .looped import ARCHITECTURE, FinestStageForecast, restore_weights
from .rollout import evaluate_rollout

ROOT = Path(__file__).resolve().parents[1]
CASES = ('matey-control', 'matey-looped', 'yalamanchi-control', 'yalamanchi-looped')


def source_fingerprint():
    paths = [ROOT/'forecasting/looped.py', ROOT/'forecasting/looped_experiment.py',
             ROOT/'forecasting/rollout.py', ROOT/'forecasting/energy.py',
             ROOT/'yalamanchi_vit/model.py', ROOT/'ornl_vit/model.py',
             ROOT/'ornl_vit/matey/compat.py']
    paths += sorted((ROOT/'ornl_vit/matey/models').glob('*.py'))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode()); digest.update(path.read_bytes())
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def config_for(name):
    matey = matey_config(ROOT / 'ornl_vit/config.yaml')
    if name == 'matey':
        cfg = matey
    elif name == 'yalamanchi':
        cfg = yala_config(ROOT / 'yalamanchi_vit/config-matched.yaml')
        cfg['data'] = copy.deepcopy(matey['data'])
    else:
        raise ValueError('Unknown model')
    cfg['training']['epochs'] = 25
    return cfg


def datasets(cfg, name, index):
    if name == 'matey':
        return HistoryDataset(index, 'train'), HistoryDataset(index, 'val')
    return SequenceDataset(index, 'train', 5), SequenceDataset(index, 'val', 5)


def rollout_loss(model, batch, stats, device):
    """Five fully differentiable steps; auxiliary outputs never feed the future."""
    initial, targets, intervals = [item.to(device) for item in batch]
    losses, auxiliary_losses, final_losses = [], [], []
    if model.name == 'yalamanchi':
        mean, std = normalization_tensors(stats, device)
        state = (initial - mean) / std
        targets = (targets - mean[:, None]) / std[:, None]
    else:
        state = initial
        std = torch.tensor(stats['std'], device=device).reshape(1, 3, 1, 1, 1)
    for lead in range(5):
        args = ((state, intervals[:, lead:lead+4], intervals[:, lead+4])
                if model.name == 'matey' else (state, intervals[:, lead]))
        auxiliary, final = model.supervised(*args)
        scale = std if model.name == 'matey' else 1.
        a = ((auxiliary - targets[:, lead]) / scale).square().mean()
        f = ((final - targets[:, lead]) / scale).square().mean()
        auxiliary_losses.append(a); final_losses.append(f)
        losses.append((a + f) / 2 if model.looped else f)
        state = (torch.cat((state[:, 1:], final[:, None]), dim=1)
                 if model.name == 'matey' else final)
    return torch.stack(losses).mean(), torch.stack(auxiliary_losses).mean(), torch.stack(final_losses).mean()


def update(model, batch, optimizer, stats, cfg, device):
    model.train(); optimizer.zero_grad(set_to_none=True)
    loss, auxiliary, final = rollout_loss(model, batch, stats, device)
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite rollout loss')
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['grad_clip'], error_if_nonfinite=True)
    optimizer.step()
    return dict(training_loss=float(loss.detach()), auxiliary_mse=float(auxiliary.detach()),
                final_mse=float(final.detach()))


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def validate(model, loader, stats, device):
    return (evaluate_matey(model, loader, stats, device) if model.name == 'matey'
            else evaluate_rollout(model, loader, stats, device))


@torch.no_grad()
def common_evaluation(model, cfg, index, stats, device, output=None, limit=None, split='test'):
    """Common four-frame origins; Yalamanchi uses only the last observed frame."""
    model.eval()
    data = HistoryDataset(index, split, cross_file=True,
        dt_rtol=cfg['data']['dt_rtol'], dt_atol=cfg['data']['dt_atol'])
    starts = [dict(file_id=data.continuous.frames[s+3][0], frame=data.continuous.frames[s+3][1],
                   time=float(data.continuous.times[s+3])) for s in data.pairs]
    if limit is not None:
        data = Subset(data, range(min(limit, len(data))))
        starts = starts[:len(data)]
    mean, std = normalization_tensors(stats, device)
    total_error = np.zeros(5); total_norm = np.zeros(5); total_actual = np.zeros(5)
    actual_energy, predicted_energy, times_saved = [], [], []
    actual_spectra, predicted_spectra = [], []
    count = 0
    for case, (history, targets, times) in enumerate(DataLoader(data, batch_size=1)):
        history, times = history.to(device), times.to(device)
        state = history if model.name == 'matey' else (history[:, -1] - mean) / std
        predictions = []
        for lead in range(5):
            if model.name == 'matey':
                physical = model(state, times[:, lead:lead+4], times[:, lead+4])
                state = torch.cat((state[:, 1:], physical[:, None]), dim=1)
            else:
                dt = ((times[:, lead+4] - times[:, lead+3]) / index.dt_reference).to(state.dtype)
                state = model(state, dt)
                physical = state * std + mean
            if not torch.isfinite(physical).all():
                raise FloatingPointError(f'Nonfinite forecast at lead {lead+1}')
            predictions.append(physical[0].cpu().numpy())
        predicted, actual = np.asarray(predictions), targets[0].numpy()
        error = predicted.astype(np.float64) - actual
        total_error += np.sum(error**2, axis=(1,2,3,4))
        total_norm += np.sum((error / np.asarray(stats['std'])[None,:,None,None,None])**2, axis=(1,2,3,4))
        total_actual += np.sum(actual.astype(np.float64)**2, axis=(1,2,3,4))
        count += actual[0].size
        origin = history[0,-1].cpu().numpy()
        a = [energy_spectrum(v) for v in np.concatenate((origin[None], actual))]
        p = [energy_spectrum(v) for v in np.concatenate((origin[None], predicted))]
        actual_energy.append([v[0] for v in a]); predicted_energy.append([v[0] for v in p])
        actual_spectra.append([v[1] for v in a]); predicted_spectra.append([v[1] for v in p])
        times_saved.append(times[0,3:].cpu().numpy())
        if case == 0 and output is not None:
            # Keep only the plotted mid-z slices in home, not full 3D fields.
            z = actual.shape[-1] // 2
            np.savez_compressed(Path(output)/'first-forecast-slices.npz',
                actual=actual[:,:,:,:,z], predicted=predicted[:,:,:,:,z],
                times=times_saved[-1])
    horizons = [dict(lead_steps=i+1, physical_mse=float(total_error[i]/count),
        normalized_mse=float(total_norm[i]/count),
        relative_l2=float(np.sqrt(total_error[i]/total_actual[i])) if total_actual[i] else None,
        target_kinetic_energy=float(np.mean(actual_energy, axis=0)[i+1]),
        predicted_kinetic_energy=float(np.mean(predicted_energy, axis=0)[i+1])) for i in range(5)]
    report = dict(model=model.name + (' looped' if model.looped else ' control'),
        samples=len(data), horizons=horizons, mean_normalized_mse=float(total_norm.mean()/count),
        test_starts=starts)
    arrays = dict(lead_steps=np.arange(6), times=np.asarray(times_saved),
        actual_energy=np.asarray(actual_energy), predicted_energy=np.asarray(predicted_energy),
        actual_spectrum=np.mean(actual_spectra, axis=0), predicted_spectrum=np.mean(predicted_spectra, axis=0),
        modes=np.arange(len(actual_spectra[0][0])), grid_side=np.asarray(cfg['model']['grid_size']))
    if output is not None:
        np.savez_compressed(Path(output)/'energy-data.npz', **arrays)
        plot_energy(report, arrays, output)
        write_json(Path(output)/'test.json', report)
        plot_contours(output)
    return report


def plot_contours(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    with np.load(output/'first-forecast-slices.npz') as data:
        actual = np.linalg.norm(data['actual'].astype(float), axis=1)
        predicted = np.linalg.norm(data['predicted'].astype(float), axis=1)
    error = predicted - actual
    vmax = max(actual.max(), predicted.max(), 1e-12)
    emax = max(np.abs(error).max(), 1e-12)
    fig, axes = plt.subplots(3, 5, figsize=(15,9), constrained_layout=True)
    for row, (values, label) in enumerate(((actual,'DNS'),(predicted,'Prediction'),(error,'Speed error'))):
        for col in range(5):
            image = axes[row,col].imshow(values[col].T, origin='lower', cmap='RdBu_r' if row==2 else 'viridis',
                vmin=-emax if row==2 else 0, vmax=emax if row==2 else vmax)
            axes[row,col].set_title(f'{label}: lead {col+1}')
        fig.colorbar(image, ax=list(axes[row]), shrink=.7)
    fig.savefig(output/'velocity-contours.png', dpi=160); plt.close(fig)


def plot_learning(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    rows = [json.loads(line) for line in (output/'metrics.jsonl').read_text().splitlines()]
    fig, ax = plt.subplots(figsize=(8,4))
    for key, label in [('training_loss','Training objective'),('final_mse','Training final-loop MSE')]:
        ax.plot([r['epoch'] for r in rows], [r[key] for r in rows], label=label)
    ax.plot([r['epoch'] for r in rows], [r['validation']['mean_normalized_mse'] for r in rows], label='Validation final-loop MSE')
    ax.set(xlabel='Epoch', ylabel='Normalized MSE'); ax.legend(); fig.tight_layout()
    fig.savefig(output/'learning-curves.png', dpi=160); plt.close(fig)


def profile(case, cfg, device, output):
    """Disposable three-update full-size gate, including backward and AdamW."""
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    name, variant = case.split('-')
    seed_all(cfg['seed'])
    index = CacheIndex(cfg); stats = training_statistics(index)
    training, validation = datasets(cfg, name, index)
    batch_size = cfg['training']['batch_size']
    loader = DataLoader(training, batch_size=batch_size, shuffle=False)
    batch = next(iter(loader))
    model = FinestStageForecast(name, cfg, index.dt_reference, looped=variant=='looped').to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['training']['learning_rate'], weight_decay=cfg['training']['weight_decay'])
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    seconds, losses = [], []
    for _ in range(3):
        synchronize(device); start = time.perf_counter()
        losses.append(update(model, batch, optimizer, stats, cfg, device))
        synchronize(device); seconds.append(time.perf_counter()-start)
    start = time.perf_counter()
    validation_sample = Subset(validation, range(min(batch_size, len(validation))))
    validation_sample.horizon = 5
    val = validate(model, DataLoader(validation_sample, batch_size=batch_size), stats, device)
    synchronize(device); val_seconds = time.perf_counter()-start
    start = time.perf_counter()
    timing_forecast = common_evaluation(model, cfg, index, stats, device, limit=1, split='val')
    synchronize(device); test_seconds = time.perf_counter()-start
    train_seconds = max(seconds[1:])
    # Headroom for variable I/O and runtime; include validation, 353-case test,
    # checkpoints/plots, and 30 minutes fixed reserve. It is an estimate, not a guarantee.
    projected = cfg['training']['epochs'] * (math.ceil(len(training)/batch_size)*train_seconds
                + math.ceil(len(validation)/batch_size)*val_seconds)
    test_cases = len(HistoryDataset(index, 'test', cross_file=True,
        dt_rtol=cfg['data']['dt_rtol'], dt_atol=cfg['data']['dt_atol']))
    request_minutes = math.ceil((1.35*(projected + test_cases*test_seconds) + 1800) / 60)
    peak = torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None
    total = torch.cuda.get_device_properties(device).total_memory if device.type=='cuda' else None
    accepted = total is None or peak < .9*total
    report = dict(case=case, config=cfg, normalization=stats, cache_fingerprint=index.fingerprint,
        source_fingerprint=source_fingerprint(),
        dt_reference=index.dt_reference, parameters=sum(p.numel() for p in model.parameters()),
        training_windows=len(training), validation_windows=len(validation), test_windows=test_cases,
        update_seconds=seconds, validation_batch_seconds=val_seconds, test_case_seconds=test_seconds,
        peak_cuda_memory_bytes=peak, total_cuda_memory_bytes=total, accepted=accepted,
        requested_minutes=request_minutes, estimated_training_seconds=projected,
        losses=losses, validation=val, timing_forecast=timing_forecast,
        torch=str(torch.__version__), cuda_device=torch.cuda.get_device_name(device) if device.type=='cuda' else None)
    write_json(output/'profile.json', report)
    if not accepted:
        raise RuntimeError('Memory gate exceeded 90% of device capacity; training not authorized by gate')
    return report


def train(case, cfg, device, output, scratch, gate):
    output, scratch = Path(output), Path(scratch)
    if not gate['accepted'] or gate['case'] != case or gate['config'] != cfg:
        raise ValueError('Training configuration does not match accepted profile')
    if gate['source_fingerprint'] != source_fingerprint():
        raise ValueError('Model/training source changed after profiling; profile again')
    if gate['requested_minutes'] > 1440:
        raise ValueError('Estimated request exceeds 24 hours; prepare segmented training first')
    if device.type == 'cuda' and (gate['peak_cuda_memory_bytes'] is None
            or gate['peak_cuda_memory_bytes'] >= .9*torch.cuda.get_device_properties(device).total_memory):
        raise ValueError('Assigned GPU does not provide the memory headroom measured by the gate')
    output.mkdir(parents=True, exist_ok=False); scratch.mkdir(parents=True, exist_ok=False)
    name, variant = case.split('-')
    seed_all(cfg['seed']); index = CacheIndex(cfg)
    if index.fingerprint != gate['cache_fingerprint']:
        raise ValueError('Cache changed after profiling')
    stats = gate['normalization']
    training, validation = datasets(cfg, name, index)
    loader = DataLoader(training, batch_size=cfg['training']['batch_size'], shuffle=True,
        generator=torch.Generator().manual_seed(cfg['seed']))
    val_loader = DataLoader(validation, batch_size=cfg['training']['batch_size'])
    model = FinestStageForecast(name, cfg, index.dt_reference, looped=variant=='looped').to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['training']['learning_rate'], weight_decay=cfg['training']['weight_decay'])
    write_json(output/'experiment.json', dict(case=case, config=cfg, profile=gate,
        architecture=ARCHITECTURE, scratch=str(scratch),
        treatment='finest-stage sharing + headwise sigmoid gates + equal auxiliary/final MSE' if model.looped else 'original model, final-only MSE'))
    best = float('inf'); updates = 0
    for epoch in range(1, cfg['training']['epochs']+1):
        start = time.perf_counter(); totals = dict(training_loss=0., auxiliary_mse=0., final_mse=0.); count=0
        if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
        for batch in loader:
            record = update(model, batch, optimizer, stats, cfg, device)
            size = batch[0].shape[0]; count += size; updates += 1
            for key in totals: totals[key] += record[key]*size
        validation_report = validate(model, val_loader, stats, device)
        synchronize(device)
        row = dict(epoch=epoch, optimizer_updates=updates, seconds=time.perf_counter()-start,
            **{key:value/count for key,value in totals.items()}, validation=validation_report,
            peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None)
        with (output/'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        print(json.dumps(row, allow_nan=False), flush=True)
        saved = dict(format_version=1, architecture=ARCHITECTURE, case=case, config=cfg,
            normalization=stats, dt_reference=index.dt_reference, cache_fingerprint=index.fingerprint,
            epoch=epoch, optimizer_updates=updates, model=model.state_dict(), optimizer=optimizer.state_dict(),
            validation=validation_report, torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(device) if device.type=='cuda' else None)
        improved = validation_report['mean_normalized_mse'] < best
        for filename in (['best.pt'] if improved else []) + ['last.pt']:
            temporary = scratch/(filename+'.tmp'); torch.save(saved, temporary); temporary.replace(scratch/filename)
        if improved: best = validation_report['mean_normalized_mse']
        plot_learning(output)
    saved = torch.load(scratch/'best.pt', map_location='cpu', weights_only=False)
    restore_weights(model, saved['model'])
    report = common_evaluation(model, cfg, index, stats, device, output)
    write_json(output/'complete.json', dict(case=case, epochs=cfg['training']['epochs'], best_epoch=saved['epoch'],
        best_validation=best, test=report, best_checkpoint=str(scratch/'best.pt'), last_checkpoint=str(scratch/'last.pt')))
    (output/'README.md').write_text(f'# {case}\n\n25-epoch finest-stage looping diagnostic.\n\nWeights: {scratch}\n\n'
        'See experiment.json for the exact treatment and profile. Test uses only final loop predictions. '
        'Energy spectra, kinetic energy, speed contours and learning curves are saved here. '
        'This compares a combined treatment, so improvements cannot be attributed to sharing alone.\n')


def summarize(root):
    root = Path(root); reports={}
    for case in CASES:
        path=root/case/'complete.json'
        if not path.exists(): raise RuntimeError(f'Incomplete experiment: missing {case}')
        reports[case]=json.loads(path.read_text())
    reference=reports[CASES[0]]['test']['test_starts']
    if any(r['test']['test_starts'] != reference for r in reports.values()):
        raise ValueError('Test origins differ')
    with (root/'comparison.csv').open('w', newline='') as stream:
        writer=csv.writer(stream); writer.writerow(['case','lead','normalized_mse','physical_mse','relative_l2','predicted_K','actual_K'])
        for case, report in reports.items():
            for row in report['test']['horizons']:
                writer.writerow([case,row['lead_steps'],row['normalized_mse'],row['physical_mse'],row['relative_l2'],
                                 row['predicted_kinetic_energy'],row['target_kinetic_energy']])
    write_json(root/'comparison.json', reports)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('profile','train','summarize'))
    parser.add_argument('--case', choices=CASES)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--scratch', type=Path)
    parser.add_argument('--device', default='cuda')
    args=parser.parse_args()
    if args.action=='summarize': return summarize(args.root)
    if args.case is None: parser.error('--case is required')
    cfg=config_for(args.case.split('-')[0]); device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    try:
        if args.action=='profile':
            profile(args.case, cfg, device, args.root/'profiles'/args.case)
        else:
            if args.scratch is None: parser.error('--scratch is required for training')
            gate=json.loads((args.root/'profiles'/args.case/'profile.json').read_text())
            train(args.case,cfg,device,args.root/args.case,args.scratch/args.case,gate)
    except Exception as error:
        args.root.mkdir(parents=True,exist_ok=True)
        write_json(args.root/f'{args.case}-{args.action}-failure.json',dict(type=type(error).__name__,message=str(error)))
        raise


if __name__=='__main__':
    main()
