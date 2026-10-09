"""Fresh MATEY / Yalamanchi five-step training and common-origin energy evaluation."""
import argparse
import copy
import csv
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from baseline_vit.dataset import CacheIndex, training_statistics
from baseline_vit.engine import normalization_tensors, select_device
from ornl_vit.config import load_config as load_matey_config
from ornl_vit.dataset import HistoryDataset
from ornl_vit.engine import make_model, normalized_mse, restore, train as train_matey
from ornl_vit.rollout import forecast_five
from yalamanchi_vit.config import load_config as load_yalamanchi_config
from yalamanchi_vit.model import ARCHITECTURE, YalamanchiViT
from .checkpoint import check_cache, read_checkpoint, restore_model
from .compare_training import write_json
from .dataset import SequenceDataset
from .energy import energy_spectrum, plot_comparison, plot_energy
from .rollout import forecast_loss
from .train import train as train_yalamanchi

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('matey', 'yalamanchi')


def common_starts(dataset):
    continuous = dataset.continuous
    return [dict(history_start_time=float(continuous.times[start]),
                 file_id=continuous.frames[start + 3][0], frame=continuous.frames[start + 3][1],
                 time=float(continuous.times[start + 3]), end_time=float(continuous.times[start + 8]))
            for start in dataset.pairs]


def prepare(output, matey_config, yalamanchi_config, device='cuda'):
    matey = load_matey_config(matey_config)
    yala = load_yalamanchi_config(yalamanchi_config)
    if yala['training']['objective'] != 'rollout' or yala['training']['horizon'] != 5:
        raise ValueError('Yalamanchi source must specify five-step rollout MSE')
    if yala['model']['grid_size'] != matey['model']['grid_size']:
        raise ValueError('Both models must use the same field grid')
    # The MATEY config retains the already-approved 107-file experiment subset.
    yala['data'] = copy.deepcopy(matey['data'])
    for key in ('epochs', 'learning_rate', 'weight_decay', 'grad_clip'):
        if yala['training'][key] != matey['training'][key]:
            raise ValueError(f'Configured training {key} differs between models')
    if yala['seed'] != matey['seed']:
        raise ValueError('Seeds must match')
    if matey['training']['max_train_windows'] is not None or yala['training']['max_train_pairs'] is not None:
        raise ValueError('Production comparison uses all within-file windows')
    index = CacheIndex(matey)
    common = HistoryDataset(index, 'test', cross_file=True,
                            dt_rtol=matey['data']['dt_rtol'], dt_atol=matey['data']['dt_atol'])
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    specs = {}
    for name, cfg in [('matey', matey), ('yalamanchi', yala)]:
        cfg['training']['output_dir'] = str(output / name)
        cfg['training']['device'] = device
        (output / f'{name}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
        dataset_class = HistoryDataset if name == 'matey' else SequenceDataset
        kwargs = {} if name == 'matey' else {'horizon': 5}
        specs[name] = dict(model=cfg['model'], training=cfg['training'],
                          training_windows=len(dataset_class(index, 'train', **kwargs)),
                          validation_windows=len(dataset_class(index, 'val', **kwargs)))
    spec = dict(source_configs=dict(matey=str(Path(matey_config).resolve()),
                                   yalamanchi=str(Path(yalamanchi_config).resolve())),
                data=matey['data'], seed=matey['seed'], training_horizon=5, test_horizon=5,
                models=specs, test_windows=len(common), test_starts=common_starts(common),
                cache_fingerprint=index.fingerprint,
                selection='Each model selects best mean five-step validation normalized MSE on its own within-file windows',
                comparison='Same files, epochs, optimizer settings, loss and test origins; differing history, patches, width, windows, batches and updates',
                energy='K=.5*mean_space(u²+v²+w²); includes mean flow; spectra use unit-width integer Fourier-mode shells')
    write_json(output / 'experiment.json', spec)
    print(json.dumps({k: v for k, v in spec.items() if k not in ('data', 'test_starts')}, indent=2), flush=True)


def smoke(cfg, name, device):
    """One actual configured rollout/backward/update before spending the run budget."""
    index = CacheIndex(cfg)
    stats = training_statistics(index)
    dataset = HistoryDataset(index, 'train') if name == 'matey' else SequenceDataset(index, 'train', 5)
    loader = DataLoader(dataset, batch_size=cfg['training']['batch_size'], num_workers=0)
    batch = [x.to(device) for x in next(iter(loader))]
    model = make_model(cfg, index.dt_reference, device) if name == 'matey' else YalamanchiViT(**cfg['model']).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['training']['learning_rate'],
                                  weight_decay=cfg['training']['weight_decay'])
    if device.type == 'cuda':
        torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    if name == 'matey':
        history, targets, times = batch
        loss = normalized_mse(forecast_five(model, history, times), targets, stats)
    else:
        initial, targets, intervals = batch
        mean, std = normalization_tensors(stats, device)
        loss = forecast_loss(model, (initial - mean) / std,
                             (targets - mean[:, None]) / std[:, None], intervals, 'rollout')
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite smoke-test loss')
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['grad_clip'], error_if_nonfinite=True)
    optimizer.step()
    if device.type == 'cuda': torch.cuda.synchronize(device)
    report = dict(model=name, batch_size=cfg['training']['batch_size'], five_step_loss=loss.item(),
                  seconds=time.perf_counter() - started,
                  peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None)
    # Smoke weights are discarded. Each real trainer reseeds and initializes afresh.
    del batch, model, optimizer, loss, loader
    gc.collect()
    if device.type == 'cuda': torch.cuda.empty_cache()
    return report


@torch.no_grad()
def predict(model, name, history, times, stats, dt_reference):
    if name == 'matey':
        return forecast_five(model, history, times)
    if name != 'yalamanchi':
        raise ValueError('Unknown model')
    mean, std = normalization_tensors(stats, history.device)
    state = (history[:, -1] - mean) / std
    predictions = []
    for lead in range(5):
        interval = ((times[:, lead + 4] - times[:, lead + 3]) / dt_reference).to(state.dtype)
        state = model(state, interval)
        physical = state * std + mean
        if not torch.isfinite(physical).all():
            raise FloatingPointError(f'Nonfinite Yalamanchi forecast at lead {lead + 1}')
        predictions.append(physical)
    return torch.stack(predictions, dim=1)


def evaluate_common(checkpoint, name, spec, output, device):
    if name == 'matey':
        model, saved, index = restore(checkpoint, device)
    else:
        saved = read_checkpoint(checkpoint, 'yalamanchi_vit')
        index = CacheIndex(saved['config']); check_cache(saved, index)
        model = restore_model(saved, device, 'yalamanchi_vit')
    if index.fingerprint != spec['cache_fingerprint']:
        raise ValueError('Cache changed after comparison preparation')
    cfg = saved['config']
    if cfg['data'] != spec['data']:
        raise ValueError('Checkpoint data differs from experiment')
    data = HistoryDataset(index, 'test', cross_file=True,
                          dt_rtol=cfg['data']['dt_rtol'], dt_atol=cfg['data']['dt_atol'])
    starts = common_starts(data)
    if starts != spec['test_starts']:
        raise ValueError('Evaluation origins differ from prepared comparison')
    model.eval()
    totals = np.zeros((5, 6), dtype=np.float64)
    count = 0
    std = np.asarray(saved['normalization']['std'], dtype=np.float64).reshape(1, 3, 1, 1, 1)
    actual_energy, predicted_energy, recorded_times = [], [], []
    actual_spectra, predicted_spectra = None, None
    output = Path(output)
    started = time.perf_counter()
    for case, (history, targets, times) in enumerate(DataLoader(data, batch_size=1, shuffle=False, num_workers=0)):
        with torch.no_grad():
            predicted = predict(model, name, history.to(device), times.to(device),
                                saved['normalization'], index.dt_reference)[0].cpu().numpy()
        targets = targets[0].numpy()
        initial = history[0, -1].numpy()
        error = predicted.astype(np.float64) - targets
        sums = [np.sum(error**2, axis=(1,2,3,4)), np.sum((error / std)**2, axis=(1,2,3,4)),
                np.sum(np.abs(error), axis=(1,2,3,4)),
                np.sum((initial.astype(np.float64)[None] - targets)**2, axis=(1,2,3,4)),
                np.sum(predicted.astype(np.float64)**2, axis=(1,2,3,4)),
                np.sum(targets.astype(np.float64)**2, axis=(1,2,3,4))]
        totals += np.stack(sums, axis=1)
        count += initial.size
        actual_fields = np.concatenate((initial[None], targets), axis=0)
        predicted_fields = np.concatenate((initial[None], predicted), axis=0)
        a_energy, p_energy, a_spectra, p_spectra = [], [], [], []
        for actual_field, predicted_field in zip(actual_fields, predicted_fields):
            ae, asp = energy_spectrum(actual_field)
            pe, psp = energy_spectrum(predicted_field)
            a_energy.append(ae); p_energy.append(pe); a_spectra.append(asp); p_spectra.append(psp)
        actual_energy.append(a_energy); predicted_energy.append(p_energy)
        recorded_times.append(times[0, 3:].numpy())
        if actual_spectra is None:
            actual_spectra = np.zeros_like(a_spectra); predicted_spectra = np.zeros_like(p_spectra)
        actual_spectra += a_spectra; predicted_spectra += p_spectra
        if case == 0:
            np.savez_compressed(output / 'first-forecast.npz', history=history[0].numpy(),
                                actual=targets, predicted=predicted, times=times[0].numpy())
        if (case + 1) % 25 == 0:
            print(f'{name}: evaluated {case + 1}/{len(data)} common origins', flush=True)
    metrics = []
    for lead, (error, norm, absolute, persistence, predicted, actual) in enumerate(totals.tolist(), 1):
        metrics.append(dict(lead_steps=lead, samples=len(data), physical_mse=error / count,
            normalized_mse=norm / count, physical_mae=absolute / count,
            relative_l2=float(np.sqrt(error / actual)) if actual else None,
            persistence_mse=persistence / count,
            mse_skill_vs_persistence=1 - error / persistence if persistence else None,
            predicted_kinetic_energy=1.5 * predicted / count, target_kinetic_energy=1.5 * actual / count))
    arrays = dict(lead_steps=np.arange(6), times=np.asarray(recorded_times),
                  actual_energy=np.asarray(actual_energy), predicted_energy=np.asarray(predicted_energy),
                  actual_spectrum=actual_spectra / len(data), predicted_spectrum=predicted_spectra / len(data),
                  modes=np.arange(actual_spectra.shape[-1]), grid_side=np.asarray(cfg['model']['grid_size']))
    report = dict(model=name, checkpoint=str(Path(checkpoint).resolve()), checkpoint_epoch=saved['epoch'],
                  history=4 if name == 'matey' else 1, split='test', test_starts=starts,
                  normalization=saved['normalization'], cache_fingerprint=index.fingerprint,
                  rollout=dict(horizon=5, samples=len(data), horizons=metrics,
                               mean_normalized_mse=float(totals[:,1].mean() / count)),
                  evaluation_seconds=time.perf_counter() - started,
                  energy_definition='K=.5*mean_space(u²+v²+w²), includes mean flow',
                  spectrum_definition='Shell-integrated kinetic energy, nearest integer Fourier mode; all shells saved, m=0 and m>N/2 omitted from plots')
    np.savez_compressed(output / 'energy-data.npz', **arrays)
    with (output / 'energy-trajectories.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['case','origin_time','lead_steps','time','actual_energy','predicted_energy'])
        for case, origin in enumerate(starts):
            for lead in range(6):
                writer.writerow([case, origin['time'], lead, arrays['times'][case,lead],
                                 arrays['actual_energy'][case,lead], arrays['predicted_energy'][case,lead]])
    with (output / 'energy-spectra.csv').open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['lead_steps','mode','actual_energy','predicted_energy'])
        for lead in range(6):
            for mode in arrays['modes']:
                writer.writerow([lead,mode,arrays['actual_spectrum'][lead,mode],arrays['predicted_spectrum'][lead,mode]])
    plot_energy(report, arrays, output)
    # A completion report certifies that plots were generated too.
    write_json(output / 'evaluation-test-5.json', report)
    return report


def worker(output, name):
    output = Path(output).resolve()
    spec = json.loads((output / 'experiment.json').read_text())
    load = load_matey_config if name == 'matey' else load_yalamanchi_config
    cfg = load(output / f'{name}.yaml')
    if (cfg['model'] != spec['models'][name]['model']
            or cfg['training'] != spec['models'][name]['training']
            or cfg['data'] != spec['data'] or cfg['seed'] != spec['seed']):
        raise ValueError('Worker configuration changed after preparation')
    try:
        if CacheIndex(cfg).fingerprint != spec['cache_fingerprint']:
            raise ValueError('Cache changed after experiment preparation')
        device = select_device(cfg['training']['device'])
        write_json(output / f'{name}-smoke.json', smoke(cfg, name, device))
        if name == 'matey':
            train_matey(cfg)
        else:
            train_yalamanchi(cfg, YalamanchiViT, ARCHITECTURE, save_last=True)
        print(f'{name}: training complete, evaluating common five-step test origins', flush=True)
        evaluate_common(output / name / 'best.pt', name, spec, output / name, device)
    except Exception as error:
        write_json(output / f'{name}-failure.json', dict(type=type(error).__name__, message=str(error)))
        raise


def summarize(output):
    output = Path(output).resolve()
    reports, arrays, missing, rows = {}, {}, [], []
    for name in MODELS:
        report_file = output / name / 'evaluation-test-5.json'
        if not report_file.is_file():
            missing.append(name); continue
        reports[name] = json.loads(report_file.read_text())
        with np.load(output / name / 'energy-data.npz', allow_pickle=False) as data:
            arrays[name] = {key:data[key].copy() for key in data.files}
        rows.extend(dict(model=name, checkpoint_epoch=reports[name]['checkpoint_epoch'], **lead)
                    for lead in reports[name]['rollout']['horizons'])
    if not missing:
        a, b = [reports[name] for name in MODELS]
        if (a['test_starts'] != b['test_starts'] or a['normalization'] != b['normalization']
                or a['cache_fingerprint'] != b['cache_fingerprint']):
            raise ValueError('Cannot compare different origins, normalization or caches')
        for key in ('times', 'actual_energy', 'actual_spectrum'):
            if not np.array_equal(arrays['matey'][key], arrays['yalamanchi'][key]):
                raise ValueError(f'Actual evaluation data differ: {key}')
        plot_comparison(reports, arrays, output)
    write_json(output / 'comparison.json', dict(complete=not missing, missing=missing, results=reports))
    if rows:
        with (output / 'comparison.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    if missing:
        raise RuntimeError('Incomplete comparison; available results retained: ' + ', '.join(missing))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'worker', 'evaluate', 'summarize'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda',
                        help='Device written by prepare; CPU is for explicit small diagnostics')
    parser.add_argument('--matey-config', type=Path, default=ROOT / 'ornl_vit/config.yaml')
    parser.add_argument('--yalamanchi-config', type=Path, default=ROOT / 'yalamanchi_vit/config-runtime.yaml')
    args = parser.parse_args(argv)
    if args.action == 'prepare':
        prepare(args.output, args.matey_config, args.yalamanchi_config, args.device)
    elif args.action == 'summarize':
        summarize(args.output)
    else:
        if args.model is None: parser.error(f'{args.action} requires --model')
        if args.action == 'worker':
            worker(args.output, args.model)
        else:
            spec = json.loads((args.output / 'experiment.json').read_text())
            evaluate_common(args.output / args.model / 'best.pt', args.model, spec,
                            args.output / args.model, select_device(args.device))


if __name__ == '__main__':
    main()
