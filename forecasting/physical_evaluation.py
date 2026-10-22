"""Checkpoint-only, common-origin turbulence diagnostics. See docs/rollout-statistics.md."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from baseline_vit.dataset import CacheIndex
from baseline_vit.engine import normalization_tensors
from forecasting.checkpoint import check_cache, read_checkpoint, restore_model
from forecasting.dataset import CrossFileSequenceDataset
from forecasting.energy import energy_spectrum, spectral_geometry


def autocorrelation(fields, cuts=(4, 12), max_lag=50):
    """Exact spatial dot products via Parseval; average all valid time pairs.

    Removes each frame's spatial mean. Zero-power bands return NaN, not 1.
    Chunked time FFT avoids storing pairwise space/time products.
    """
    count, _, n, _, _ = fields.shape
    max_lag = min(max_lag, count - 1)
    modes = np.fft.fftfreq(n) * n
    radius = np.sqrt(modes[:, None, None]**2 + modes[None, :, None]**2
                     + (np.fft.rfftfreq(n) * n)[None, None, :]**2)
    _, weights, _ = spectral_geometry(n)
    weights = np.broadcast_to(weights, radius.shape).ravel()
    ft = np.fft.rfftn(fields, axes=(-3, -2, -1)).reshape(count, 3, -1)
    result = []
    for low, high in zip((0, *cuts), (*cuts, np.inf)):
        ids = np.flatnonzero((radius.ravel() > low) & (radius.ravel() <= high))
        numerator = np.zeros(max_lag + 1)
        power = np.zeros(count)
        for begin in range(0, len(ids), 2048):
            selected = ids[begin:begin + 2048]
            z = ft[:, :, selected] * np.sqrt(weights[selected])[None, None, :]
            power += np.sum(np.abs(z)**2, axis=(1, 2))
            temporal = np.fft.fft(z, n=2 * count, axis=0)
            numerator += np.fft.ifft(np.abs(temporal)**2, axis=0).real[:max_lag + 1].sum(axis=(1, 2))
        denominator = np.array([np.sqrt(power[:count-lag].sum() * power[lag:].sum())
                                for lag in range(max_lag + 1)])
        result.append(np.divide(numerator, denominator, out=np.full_like(numerator, np.nan),
                                where=denominator > 0))
    return np.array(result)


def increment_pdf(field, separations, edges):
    """Signed longitudinal increments; pool all cells and three axial directions.

    Fixed physical-unit bins shared by every model/reference. Tail probability
    is retained separately; finite-range densities are NOT renormalized.
    """
    pdf, tails = [], []
    for separation in separations:
        values = np.concatenate([(np.roll(field[c], -separation, axis=c) - field[c]).ravel()
                                 for c in range(3)])
        hist = np.histogram(values, edges)[0]
        pdf.append(hist / (values.size * np.diff(edges)))
        tails.append([np.mean(values < edges[0]), np.mean(values > edges[-1])])
    return np.array(pdf), np.array(tails)


def metadata(path, name):
    saved = (torch.load(path, map_location='cpu', weights_only=True) if name == 'matey'
             else read_checkpoint(path, 'yalamanchi_vit'))
    if name == 'matey':
        if (saved.get('architecture') != 'matey_checkpointed_rollout5_v1'
                or saved.get('variant') != 'baseline' or saved.get('world_size') != 1):
            raise ValueError('Requires full-rollout single-GPU MATEY weights, not pushforward/RingX')
    elif saved['config']['training']['objective'] != 'rollout':
        raise ValueError('Requires rollout-trained Yalamanchi weights')
    return saved


def normalization_matches(first, second):
    """Allow reduction roundoff only; normalization provenance stays exact."""
    if first.keys() != second.keys():
        return False
    for key in first:
        if key in ('mean', 'std'):
            a, b = np.asarray(first[key]), np.asarray(second[key])
            if (a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all()
                    or not np.allclose(a, b, rtol=1e-12, atol=1e-15)):
                return False
            if key == 'std' and (np.any(a <= 0) or np.any(b <= 0)):
                return False
        elif first[key] != second[key]:
            return False
    return True


def prepare(args):
    output = Path(args.output)
    if output.exists():
        raise ValueError('Use a new output directory')
    saved = {name: metadata(getattr(args, name), name) for name in ('matey', 'yalamanchi')}
    first, second = saved.values()
    for key in ('cache_fingerprint', 'dt_reference'):
        if first[key] != second[key]:
            raise ValueError(f'Checkpoint mismatch: {key}')
    if not normalization_matches(first['normalization'], second['normalization']):
        raise ValueError('Checkpoint mismatch: normalization')
    if first['config']['data'] != second['config']['data']:
        raise ValueError('Checkpoint data/splits differ')
    index = CacheIndex(first['config']); check_cache(first, index)
    side = first['config']['model']['grid_size']
    if not 0 < args.cuts[0] < args.cuts[1] < side / 2:
        raise ValueError('Band cuts must increase between zero and axial Nyquist')
    if any(r < 1 or r > side // 2 for r in args.separations):
        raise ValueError('Increment separations must be between 1 and half the grid size')
    data = CrossFileSequenceDataset(index, 'test', args.horizon + 3)
    if args.origins < 1 or args.horizon < 2 or args.origins > len(data):
        raise ValueError('Invalid horizon or number of origins')
    starts = np.linspace(0, len(data)-1, args.origins, dtype=int).tolist()
    spec = dict(horizon=args.horizon, starts=starts, cuts=args.cuts,
                separations=args.separations, max_lag=min(50, args.horizon-1),
                checkpoints={n: str(Path(getattr(args, n)).resolve()) for n in saved},
                epochs={n: s['epoch'] for n, s in saved.items()},
                cache_fingerprint=index.fingerprint, normalization=first['normalization'],
                normalization_by_model={n: s['normalization'] for n, s in saved.items()},
                frames=data.frames, times=data.times.tolist())
    output.mkdir(parents=True)
    (output / 'spec.json').write_text(json.dumps(spec, indent=2))


def worker(args):
    out = Path(args.output); spec = json.loads((out / 'spec.json').read_text())
    dest = out / args.model; dest.mkdir()
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA required by this job')
    saved = metadata(spec['checkpoints'][args.model], args.model)
    if args.model == 'matey':
        from ornl_vit.memory.run import Execution, restore
        model, saved, index = restore(spec['checkpoints'][args.model], Execution('baseline', device))
    else:
        index = CacheIndex(saved['config']); check_cache(saved, index)
        model = restore_model(saved, device, 'yalamanchi_vit')
    model.eval()
    data = CrossFileSequenceDataset(index, 'test', spec['horizon'] + 3)
    if (saved['normalization'] != spec['normalization_by_model'][args.model]
            or saved['epoch'] != spec['epochs'][args.model]
            or index.fingerprint != spec['cache_fingerprint'] or data.times.tolist() != spec['times']
            or [list(x) for x in data.frames] != spec['frames']):
        raise ValueError('Evaluation data changed since preparation')
    mean, std = normalization_tensors(saved['normalization'], device)
    # Exactly common diagnostic scales; each model still uses its own saved
    # normalization for inference, without modifying either checkpoint.
    diagnostic_std = np.asarray(spec['normalization']['std'])
    scale = np.sqrt(np.mean(np.square(diagnostic_std)))
    edges = np.linspace(-12 * scale, 12 * scale, 241)
    selected_leads = sorted(set(x for x in (1, 5, 10, 25, 50, spec['horizon']) if x <= spec['horizon']))
    for case, start in enumerate(spec['starts']):
        refs = data.frames[start:start + spec['horizon'] + 4]
        fields = np.stack([np.array(data._array(f)[t], copy=True) for f, t in refs])
        times = torch.tensor(data.times[start:start + len(refs)], device=device, dtype=torch.float64)[None]
        history = torch.from_numpy(fields[:4]).to(device)[None]
        predictions = []
        with torch.inference_mode():
            state = (history[:, -1] - mean) / std
            for step in range(spec['horizon']):
                if args.model == 'matey':
                    physical = model(history, times[:, step:step+4], times[:, step+4])
                    history = torch.cat((history[:, 1:], physical[:, None]), dim=1)
                else:
                    dt = ((times[:, step+4] - times[:, step+3]) / index.dt_reference).to(state.dtype)
                    state = model(state, dt)
                    physical = state * std + mean
                if not torch.isfinite(physical).all():
                    raise FloatingPointError(f'Nonfinite forecast: origin {case}, lead {step+1}')
                predictions.append(physical[0].cpu().numpy())
        predicted = np.stack(predictions); actual = fields[4:]
        error = predicted.astype(np.float64) - actual
        arrays = dict(mse=np.mean(error**2, axis=(1,2,3,4)),
                      normalized_mse=np.mean((error / diagnostic_std[None,:,None,None,None])**2, axis=(1,2,3,4)),
                      times=data.times[start+4:start+len(refs)], pdf_edges=edges,
                      pdf_leads=selected_leads, separations=spec['separations'])
        for label, sequence in (('actual', actual), ('predicted', predicted)):
            arrays[label + '_spectrum'] = np.array([energy_spectrum(f)[1] for f in sequence])
            arrays[label + '_correlation'] = autocorrelation(sequence, spec['cuts'], spec['max_lag'])
            pdfs = [increment_pdf(sequence[lead-1], spec['separations'], edges) for lead in selected_leads]
            arrays[label + '_pdf'] = np.array([p[0] for p in pdfs])
            arrays[label + '_tails'] = np.array([p[1] for p in pdfs])
            arrays[label + '_speed_slice'] = np.array([np.linalg.norm(sequence[lead-1], axis=0)[:, :, sequence.shape[-1]//2] for lead in selected_leads])
        arrays['lag_times'] = np.array([0.] + [np.mean(arrays['times'][lag:] - arrays['times'][:-lag]) for lag in range(1, spec['max_lag']+1)])
        np.savez_compressed(dest / f'origin-{case:03d}.npz', **arrays)
        print(f'{args.model}: origin {case+1}/{len(spec["starts"])} complete', flush=True)
    (dest / 'complete.json').write_text(json.dumps(dict(checkpoint=spec['checkpoints'][args.model], epoch=saved['epoch'], origins=len(spec['starts']))))


def summarize(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(args.output); spec = json.loads((out / 'spec.json').read_text())
    results = {}
    for name in ('matey', 'yalamanchi'):
        if not (out / name / 'complete.json').exists():
            raise RuntimeError(f'{name} incomplete; completed origin files retained')
        cases = []
        for i in range(len(spec['starts'])):
            with np.load(out / name / f'origin-{i:03d}.npz') as z:
                cases.append({k: z[k] for k in z.files})
        results[name] = {k: np.stack([c[k] for c in cases]) for k in cases[0]}
    a, b = results.values()
    for key in ('times', 'actual_spectrum', 'actual_pdf', 'actual_correlation'):
        if not np.allclose(a[key], b[key], equal_nan=True):
            raise ValueError(f'Reference mismatch: {key}')
    fig, ax = plt.subplots()
    for name, z in results.items():
        y = z['normalized_mse']; x = np.arange(1, y.shape[1]+1)
        ax.plot(x, y.mean(0), label=name)
        ax.fill_between(x, y.min(0), y.max(0), alpha=.15)
    ax.set(xlabel='Forecast step', ylabel='Normalized MSE', title='Mean and range across overlapping origins'); ax.legend()
    fig.tight_layout(); fig.savefig(out / 'mse.png'); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13,4))
    for band, ax in enumerate(axes):
        ax.plot(a['lag_times'].mean(0), a['actual_correlation'][:,band].mean(0), 'k', label='DNS')
        for name,z in results.items():
            ax.plot(z['lag_times'].mean(0), z['predicted_correlation'][:,band].mean(0), label=name)
        ax.set(xlabel='Lag (recorded simulation time)', ylabel='Correlation', title=f'Band {band+1}'); ax.legend()
    fig.tight_layout(); fig.savefig(out / 'temporal-correlation.png'); plt.close(fig)
    leads = a['pdf_leads'][0].astype(int)
    fig, axes = plt.subplots(1, len(leads), figsize=(4*len(leads),4), squeeze=False)
    for lead,ax in zip(leads,axes.flat):
        for label,z,key in [('DNS',a,'actual_spectrum')] + [(n,z,'predicted_spectrum') for n,z in results.items()]:
            nyquist = z['actual_speed_slice'].shape[-1] // 2
            y=z[key][:,lead-1].mean(0); ax.loglog(np.arange(1,nyquist+1), y[1:nyquist+1], label=label)
        ax.set(title=f'Lead {lead}', xlabel='Fourier mode', ylabel='Shell energy'); ax.legend()
    fig.tight_layout(); fig.savefig(out / 'spectra.png'); plt.close(fig)
    fig, axes = plt.subplots(len(spec['separations']),len(leads), figsize=(4*len(leads),3*len(spec['separations'])), squeeze=False)
    centers=(a['pdf_edges'][0,1:]+a['pdf_edges'][0,:-1])/2
    for row,r in enumerate(spec['separations']):
        for col,lead in enumerate(leads):
            ax=axes[row,col]
            for label,z,key in [('DNS',a,'actual_pdf')] + [(n,z,'predicted_pdf') for n,z in results.items()]:
                y=z[key][:,col,row].mean(0); ax.semilogy(centers,np.where(y>0,y,np.nan),label=label)
            ax.set(title=f'Lead {lead}; separation {r} cells', xlabel='Longitudinal increment (physical units)', ylabel='PDF'); ax.legend()
    fig.tight_layout(); fig.savefig(out / 'increment-pdfs.png'); plt.close(fig)
    fig, axes = plt.subplots(3,len(leads),figsize=(3*len(leads),9),squeeze=False)
    slices=[a['actual_speed_slice'][0],a['predicted_speed_slice'][0],b['predicted_speed_slice'][0]]
    vmax=max(float(s.max()) for s in slices)
    for row,(name,values) in enumerate(zip(('DNS','MATEY','Yalamanchi'),slices)):
        for col,lead in enumerate(leads):
            im=axes[row,col].imshow(values[col].T,origin='lower',vmin=0,vmax=vmax)
            axes[row,col].set_title(f'{name}, step {lead}')
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Speed'); fig.savefig(out/'speed-contours.png'); plt.close(fig)
    with (out/'mse.csv').open('w') as f:
        f.write('model,lead,physical_mse,normalized_mse\n')
        for name,z in results.items():
            for i in range(spec['horizon']):
                f.write(f'{name},{i+1},{z["mse"][:,i].mean()},{z["normalized_mse"][:,i].mean()}\n')
    (out/'README.md').write_text('# Rollout statistics\n\n'
        + f'{len(spec["starts"])} common test origins, {spec["horizon"]} autonomous steps. No retraining.\n\n'
        + 'Origins overlap; ranges are descriptive, not independent-sample confidence intervals.\n'
        + 'MATEY uses four observed frames; Yalamanchi uses the last one. This is a checkpoint comparison, not a controlled architecture ablation.\n'
        + 'Autocorrelations use predicted frames only, all valid pairs, excluding spatial mean; they include Eulerian advection.\n'
        + 'PDFs use fixed physical-unit bins; out-of-range tail probabilities are saved in each NPZ.\n'
        + 'See spec.json for checkpoints, epochs, origin indices, times, bands, and separations.\n\n'
        + '\n'.join(f'![{p.stem}]({p.name})\n' for p in sorted(out.glob('*.png'))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare'); p.add_argument('--matey',required=True); p.add_argument('--yalamanchi',required=True)
    p.add_argument('--horizon',type=int,default=100); p.add_argument('--origins',type=int,default=8)
    p.add_argument('--cuts',type=int,nargs=2,default=[4,12]); p.add_argument('--separations',type=int,nargs='+',default=[1,2,4,8,16])
    p.add_argument('--output',required=True)
    p=sub.add_parser('worker'); p.add_argument('--model',choices=['matey','yalamanchi'],required=True)
    p.add_argument('--device',default='cuda'); p.add_argument('--output',required=True)
    p=sub.add_parser('summarize'); p.add_argument('--output',required=True)
    args=parser.parse_args(); globals()[args.command](args)


if __name__ == '__main__':
    main()
