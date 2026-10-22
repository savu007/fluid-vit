"""Inspect the existing MATEY hierarchy without changing predictions or weights."""
import argparse
import csv
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from forecasting.dataset import CrossFileSequenceDataset
from forecasting.physical_evaluation import metadata
from .run import Execution, restore


@contextmanager
def capture_hierarchy(core):
    """Temporary observational wrappers. Preserve original instance attributes."""
    names = ('forward', 'get_spatiotemporalfromsequence', 'upsampeldata')
    originals = {name: getattr(core, name) for name in names}
    instance = {name: core.__dict__[name] for name in names if name in core.__dict__}
    records = {}

    def forward(*args, **kwargs):
        opts = kwargs.get('opts', args[3] if len(args) > 3 else None)
        level = int(opts.imod)  # opts is mutated during recursive calls.
        out = originals['forward'](*args, **kwargs)
        records.setdefault(level, {})['returned'] = out.detach()
        return out

    def decoded(*args, **kwargs):
        out = originals['get_spatiotemporalfromsequence'](*args, **kwargs)
        level = kwargs.get('ilevel', args[5] if len(args) > 5 else 0)
        records.setdefault(int(level), {})['candidate'] = out[-1].detach()
        return out

    def upsample(data, imod):
        out = originals['upsampeldata'](data, imod)
        records.setdefault(int(imod), {}).setdefault('upsamples', []).append(out.detach())
        return out

    core.forward, core.get_spatiotemporalfromsequence, core.upsampeldata = forward, decoded, upsample
    try:
        yield records
    finally:
        for name in names:
            if name in instance:
                setattr(core, name, instance[name])
            else:
                delattr(core, name)


def correction_scores(learned, required):
    """Signed least-squares alpha is diagnostic only; it never alters rollout."""
    c, d = np.asarray(learned, dtype=np.float64), np.asarray(required, dtype=np.float64)
    if c.shape != d.shape or not np.isfinite(c).all() or not np.isfinite(d).all():
        raise ValueError('Need matching finite correction fields')
    cc, dd, cd = np.sum(c*c), np.sum(d*d), np.sum(c*d)
    return dict(learned_rms=float(np.sqrt(cc/c.size)), required_rms=float(np.sqrt(dd/c.size)),
                rms_ratio=float(np.sqrt(cc/dd)) if dd else None,
                cosine=float(np.clip(cd/np.sqrt(cc*dd), -1, 1)) if cc and dd else None,
                oracle_alpha=float(cd/cc) if cc else None,
                projection_fraction=float(cd/dd) if dd else None,
                mse_before=float(dd/c.size), mse_after=float(np.mean((c-d)**2)),
                error_reduction=float(1-np.sum((c-d)**2)/dd) if dd else None)


@torch.no_grad()
def decompose(core, records, history, target):
    """Return exact branch components and level-relative physical-unit views.

    Internal coarse returns are not denormalized by upstream. Intermediate
    physical views use that level's input-history mean/std only for diagnostics.
    At the finest level the mapping is the actual final denormalization.
    """
    from ..matey.data_utils.shared_utils import normalize_spatiotemporal_persample
    histories = {core.nhlevels-1: history.transpose(0, 1)}
    targets = {core.nhlevels-1: target}
    for level in range(core.nhlevels-2, -1, -1):
        histories[level] = core.filterdata(histories[level+1])[0]
        targets[level] = core.filterdata(targets[level+1][None])[0][0]
    fields, rows = {}, []
    for level in range(core.nhlevels):
        record = records[level]
        _, mean, std = normalize_spatiotemporal_persample(histories[level])
        mean, std = mean[0], std[0]
        candidate = record['candidate']
        if level == 0:
            if record.get('upsamples'):
                raise ValueError('Unexpected coarsest upsample')
            torch.testing.assert_close(candidate, record['returned'], rtol=0, atol=0)
            predicted = candidate * std + mean
            fields[level] = dict(target=targets[level], predicted=predicted)
            continue
        upsamples = record.get('upsamples', [])
        if len(upsamples) != 2:
            raise ValueError('Upstream correction path changed: expected two upsampling calls')
        filtered_candidate, base = upsamples
        correction = candidate-filtered_candidate
        combined = correction+base
        expected = combined*std+mean if level == core.nhlevels-1 else combined
        torch.testing.assert_close(expected, record['returned'], rtol=0, atol=0)
        physical_base = base*std+mean
        physical_correction = correction*std
        required = targets[level]-physical_base
        # Preserve actual arithmetic for output; decomposition sums can differ by roundoff.
        predicted = combined*std+mean
        tensors = dict(target=targets[level], base=physical_base,
                       learned=physical_correction, required=required, predicted=predicted,
                       candidate_internal=candidate, removed_internal=filtered_candidate,
                       base_internal=base, correction_internal=correction,
                       mean=mean, std=std)
        fields[level] = tensors
        row = correction_scores(physical_correction.cpu().numpy(), required.cpu().numpy())
        row.update(level=level, grid_side=target.shape[-1]//core.filtersize**(core.nhlevels-1-level),
                   actual_prediction_mse=float((predicted.double()-targets[level].double()).square().mean()))
        rows.append(row)
    return fields, rows


def render_report(output, rows, example):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    levels = sorted({row['level'] for row in rows})
    fig, axes = plt.subplots(1, 3, figsize=(14,4), constrained_layout=True)
    for level in levels:
        selected = [r for r in rows if r['level'] == level]
        leads = sorted({r['lead'] for r in selected})
        for ax,key,title in zip(axes, ('rms_ratio','cosine','error_reduction'),
                               ('Correction RMS / required RMS','Directional alignment','Fraction of coarse error removed')):
            values = np.array([[next(r[key] for r in selected if r['origin'] == origin and r['lead'] == lead)
                                for lead in leads] for origin in sorted({r['origin'] for r in selected})], dtype=float)
            ax.plot(leads, values.mean(0), 'o-', label=f'Level {level}')
            ax.fill_between(leads, values.min(0), values.max(0), alpha=.15)
            ax.set(title=title, xlabel='Forecast lead'); ax.grid(alpha=.2); ax.legend()
    axes[0].axhline(1,c='k',lw=.7);axes[1].axhline(0,c='k',lw=.7);axes[2].axhline(0,c='k',lw=.7)
    fig.savefig(output/'correction-metrics.png',dpi=160);plt.close(fig)
    for level in levels:
        cases = sorted((lead, fields[level]) for lead, fields in example.items())
        # One signed velocity component reveals direction as well as amplitude.
        keys = ('target','base','learned','required','predicted')
        limits = {key:max(float(np.abs(fields[key][0,0,:,:,fields[key].shape[-1]//2]).max())
                          for _,fields in cases) for key in keys}
        limit = max(limits[k] for k in ('target','base','predicted'))
        correction_limit = max(limits['learned'],limits['required'],1e-12)
        fig,axes=plt.subplots(5,len(cases),figsize=(3*len(cases),13),squeeze=False,constrained_layout=True)
        for col,(lead,fields) in enumerate(cases):
            for row,key in enumerate(keys):
                a=fields[key][0,0,:,:,fields[key].shape[-1]//2]
                bound=correction_limit if key in ('learned','required') else limit
                im=axes[row,col].imshow(a.T,origin='lower',cmap='RdBu_r',vmin=-bound,vmax=bound,interpolation='nearest')
                axes[row,col].set_title(f'{key}, lead {lead}',fontsize=11)
                axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
                if col==len(cases)-1:fig.colorbar(im,ax=list(axes[row]),shrink=.6)
        fig.suptitle(f'First origin; level {level}; u component; learned/required use identical limits')
        fig.savefig(output/f'level-{level}-corrections.png',dpi=150);plt.close(fig)
    text = ('# MATEY correction diagnostic\n\n'
            'No training, rescaling, or target feedback. Same saved checkpoint throughout.\n\n'
            'RMS ratio measures strength; cosine measures alignment; error reduction measures usefulness. '
            'An RMS ratio below one alone does not establish under-correction. '
            'oracle_alpha is a signed least-squares multiplier calculated using ground truth, not a deployable fix.\n\n'
            'Intermediate-level physical-unit views apply that level\'s input-history mean/std. '
            'Upstream only denormalizes the finest return; intermediate views are diagnostic mappings, '
            'not separately emitted physical forecasts. Required correction is target minus this level\'s '
            'upsampled base, not the true high-pass component or an independently supervised target.\n\n'
            'Ranges across overlapping origins are descriptive, not confidence intervals. '
            'Autonomous rollout uses predictions only after the initial four observed frames.\n\n')
    text += '\n'.join(f'![{p.stem}]({p.name})\n' for p in sorted(output.glob('*.png')))
    (output/'README.md').write_text(text)
    images=''.join(f'<h2>{p.stem}</h2><img style="max-width:100%" src="data:image/png;base64,'
                   +__import__('base64').b64encode(p.read_bytes()).decode()+'">' for p in sorted(output.glob('*.png')))
    (output/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>MATEY corrections</title>'
        '<h1>MATEY correction diagnostic</h1><p>Checkpoint-only autonomous rollout. See README.md for definitions and caveats.</p>'+images)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True);parser.add_argument('--origins',type=int,default=8)
    parser.add_argument('--horizon',type=int,default=5);parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    if args.origins<1 or args.horizon<1:raise ValueError('origins and horizon must be positive')
    if args.output.exists() or args.report.exists():raise ValueError('Use new output/report directories')
    args.output.mkdir(parents=True);args.report.mkdir(parents=True)
    device=torch.device(args.device)
    rows=[];examples={}
    try:
        saved=metadata(args.checkpoint,'matey')
        model,saved,index=restore(args.checkpoint,Execution('baseline',device));model.eval()
        cfg=saved['config']['data']
        data=CrossFileSequenceDataset(index,'test',args.horizon+3,dt_rtol=cfg['dt_rtol'],dt_atol=cfg['dt_atol'])
        if args.origins>len(data):raise ValueError('Too many origins')
        starts=np.linspace(0,len(data)-1,args.origins,dtype=int).tolist()
        spec=dict(checkpoint=str(Path(args.checkpoint).resolve()),epoch=saved['epoch'],
                  starts=starts,horizon=args.horizon,cache_fingerprint=index.fingerprint,
                  source_frames=[data.frames[s:s+args.horizon+4] for s in starts],
                  times=[data.times[s:s+args.horizon+4].tolist() for s in starts],
                  objective=saved['config']['training']['objective'])
        (args.report/'spec.json').write_text(json.dumps(spec,indent=2))
        with torch.inference_mode(),capture_hierarchy(model.core) as records:
            for origin,start in enumerate(starts):
                refs=data.frames[start:start+args.horizon+4]
                fields=np.stack([np.array(data._array(f)[t],copy=True) for f,t in refs])
                if not np.isfinite(fields).all():raise ValueError('Nonfinite source data')
                times=torch.tensor(data.times[start:start+len(refs)],device=device)[None]
                history=torch.from_numpy(fields[:4]).to(device)[None]
                for step in range(args.horizon):
                    records.clear()
                    prediction=model(history,times[:,step:step+4],times[:,step+4])
                    target=torch.from_numpy(fields[step+4]).to(device)[None]
                    components,scores=decompose(model.core,records,history,target)
                    snapshots={level:{k:v.float().cpu().numpy() for k,v in values.items()}
                               for level,values in components.items()}
                    for row in scores:row.update(origin=origin,lead=step+1);rows.append(row)
                    packed={f'level{level}_{key}':value for level,values in snapshots.items() for key,value in values.items()}
                    np.savez_compressed(args.output/f'origin-{origin:03d}-lead-{step+1:03d}.npz',**packed)
                    if origin==0:examples[step+1]=snapshots
                    history=torch.cat((history[:,1:],prediction[:,None]),dim=1)
                (args.report/'metrics.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
                print(f'Completed origin {origin+1}/{args.origins}',flush=True)
        with (args.report/'metrics.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        # Small slices are portable; full component arrays remain in scratch.
        np.savez_compressed(args.report/'first-origin-slices.npz',**{
            f'lead{lead}_level{level}_{key}':value[...,value.shape[-1]//2]
            for lead,levels in examples.items() for level,values in levels.items()
            for key,value in values.items() if value.ndim==5 and value.shape[-1]>1})
        render_report(args.report,rows,examples)
        (args.report/'complete.json').write_text(json.dumps(dict(origins=args.origins,horizon=args.horizon,
            checkpoint=args.checkpoint,epoch=saved['epoch'],full_arrays=str(args.output)),indent=2))
    except Exception as exc:
        (args.report/'failure.json').write_text(json.dumps(dict(error=repr(exc))))
        raise


if __name__=='__main__':main()
