"""Isolated Yalamanchi MSE + kinetic-energy experiment; compare to old slides."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from baseline_vit.dataset import CacheIndex, training_statistics
from baseline_vit.engine import normalization_tensors, select_device
from forecasting.checkpoint import check_cache, read_checkpoint, restore_model
from forecasting.compare_training import write_json
from forecasting.dataset import CrossFileSequenceDataset, SequenceDataset
from forecasting.presentation import forecast_window, render_deck
from .energy_loss import energy_floor, rollout_loss
from .model import ARCHITECTURE, YalamanchiViT

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'runs/validation-slides-59353752/report.json'
WEIGHTS = {'energy-1': 1., 'energy-10': 10.}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(output, reference_report=REFERENCE, device='cuda'):
    reference_report = Path(reference_report).resolve()
    reference = json.loads(reference_report.read_text())
    metadata = reference['metadata']
    checkpoint = Path(metadata['checkpoint'])
    if digest(checkpoint) != metadata['checkpoint_sha256']:
        raise ValueError('Reference checkpoint no longer matches the original slides')
    saved = read_checkpoint(checkpoint, 'yalamanchi_vit')
    cfg = copy.deepcopy(saved['config'])
    if cfg['training']['objective'] != 'rollout' or cfg['training']['horizon'] != 5:
        raise ValueError('Reference must be MSE-only five-step rollout training')
    if cfg['training']['max_train_pairs'] is not None:
        raise ValueError('Reference must use all selected training windows')
    index = CacheIndex(cfg); check_cache(saved, index)
    if metadata['cache_fingerprint'] != index.fingerprint:
        raise ValueError('Cache differs from the original slide evaluation')
    validation = CrossFileSequenceDataset(index, 'val', 10,
        dt_rtol=cfg['data']['dt_rtol'], dt_atol=cfg['data']['dt_atol'])
    if (metadata['split'] != 'val' or metadata['horizon'] != 10
            or metadata['start'] != validation.starts()[0]
            or metadata['validation_windows'] != len(validation)):
        raise ValueError('Reference slides must match the first ten-step validation window')
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    spec = dict(reference_checkpoint=str(checkpoint.resolve()), reference_sha256=metadata['checkpoint_sha256'],
        reference_report=str(reference_report), reference_report_sha256=digest(reference_report),
        cache_fingerprint=index.fingerprint, data=cfg['data'], seed=cfg['seed'],
        training_files=len(cfg['data']['train']),
        training_windows=len(SequenceDataset(index, 'train', 5)),
        training_validation_windows=len(SequenceDataset(index, 'val', 5)),
        evaluation_starts=validation.starts(), evaluation_split='val', evaluation_horizon=10,
        slice_axis=metadata['slice_axis'], slice_index=metadata['slice_index'],
        weights=WEIGHTS, primary_checkpoint='best.pt: lowest five-step validation normalized MSE',
        secondary_checkpoint='best-objective.pt: lowest five-step validation combined loss',
        loss='mean_lead(normalized MSE + lambda * mean_sample(((Kpred-Ktarget)/max(Ktarget, floor))^2))',
        energy='K=.5*mean_space(u²+v²+w²), restored physical velocity; mean flow included',
        floor='max(1e-8 * training mean kinetic energy, 1e-12)', reference_normalization=saved['normalization'])
    spec['arm_config_sha256'] = {}
    for arm, weight in WEIGHTS.items():
        arm_cfg = copy.deepcopy(cfg)
        arm_cfg['training'].update(output_dir=str(output / arm), device=device,
                                   objective='rollout_mse_kinetic_energy')
        write_json(output / f'{arm}.json', dict(config=arm_cfg, energy_weight=weight))
        spec['arm_config_sha256'][arm] = digest(output / f'{arm}.json')
    write_json(output / 'experiment.json', spec)
    print(json.dumps({k:v for k,v in spec.items() if k not in ('data','evaluation_starts','reference_normalization')},indent=2),flush=True)


@torch.no_grad()
def validate(model, loader, stats, device, weight, floor):
    model.eval(); mean, std = normalization_tensors(stats, device)
    sums = np.zeros(3, dtype=np.float64); count = 0
    for initial, targets, intervals in loader:
        initial, targets, intervals = initial.to(device), targets.to(device), intervals.to(device)
        loss, mse, energy = rollout_loss(model, (initial-mean)/std,
            (targets-mean[:,None])/std[:,None], intervals, mean, std, weight, floor)
        values = np.asarray([loss.item(), mse.item(), energy.item()])
        if not np.isfinite(values).all(): raise FloatingPointError('Nonfinite validation loss')
        sums += len(initial) * values; count += len(initial)
    if not count: raise ValueError('Empty validation split')
    return dict(zip(('combined_loss','normalized_mse','relative_energy_mse'),(sums/count).tolist()))


def train(cfg, weight, spec):
    opt=cfg['training']; device=select_device(opt['device'])
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    index=CacheIndex(cfg)
    if index.fingerprint != spec['cache_fingerprint'] or cfg['data'] != spec['data']:
        raise ValueError('Training data changed after preparation')
    stats=training_statistics(index)
    if stats != spec['reference_normalization']:
        raise ValueError('Training statistics differ from the MSE reference')
    floor=energy_floor(stats); mean,std=normalization_tensors(stats,device)
    training=SequenceDataset(index,'train',5); validation=SequenceDataset(index,'val',5)
    output=Path(opt['output_dir']); output.mkdir(parents=True,exist_ok=False)
    write_json(output/'config.json',cfg); write_json(output/'normalization.json',stats)
    loss_config=dict(kind='relative_squared_global_kinetic_energy',weight=weight,denominator_floor=floor,
                     includes_mean_flow=True, physical_velocity=True, horizon=5)
    write_json(output/'loss.json',loss_config)
    model=YalamanchiViT(**cfg['model']).to(device)
    environment=dict(architecture=ARCHITECTURE,torch=str(torch.__version__),numpy=np.__version__,
        device=str(device),parameters=sum(p.numel() for p in model.parameters()),
        cuda_device=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        training_windows=len(training),validation_windows=len(validation))
    write_json(output/'environment.json',environment)
    print(json.dumps({**environment,'energy_loss':loss_config}),flush=True)
    common=dict(batch_size=opt['batch_size'],num_workers=opt['num_workers'],pin_memory=device.type=='cuda')
    loader=DataLoader(training,shuffle=True,generator=torch.Generator().manual_seed(cfg['seed']),**common)
    val_loader=DataLoader(validation,shuffle=False,**common)
    optimizer=torch.optim.AdamW(model.parameters(),lr=opt['learning_rate'],weight_decay=opt['weight_decay'])
    best_mse,best_objective=float('inf'),float('inf')
    for epoch in range(1,opt['epochs']+1):
        if device.type=='cuda':torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
        started=time.perf_counter();model.train();totals=np.zeros(3,dtype=np.float64);count=0
        for initial,targets,intervals in loader:
            initial,targets,intervals=initial.to(device),targets.to(device),intervals.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss,mse,energy=rollout_loss(model,(initial-mean)/std,(targets-mean[:,None])/std[:,None],
                                        intervals,mean,std,weight,floor)
            if not torch.isfinite(torch.stack((loss,mse,energy))).all():
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),opt['grad_clip'],error_if_nonfinite=True)
            optimizer.step()
            totals+=len(initial)*np.asarray([loss.item(),mse.item(),energy.item()]);count+=len(initial)
        report=validate(model,val_loader,stats,device,weight,floor)
        if device.type=='cuda':torch.cuda.synchronize(device)
        record=dict(epoch=epoch,energy_weight=weight,objective=opt['objective'],horizon=5,
            training=dict(zip(('combined_loss','normalized_mse','relative_energy_mse'),(totals/count).tolist())),
            validation=report,optimizer_updates=epoch*len(loader),seconds=time.perf_counter()-started,
            peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None)
        with (output/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(record,allow_nan=False)+'\n')
        print(json.dumps(record,allow_nan=False),flush=True)
        improved_mse=report['normalized_mse']<best_mse
        improved_objective=report['combined_loss']<best_objective
        saved=dict(format_version=1,architecture=ARCHITECTURE,model=model.state_dict(),config=cfg,
            normalization=stats,epoch=epoch,validation=report,dt_reference=index.dt_reference,
            cache_fingerprint=index.fingerprint,environment=environment,energy_loss=loss_config)
        names=['last.pt']+(['best.pt'] if improved_mse else [])+(['best-objective.pt'] if improved_objective else [])
        for name in names:
            temporary=output/(name+'.tmp');torch.save(saved,temporary);temporary.replace(output/name)
        if improved_mse:best_mse=report['normalized_mse']
        if improved_objective:best_objective=report['combined_loss']


def plot_comparison(output, reports):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels={'reference':'MSE-only reference','energy-1':'MSE + KE, λ=1','energy-10':'MSE + KE, λ=10'}
    first=next(iter(reports.values()))
    for scope,title in [('full_validation','Mean over validation origins'),
                        ('selected_trajectory','Original slide trajectory')]:
        fig,axes=plt.subplots(1,3,figsize=(16,4.6))
        actual=first[scope]
        leads=[row['lead_steps'] for row in actual]
        axes[0].plot(leads,[row['target_kinetic_energy'] for row in actual],'ko-',label='Actual')
        for arm,report in reports.items():
            rows=report[scope]
            axes[0].plot(leads,[r['predicted_kinetic_energy'] for r in rows],'--',label=labels[arm])
            axes[1].plot(leads,[r['relative_energy_rmse'] for r in rows],label=labels[arm])
            axes[2].plot(leads,[r['physical_mse'] for r in rows],label=labels[arm])
        for ax,ylabel in zip(axes,('Kinetic energy','Relative energy RMSE','Velocity MSE')):
            ax.set(xlabel='Forecast lead (saved steps)',ylabel=ylabel);ax.grid(alpha=.25);ax.legend(fontsize=8)
        if scope=='full_validation':title=f'Mean over {first["metadata"]["validation_windows"]} validation origins'
        fig.suptitle(title+' · primary checkpoints selected by validation MSE')
        fig.tight_layout();fig.savefig(Path(output)/f'{scope}-comparison.png',dpi=180);plt.close(fig)


def evaluate(checkpoint, output, spec, device, label, slides=True):
    saved=read_checkpoint(checkpoint,'yalamanchi_vit');cfg=saved['config'];index=CacheIndex(cfg)
    check_cache(saved,index)
    if index.fingerprint!=spec['cache_fingerprint'] or cfg['data']!=spec['data']:
        raise ValueError('Evaluation data differ from the slide reference')
    if saved['normalization']!=spec['reference_normalization']:
        raise ValueError('Evaluation normalization differs from reference')
    dataset=CrossFileSequenceDataset(index,'val',10,dt_rtol=cfg['data']['dt_rtol'],dt_atol=cfg['data']['dt_atol'])
    if dataset.starts()!=spec['evaluation_starts']:raise ValueError('Validation origins changed')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    model=restore_model(saved,device,'yalamanchi_vit');model.eval()
    stats=saved['normalization'];floor=energy_floor(stats)
    totals=np.zeros((10,9),dtype=np.float64);selected=None;count=0;energies=[]
    for case,(initial,targets,intervals) in enumerate(dataset):
        prediction,metrics=forecast_window(model,initial,targets,intervals,stats,device)
        actual=targets.numpy().astype(np.float64);predicted=prediction.astype(np.float64)
        error=predicted-actual
        actual_k=.5*np.mean(np.sum(actual**2,axis=1),axis=(1,2,3))
        predicted_k=.5*np.mean(np.sum(predicted**2,axis=1),axis=(1,2,3))
        relative=(predicted_k-actual_k)/np.maximum(actual_k,floor)
        std=np.asarray(stats['std']).reshape(1,3,1,1,1)
        totals+=np.stack((np.mean(error**2,axis=(1,2,3,4)),np.mean((error/std)**2,axis=(1,2,3,4)),
            predicted_k,actual_k,relative**2,relative,np.abs(relative),
            np.mean((initial.numpy().astype(np.float64)[None]-actual)**2,axis=(1,2,3,4)),
            np.mean(actual**2,axis=(1,2,3,4))),axis=1)
        count+=1
        energies.append(np.stack((actual_k,predicted_k),axis=0))
        if case==0:
            for lead,row in enumerate(metrics):
                row.update(relative_energy_rmse=float(abs(relative[lead])),relative_energy_bias=float(relative[lead]))
            selected=metrics
            np.savez_compressed(output/'selected-forecast.npz',initial=initial.numpy(),actual=targets.numpy(),
                                predicted=prediction,times=dataset.times[:11],normalized_intervals=intervals.numpy())
        if count%50==0:print(f'{label}: evaluated {count}/{len(dataset)} validation origins',flush=True)
    rows=[]
    for lead,values in enumerate((totals/count).tolist(),1):
        mse,nmse,pred,actual,re2,bias,rae,persistence,target2=values
        rows.append(dict(lead_steps=lead,physical_mse=mse,normalized_mse=nmse,predicted_kinetic_energy=pred,
            target_kinetic_energy=actual,relative_energy_rmse=float(np.sqrt(re2)),relative_energy_bias=bias,
            relative_energy_mae=rae,persistence_mse=persistence,
            mse_skill_vs_persistence=1-mse/persistence if persistence else None,
            relative_l2=float(np.sqrt(mse/target2)) if target2 else None))
    weight=saved.get('energy_loss',{}).get('weight',0.)
    meta=dict(architecture='yalamanchi_vit',epoch=saved['epoch'],
        objective=f'five-step rollout, normalized MSE + {weight:g} × relative KE error²',training_horizon=5,
        checkpoint=str(Path(checkpoint).resolve()),checkpoint_sha256=digest(checkpoint),
        start=dataset.starts()[0],split='val',selection='first chronological window',
        validation_windows=len(dataset),horizon=10,cache_fingerprint=index.fingerprint,
        dt_reference=index.dt_reference,slice_axis=spec['slice_axis'],slice_index=spec['slice_index'],energy_weight=weight)
    if slides:
        with np.load(output/'selected-forecast.npz') as selected_arrays:
            render_deck(output,selected_arrays['initial'],selected_arrays['actual'],selected_arrays['predicted'],
                        selected_arrays['times'],meta,selected,spec['slice_axis'],spec['slice_index'])
    np.savez_compressed(output/'energy-trajectories.npz',energies=np.asarray(energies),
                       origin_times=np.asarray([s['time'] for s in dataset.starts()]),lead_steps=np.arange(1,11))
    report=dict(metadata=meta,selected_trajectory=selected,full_validation=rows,
                evaluation_starts=dataset.starts(),denominator_floor=floor)
    write_json(output/'report.json',report)
    return report


def worker(output,arm):
    output=Path(output).resolve();spec=json.loads((output/'experiment.json').read_text())
    if digest(output/f'{arm}.json')!=spec['arm_config_sha256'][arm]:
        raise ValueError('Worker configuration changed after preparation')
    settings=json.loads((output/f'{arm}.json').read_text());cfg=settings['config'];weight=settings['energy_weight']
    if weight!=spec['weights'][arm]:raise ValueError('Energy weight changed after preparation')
    try:
        train(cfg,weight,spec)
        device=select_device(cfg['training']['device'])
        evaluate(output/arm/'best.pt',output/arm/'validation-slides',spec,device,arm)
        # One worker reevaluates the unchanged MSE checkpoint to measure per-case
        # energy errors absent from the old aggregate report. No reference training.
        if arm=='energy-1':
            if digest(spec['reference_checkpoint'])!=spec['reference_sha256']:
                raise ValueError('Reference weights changed')
            evaluate(spec['reference_checkpoint'],output/'reference',spec,device,'reference',slides=False)
    except Exception as error:
        write_json(output/f'{arm}-failure.json',dict(type=type(error).__name__,message=str(error)))
        raise


def summarize(output):
    output=Path(output).resolve();reports={};missing=[];rows=[]
    for arm,path in [('reference',output/'reference/report.json')]+[(a,output/a/'validation-slides/report.json') for a in WEIGHTS]:
        if not path.is_file():missing.append(arm);continue
        report=json.loads(path.read_text());reports[arm]=report
        rows.extend(dict(arm=arm,checkpoint_epoch=report['metadata']['epoch'],**row) for row in report['full_validation'])
    if not missing:
        reference=reports['reference']
        for arm in WEIGHTS:
            if reports[arm]['evaluation_starts']!=reference['evaluation_starts']:
                raise ValueError('Cannot compare different validation origins')
            for scope in ('full_validation','selected_trajectory'):
                if [r['target_kinetic_energy'] for r in reports[arm][scope]]!=[r['target_kinetic_energy'] for r in reference[scope]]:
                    raise ValueError('Actual energy differs between runs')
        plot_comparison(output,reports)
    write_json(output/'comparison.json',dict(complete=not missing,missing=missing,results=reports))
    if rows:
        with (output/'comparison.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    if missing:raise RuntimeError('Incomplete experiment: '+', '.join(missing))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','worker','summarize','evaluate'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference-report',type=Path,default=REFERENCE)
    parser.add_argument('--arm',choices=list(WEIGHTS))
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--evaluation-output',type=Path)
    args=parser.parse_args(argv)
    if args.action=='prepare':prepare(args.output,args.reference_report,args.device)
    elif args.action=='summarize':summarize(args.output)
    elif args.action=='worker':
        if args.arm is None:parser.error('worker requires --arm')
        worker(args.output,args.arm)
    else:
        if args.checkpoint is None or args.evaluation_output is None:
            parser.error('evaluate requires --checkpoint and --evaluation-output (a new directory)')
        spec=json.loads((args.output/'experiment.json').read_text())
        evaluate(args.checkpoint,args.evaluation_output,spec,select_device(args.device),str(args.checkpoint))


if __name__=='__main__':main()
