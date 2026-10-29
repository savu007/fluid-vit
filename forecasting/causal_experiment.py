"""Segmented 200-epoch causal attention versus normalization-matched MATEY."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from baseline_vit.dataset import CacheIndex, training_statistics
from .causal_model import ARCHITECTURE, OrderedMateyForecast
from .looped import restore_weights
from .looped_experiment import (ROOT, config_for, seed_all, datasets, update, validate,
    common_evaluation, synchronize, plot_learning)
from .long_summary import atomic_text

CASES = ('causal', 'unmasked')


def write_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, allow_nan=False))


def config():
    cfg = config_for('matey')
    cfg['training']['epochs'] = 200
    return cfg


def fingerprint():
    paths = [ROOT/'forecasting'/name for name in ('causal.py','causal_model.py',
        'causal_experiment.py','looped.py','looped_experiment.py','energy.py','long_summary.py')]
    for folder in ('ornl_vit','baseline_vit'):
        paths += [p for p in (ROOT/folder).rglob('*.py') if not p.name.startswith('test_')]
    value=hashlib.sha256()
    for p in sorted(set(paths)):
        value.update(str(p.relative_to(ROOT)).encode());value.update(p.read_bytes())
    return value.hexdigest()


def make(case, cfg, dt, device, checkpointing=True):
    return OrderedMateyForecast(cfg, dt, causal=case=='causal', checkpointing=checkpointing).to(device)


def profile(case, cfg, root, device):
    report=root/case;report.mkdir(parents=True,exist_ok=True)
    if (report/'profile.json').exists():raise FileExistsError('Profile already exists')
    seed_all(cfg['seed']);index=CacheIndex(cfg);stats=training_statistics(index)
    training,validation=datasets(cfg,'matey',index)
    batch=next(iter(DataLoader(training,batch_size=cfg['training']['batch_size'])))
    model=make(case,cfg,index.dt_reference,device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['training']['learning_rate'],weight_decay=cfg['training']['weight_decay'])
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    timings=[]
    for _ in range(3):
        synchronize(device);began=time.monotonic();update(model,batch,optimizer,stats,cfg,device)
        synchronize(device);timings.append(time.monotonic()-began)
    subset=Subset(validation,range(min(cfg['training']['batch_size'],len(validation))));subset.horizon=5
    began=time.monotonic();validate(model,DataLoader(subset,batch_size=cfg['training']['batch_size']),stats,device)
    synchronize(device);val_seconds=time.monotonic()-began
    began=time.monotonic();common_evaluation(model,cfg,index,stats,device,limit=1,split='val')
    synchronize(device);test_seconds=time.monotonic()-began
    peak=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None
    total=torch.cuda.get_device_properties(device).total_memory if device.type=='cuda' else None
    epoch_seconds=math.ceil(len(training)/cfg['training']['batch_size'])*max(timings[1:])+math.ceil(len(validation)/cfg['training']['batch_size'])*val_seconds
    value=dict(case=case,config=cfg,source_fingerprint=fingerprint(),normalization=stats,
        cache_fingerprint=index.fingerprint,dt_reference=index.dt_reference,training_windows=len(training),
        validation_windows=len(validation),training_update_seconds=timings,epoch_seconds=epoch_seconds,
        test_case_seconds=test_seconds,peak_cuda_memory_bytes=peak,total_cuda_memory_bytes=total,
        parameters=sum(p.numel() for p in model.parameters()),accepted=total is None or peak<.9*total)
    if 1.35*epoch_seconds+1800>11*3600:value['accepted']=False
    write_json(report/'profile.json',value)
    if not value['accepted']:raise RuntimeError('Memory/runtime gate failed; no training submitted')


def summarize(root):
    if not all((root/case/'complete.json').exists() for case in CASES):return False
    reports={case:json.loads((root/case/'complete.json').read_text()) for case in CASES}
    if reports['causal']['test']['test_starts']!=reports['unmasked']['test']['test_starts']:
        raise ValueError('Different paired test origins')
    stream=io.StringIO();writer=csv.writer(stream)
    writer.writerow(['case','best_epoch','lead','normalized_mse','relative_l2','predicted_K','actual_K'])
    for case,r in reports.items():
        for h in r['test']['horizons']:
            writer.writerow([case,r['best_epoch'],h['lead_steps'],h['normalized_mse'],h['relative_l2'],
                             h['predicted_kinetic_energy'],h['target_kinetic_energy']])
    atomic_text(root/'comparison.csv',stream.getvalue());write_json(root/'comparison.json',reports)
    return True


def segment(case,cfg,root,scratch,device,*,hours=11.,end_epoch=None):
    report=root/case;weights=scratch/case;weights.mkdir(parents=True,exist_ok=True)
    gate=json.loads((report/'profile.json').read_text())
    if not gate['accepted'] or gate['config']!=cfg or gate['source_fingerprint']!=fingerprint():
        raise ValueError('Profile/config/source mismatch')
    index=CacheIndex(cfg)
    if index.fingerprint!=gate['cache_fingerprint']:raise ValueError('Cache changed')
    if device.type=='cuda' and gate['peak_cuda_memory_bytes']>=.9*torch.cuda.get_device_properties(device).total_memory:
        raise ValueError('Insufficient GPU memory headroom')
    seed_all(cfg['seed']);model=make(case,cfg,index.dt_reference,device)
    options=cfg['training'];stats=gate['normalization']
    optimizer=torch.optim.AdamW(model.parameters(),lr=options['learning_rate'],weight_decay=options['weight_decay'])
    start=1;best=float('inf');updates=0
    if (weights/'last.pt').exists():
        saved=torch.load(weights/'last.pt',map_location='cpu',weights_only=False)
        if saved['architecture']!=ARCHITECTURE or saved['case']!=case or saved['config']!=cfg or saved['source_fingerprint']!=fingerprint() or saved['cache_fingerprint']!=index.fingerprint:
            raise ValueError('Resume checkpoint mismatch')
        restore_weights(model,saved['model']);optimizer.load_state_dict(saved['optimizer'])
        start=saved['epoch']+1;best=saved['best_validation'];updates=saved['optimizer_updates']
        torch.set_rng_state(saved['torch_rng']);random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])
        if device.type=='cuda':torch.cuda.set_rng_state(saved['cuda_rng'],device)
    training,validation=datasets(cfg,'matey',index);val_loader=DataLoader(validation,batch_size=options['batch_size'])
    stop=min(options['epochs'],start+49)
    if end_epoch is not None:stop=min(stop,end_epoch)
    deadline=time.monotonic()+hours*3600;completed=start-1
    write_json(report/'experiment.json',dict(case=case,config=cfg,checkpoint_directory=str(weights),
        architecture=ARCHITECTURE,source_fingerprint=fingerprint(),
        objective='five-step full rollout normalized MSE; prefix input statistics; frame-local block norms',
        temporal_attention='causal' if case=='causal' else 'unmasked'))
    for epoch in range(start,stop+1):
        # Leave time to finish an epoch, its checkpoint and learning curves.
        if completed>=start and time.monotonic()+1.35*gate['epoch_seconds']+300>=deadline:break
        began=time.monotonic();count=0;totals=dict(training_loss=0.,auxiliary_mse=0.,final_mse=0.)
        loader=DataLoader(training,batch_size=options['batch_size'],shuffle=True,
            generator=torch.Generator().manual_seed(cfg['seed']+epoch))
        for batch in loader:
            values=update(model,batch,optimizer,stats,cfg,device);n=batch[0].shape[0];count+=n;updates+=1
            for key in totals:totals[key]+=values[key]*n
        val=validate(model,val_loader,stats,device);improved=val['mean_normalized_mse']<best
        if improved:best=val['mean_normalized_mse']
        row=dict(epoch=epoch,optimizer_updates=updates,seconds=time.monotonic()-began,validation=val,
            **{key:value/count for key,value in totals.items()})
        path=report/'metrics.jsonl';rows=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        atomic_text(path,''.join(json.dumps(r)+'\n' for r in rows if r['epoch']<epoch)+json.dumps(row)+'\n')
        saved=dict(architecture=ARCHITECTURE,case=case,config=cfg,source_fingerprint=fingerprint(),
            epoch=epoch,model=model.state_dict(),optimizer=optimizer.state_dict(),normalization=stats,
            cache_fingerprint=index.fingerprint,dt_reference=index.dt_reference,best_validation=best,
            validation=val,optimizer_updates=updates,torch_rng=torch.get_rng_state(),python_rng=random.getstate(),
            numpy_rng=np.random.get_state(),cuda_rng=torch.cuda.get_rng_state(device) if device.type=='cuda' else None)
        for name in (['best.pt'] if improved else [])+['last.pt']:
            temporary=weights/(name+'.tmp');torch.save(saved,temporary);temporary.replace(weights/name)
        completed=epoch;print(json.dumps(row),flush=True)
    plot_learning(report)
    if completed<options['epochs']:
        write_json(report/'segment.json',dict(completed_epoch=completed,next_epoch=completed+1));return
    # If training used the segment budget, evaluate in a fresh allocation.
    if time.monotonic()+1.35*gate['test_case_seconds']*353+300>=deadline:
        write_json(report/'segment.json',dict(completed_epoch=completed,next_stage='evaluation'));return
    saved=torch.load(weights/'best.pt',map_location='cpu',weights_only=False);restore_weights(model,saved['model'])
    test=common_evaluation(model,cfg,index,stats,device,report)
    test['model']='matey '+case+' temporal attention';write_json(report/'test.json',test)
    write_json(report/'complete.json',dict(epochs=options['epochs'],best_epoch=saved['epoch'],test=test,
        best_checkpoint=str(weights/'best.pt'),last_checkpoint=str(weights/'last.pt'),case=case))
    summarize(root)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['profile','train'])
    p.add_argument('--case',choices=CASES,required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--scratch',type=Path,required=True);args=p.parse_args();device=torch.device('cuda');cfg=config()
    try:
        if args.action=='profile':profile(args.case,cfg,args.root,device)
        else:segment(args.case,cfg,args.root,args.scratch,device)
    except Exception as error:
        (args.root/args.case).mkdir(parents=True,exist_ok=True)
        write_json(args.root/args.case/'failure.json',dict(type=type(error).__name__,message=str(error)))
        raise

if __name__=='__main__':main()
