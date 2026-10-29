"""Resumable 200-epoch ordinary-model runs, using existing rollout/evaluation code."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from baseline_vit.dataset import CacheIndex
from .long_summary import summarize
from .looped import FinestStageForecast, restore_weights
from .looped_experiment import (config_for, source_fingerprint, seed_all, datasets,
    profile, update, validate, write_json, plot_learning, common_evaluation)


# Exact pre-fix source: training is unchanged; only summary finalization differs.
LEGACY_FINALIZATION_SOURCE = '6628c2ed47d417d9d1a9dc51c60de72e814457d3585c1ff948dc27653a1d75b8'


def compatible_source(source):
    return (source == Path(__file__).read_text() or
            hashlib.sha256(source.encode()).hexdigest() == LEGACY_FINALIZATION_SOURCE)


def config(name):
    cfg=config_for(name)
    cfg['training']['epochs']=200
    return cfg


def segment(name, cfg, root, scratch, device, end_epoch=None):
    report=root/name
    report.mkdir(parents=True,exist_ok=True)
    weights=scratch/name
    weights.mkdir(parents=True,exist_ok=True)
    gate=json.loads((root/'profiles'/name/'profile.json').read_text())
    if not gate['accepted'] or gate['config']!=cfg or gate['source_fingerprint']!=source_fingerprint():
        raise ValueError('Profile/config/source mismatch')
    if not compatible_source(gate['long_source']):
        raise ValueError('Long-training source changed since profiling')
    index=CacheIndex(cfg)
    if index.fingerprint!=gate['cache_fingerprint']: raise ValueError('Cache changed')
    if device.type=='cuda' and gate['peak_cuda_memory_bytes']>=.9*torch.cuda.get_device_properties(device).total_memory:
        raise ValueError('Insufficient GPU memory headroom')
    seed_all(cfg['seed'])
    model=FinestStageForecast(name,cfg,index.dt_reference,looped=False).to(device)
    options=cfg['training'];stats=gate['normalization']
    optimizer=torch.optim.AdamW(model.parameters(),lr=options['learning_rate'],weight_decay=options['weight_decay'])
    start_epoch=1;best=float('inf');updates=0
    if (weights/'last.pt').exists():
        saved=torch.load(weights/'last.pt',map_location='cpu',weights_only=False)
        if saved['config']!=cfg or saved['cache_fingerprint']!=index.fingerprint:
            raise ValueError('Resume checkpoint mismatch')
        restore_weights(model,saved['model']);optimizer.load_state_dict(saved['optimizer'])
        start_epoch=saved['epoch']+1;best=saved['best_validation'];updates=saved['optimizer_updates']
        torch.set_rng_state(saved['torch_rng']);random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])
        if device.type=='cuda':torch.cuda.set_rng_state(saved['cuda_rng'],device)
    training,validation=datasets(cfg,name,index)
    val_loader=DataLoader(validation,batch_size=options['batch_size'])
    epoch_seconds=gate['estimated_training_seconds']/options['epochs']
    reserve=max(1800.,1.35*gate['test_case_seconds']*gate['test_windows']+300)
    chunk=max(1,min(50,math.floor((12*3600-reserve)/(1.35*epoch_seconds))))
    stop=min(options['epochs'],start_epoch+chunk-1)
    if end_epoch is not None:stop=min(stop,end_epoch)
    write_json(report/'experiment.json',dict(config=cfg,profile=gate,checkpoint_directory=str(weights),
        objective='ordinary model, five-step rollout normalized MSE; full gradients',
        source_fingerprint=source_fingerprint(),epochs_per_segment=chunk))
    for epoch in range(start_epoch,stop+1):
        started=time.perf_counter();count=0;totals=dict(training_loss=0.,auxiliary_mse=0.,final_mse=0.)
        # Epoch-specific shuffling makes segmentation independent of loader lifetime.
        loader=DataLoader(training,batch_size=options['batch_size'],shuffle=True,
            generator=torch.Generator().manual_seed(cfg['seed']+epoch))
        for batch in loader:
            values=update(model,batch,optimizer,stats,cfg,device);n=batch[0].shape[0];count+=n;updates+=1
            for key in totals:totals[key]+=values[key]*n
        validation_report=validate(model,val_loader,stats,device)
        improved=validation_report['mean_normalized_mse']<best
        if improved:best=validation_report['mean_normalized_mse']
        row=dict(epoch=epoch,optimizer_updates=updates,seconds=time.perf_counter()-started,
            validation=validation_report,**{key:value/count for key,value in totals.items()})
        # last.pt is authoritative: discard a journal tail left before checkpoint commit.
        path=report/'metrics.jsonl'
        rows=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        rows=[r for r in rows if r['epoch']<epoch]+[row]
        temporary=report/'metrics.jsonl.tmp';temporary.write_text(''.join(json.dumps(r)+'\n' for r in rows));temporary.replace(path)
        saved=dict(architecture='ordinary_forecast_long_v1',config=cfg,epoch=epoch,
            model=model.state_dict(),optimizer=optimizer.state_dict(),normalization=stats,
            cache_fingerprint=index.fingerprint,dt_reference=index.dt_reference,
            best_validation=best,validation=validation_report,optimizer_updates=updates,
            torch_rng=torch.get_rng_state(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),
            cuda_rng=torch.cuda.get_rng_state(device) if device.type=='cuda' else None)
        for filename in (['best.pt'] if improved else [])+['last.pt']:
            temporary=weights/(filename+'.tmp');torch.save(saved,temporary);temporary.replace(weights/filename)
        print(json.dumps(row),flush=True)
    plot_learning(report)
    if stop<options['epochs']:
        write_json(report/'segment.json',dict(completed_epoch=stop,next_epoch=stop+1))
        return
    saved=torch.load(weights/'best.pt',map_location='cpu',weights_only=False)
    restore_weights(model,saved['model'])
    test=common_evaluation(model,cfg,index,stats,device,report)
    write_json(report/'complete.json',dict(epochs=options['epochs'],best_epoch=saved['epoch'],test=test,
        best_checkpoint=str(weights/'best.pt'),last_checkpoint=str(weights/'last.pt')))
    summarize(root)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('profile','train'))
    parser.add_argument('--model',choices=('matey','yalamanchi'),required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--scratch',type=Path,required=True)
    args=parser.parse_args();cfg=config(args.model);device=torch.device('cuda')
    try:
        if args.action=='profile':
            dest=args.root/'profiles'/args.model
            gate=profile(args.model+'-control',cfg,device,dest)
            gate['long_source']=Path(__file__).read_text()
            write_json(dest/'profile.json',gate)
            epoch_seconds=gate['estimated_training_seconds']/cfg['training']['epochs']
            reserve=max(1800.,1.35*gate['test_case_seconds']*gate['test_windows']+300)
            if 1.35*epoch_seconds>12*3600-reserve:raise RuntimeError('One epoch exceeds segment budget')
        else:segment(args.model,cfg,args.root,args.scratch,device)
    except Exception as error:
        args.root.mkdir(parents=True,exist_ok=True)
        write_json(args.root/(args.model+'-failure.json'),dict(type=type(error).__name__,message=str(error)))
        raise


if __name__=='__main__':main()
