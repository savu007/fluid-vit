"""Reproducible CPU audit; no production weights or training settings are modified."""
import argparse
import copy
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from baseline_vit.dataset import CacheIndex
from ..dataset import HistoryDataset
from ..engine import evaluate, normalized_mse
from ..model import MateyForecast, UPSTREAM_COMMIT
from ..matey.models.turbt import TurbT
from ..matey.compat import attention_backend
from ..matey.utils.forward_options import ForwardOptionsBase
from ..matey.data_utils.shared_utils import normalize_spatiotemporal_persample
from .model import CheckpointedMatey, pushforward_five


def compare(a, b):
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    return float((a-b).detach().abs().max())


def gradients(a, b):
    pa, pb = dict(a.named_parameters()), dict(b.named_parameters())
    assert pa.keys() == pb.keys()
    active, absent, maximum = [], [], 0.
    for name, p in pa.items():
        q = pb[name]
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is None:
            absent.append(name)
        else:
            assert torch.isfinite(p.grad).all(), name
            maximum = max(maximum, compare(p.grad, q.grad))
            active.append(name)
    return dict(max_absolute_difference=maximum, active_parameters=active, no_gradient_parameters=absent)


def model_checks():
    torch.manual_seed(42)
    options = dict(grid_size=16, in_channels=3, history=4, patch_size=2,
                   embed_dim=24, num_heads=3, processor_blocks=12, nlevels=3,
                   filtersize=2, upsampling='learned', drop_path=.2,
                   attention_backend='sdpa', dt_reference=.03, dt_rtol=.003, dt_atol=1e-10)
    wrapper = MateyForecast(**options).train()
    # Construct the vendored core independently, rather than call wrapper.core twice.
    core = TurbT(tokenizer_heads=[dict(head_name='tk-3D', patch_size=[[2,2,2]])],
                 n_states=3, n_steps=4, embed_dim=24, num_heads=3, processor_blocks=12,
                 drop_path=.2, leadtime=True, bias_type='PositionAreaBias',
                 hierarchical=dict(nlevels=3, filtersize=2, fixedupsample=False, linearupsample=False)).train()
    core.load_state_dict(wrapper.core.state_dict())
    x = torch.randn(1,4,3,16,16,16)
    times = torch.arange(9, dtype=torch.float64)[None]*.03
    target = torch.randn(1,3,16,16,16)
    stats = dict(std=[1.3,1.4,1.2])
    torch.manual_seed(17)
    a = wrapper(x, times[:,:4], times[:,4])
    loss_a = normalized_mse(a[:,None], target[:,None], stats); loss_a.backward()
    torch.manual_seed(17)
    opts = ForwardOptionsBase(imod=2, imod_bottom=0, tkhead_name='tk-3D',
                              leadtime=((times[:,4:5]-times[:,3:4])/.03).float())
    with attention_backend('sdpa'):
        b = core(x.permute(1,0,2,3,4,5), torch.tensor([[0,1,2]]), None, opts)
    # Independent channelwise loss expression.
    loss_b = sum(((b[:,c]-target[:,c])/stats['std'][c]).square().mean() for c in range(3))/3
    torch.testing.assert_close(loss_a, loss_b)
    # Use identical reduction for strict gradient parity; independent expression checked above.
    normalized_mse(b[:,None], target[:,None], stats).backward()
    report = dict(scope=options, upstream_commit=UPSTREAM_COMMIT,
                  adapter_output_max_difference=compare(a,b), adapter_gradients=gradients(wrapper.core, core))
    plain = MateyForecast(**options).train(); plain.load_state_dict(wrapper.state_dict())
    checked = CheckpointedMatey(**options).train(); checked.load_state_dict(plain.state_dict())
    outputs=[]
    for model in (plain,checked):
        torch.manual_seed(18)
        pred=pushforward_five(model,x,times)
        normalized_mse(pred[:,None],target[:,None],stats).backward(); outputs.append(pred)
    report['checkpoint_output_max_difference']=compare(*outputs)
    report['checkpoint_gradients']=gradients(plain,checked)
    opts_a=torch.optim.AdamW(plain.parameters(),lr=1e-4)
    opts_b=torch.optim.AdamW(checked.parameters(),lr=1e-4)
    opts_a.step();opts_b.step()
    report['checkpoint_optimizer_max_difference']=max(compare(p,q) for p,q in zip(plain.parameters(),checked.parameters()))
    reference=copy.deepcopy(plain).train(); plain.zero_grad(set_to_none=True)
    reference.zero_grad(set_to_none=True)
    torch.manual_seed(19)
    a=pushforward_five(plain,x,times); a.square().mean().backward()
    torch.manual_seed(19)
    frames=list(x.unbind(dim=1))
    for k in range(5):
        with torch.set_grad_enabled(k==4):
            b=reference(torch.stack(frames[-4:],dim=1),times[:,k:k+4],times[:,k+4])
        frames.append(b)
    b.square().mean().backward()
    report['pushforward_output_max_difference']=compare(a,b)
    report['pushforward_gradients']=gradients(plain,reference)
    z,mean,std=normalize_spatiotemporal_persample(x.transpose(0,1))
    manual_std,manual_mean=torch.std_mean(x.transpose(0,1),dim=(0,3,4,5),keepdim=True)
    compare(mean,manual_mean);compare(std,manual_std.clamp_min(1e-4))
    torch.testing.assert_close(z*std+mean,x.transpose(0,1))
    report['normalization_roundtrip_max_difference']=float((z*std+mean-x.transpose(0,1)).abs().max())
    report['actual_drop_path_by_level']={level:[block.spatialtemporal.drop_path.drop_prob if hasattr(block.spatialtemporal.drop_path,'drop_prob') else 0. for block in blocks] for level,blocks in plain.core.module_blocks.items()}
    # Real-model evaluation compared to independently accumulated recursive predictions.
    plain.eval()
    targets=torch.randn(1,5,3,16,16,16)
    evaluation=evaluate(plain,DataLoader(TensorDataset(x,targets,times)),stats,torch.device('cpu'))
    frames=list(x.unbind(1)); independent=[]
    with torch.no_grad():
        for k in range(5):
            pred=plain(torch.stack(frames[-4:],1),times[:,k:k+4],times[:,k+4]);frames.append(pred)
            e=(pred.double()-targets[:,k].double()).numpy()
            independent.append(float(np.mean((e/np.array(stats['std'],dtype=np.float32).astype(float)[None,:,None,None,None])**2)))
    np.testing.assert_allclose(independent,[h['normalized_mse'] for h in evaluation['horizons']],rtol=1e-12)
    report['independent_evaluation_nmse']=independent
    return report


def data_checks(source):
    cfg=json.loads((source/'config.json').read_text()); index=CacheIndex(cfg)
    traces=[]; counts={}
    for split in ('train','val','test'):
        ds=HistoryDataset(index,split,cross_file=split=='test');counts[split]=len(ds)
        cases={0,len(ds)//2,len(ds)-1}
        if ds.continuous is not None:
            cases.add(next(i for i,s in enumerate(ds.pairs) if len({f for f,_ in ds.continuous.frames[s:s+9]})>1))
        for i in sorted(cases):
            h,y,t=ds[i]
            if ds.continuous is None:
                fid,start=ds.pairs[i]; refs=[(fid,j) for j in range(start,start+9)]
            else:
                start=ds.pairs[i];refs=ds.continuous.frames[start:start+9]
            fields=torch.cat((h,y)).numpy()
            for n,(fid,frame) in enumerate(refs):
                np.testing.assert_array_equal(fields[n],np.load(index.path(fid),mmap_mode='r')[frame])
                assert t[n].item()==index.times[fid][frame]
            traces.append(dict(split=split,case=i,history=refs[:4],targets=refs[4:],times=t.tolist()))
    return dict(cache_fingerprint=index.fingerprint,channels=index.manifest['channels'],
                spatial_dimensions=index.manifest['var_dims'],window_counts=counts,traces=traces,
                limitation='Exact selected-cache-array equality; original NetCDF-to-cache coarsening is not independently revalidated.')


def artifacts(source, output):
    for name in ('config.json','normalization.json','data.json','complete.json','test.json','metrics.jsonl',
                 'energy-evolution.png','energy-example.png','energy-spectra.png','energy-data.npz','first-forecast.npz'):
        shutil.copy2(source/name,output/name)
    arr=np.load(source/'first-forecast.npz');actual=arr['actual'];pred=arr['predicted']
    # Mid-z slice of velocity magnitude, common color scale across all frames.
    actual=np.sqrt((actual.astype(float)**2).sum(1))[:,:,:,actual.shape[-1]//2]
    pred=np.sqrt((pred.astype(float)**2).sum(1))[:,:,:,pred.shape[-1]//2]
    hi=max(actual.max(),pred.max()); err=pred-actual;emax=np.abs(err).max()
    fig,axes=plt.subplots(3,5,figsize=(15,9),constrained_layout=True)
    for k in range(5):
        for r,values in enumerate((actual,pred,err)):
            levels=np.linspace(-emax,emax,31) if r==2 else np.linspace(0,hi,31)
            im=axes[r,k].contourf(values[k].T,levels=levels,cmap='RdBu_r' if r==2 else 'viridis')
            axes[r,k].set_aspect('equal');axes[r,k].set_title(f'{["DNS","MATEY","Prediction − DNS"][r]}, lead {k+1}')
            axes[r,k].set_xlabel('x grid index');axes[r,k].set_ylabel('y grid index')
    fig.colorbar(im,ax=axes[2,:].ravel().tolist(),label="Speed error")
    fig.colorbar(axes[0,0].collections[0],ax=axes[:2,:].ravel().tolist(),label='Velocity magnitude')
    fig.savefig(output/'velocity-contours.png',dpi=150);plt.close(fig)
    rows=[json.loads(s) for s in (source/'metrics.jsonl').read_text().splitlines()]
    fig,ax=plt.subplots()
    ax.plot([r['epoch'] for r in rows],[r['training_loss_one_step'] for r in rows],label='Training (stochastic depth active)')
    for lead in (0,4):ax.plot([r['epoch'] for r in rows],[r['validation']['horizons'][lead]['normalized_mse'] for r in rows],label=f'Validation lead {lead+1}')
    ax.set(xlabel='Epoch',ylabel='Normalized MSE',yscale='log');ax.legend();fig.tight_layout();fig.savefig(output/'learning-curves.png',dpi=150);plt.close(fig)
    energy=np.load(source/'energy-data.npz');test=json.loads((source/'test.json').read_text())
    for key,col in [('actual_energy','target_kinetic_energy'),('predicted_energy','predicted_kinetic_energy')]:
        np.testing.assert_allclose(energy[key][:,1:].mean(0),[h[col] for h in test['rollout']['horizons']],rtol=1e-6)
    return dict(energy_arrays_agree_with_test=True,weights=str(source/'best.pt'),
                contour='First test case only, mid-z slice of speed; does not establish population-wide structure preservation.')


def fixed_sequence_check(source, output, updates=50):
    """Small CPU optimization diagnostic, not a production-resolution experiment."""
    cfg=json.loads((source/'config.json').read_text())
    fid=cfg['data']['train'][0]
    # Stride sampling is deliberately only a cheap debugging input, not a new cache.
    data=np.load(Path(cfg['data']['cache_dir'])/f'fluid_{fid}.npy',mmap_mode='r')
    fields=torch.from_numpy(np.array(data[:9,:,::4,::4,::4],copy=True))
    x=fields[:4][None];targets=fields[4:][None]
    manifest=json.loads((Path(cfg['data']['cache_dir'])/'manifest.json').read_text())
    times=torch.tensor(manifest['time_coordinates'][str(fid)][:9],dtype=torch.float64)[None]
    stats=json.loads((source/'normalization.json').read_text())
    options=dict(cfg['model']);options.update(grid_size=16,embed_dim=24)
    options.update(dt_reference=float(np.median(np.diff(times.numpy()))),dt_rtol=.003,dt_atol=1e-10)
    records={}
    for objective in ('one_step','pushforward'):
        torch.manual_seed(42);model=CheckpointedMatey(**options)
        opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
        rows=[]
        for step in range(updates+1):
            if step%5==0:
                model.eval()
                with torch.no_grad():
                    first=model(x,times[:,:4],times[:,4])
                    final=pushforward_five(model,x,times)
                    rows.append(dict(update=step,one_step_nmse=normalized_mse(first[:,None],targets[:,:1],stats).item(),
                                     fifth_step_nmse=normalized_mse(final[:,None],targets[:,-1:],stats).item()))
            if step==updates:break
            model.train();opt.zero_grad(set_to_none=True)
            pred=model(x,times[:,:4],times[:,4]) if objective=='one_step' else pushforward_five(model,x,times)
            target=targets[:,:1] if objective=='one_step' else targets[:,-1:]
            loss=normalized_mse(pred[:,None],target,stats);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        records[objective]=rows
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,key in zip(axes,('one_step_nmse','fifth_step_nmse')):
        for name,rows in records.items():ax.plot([r['update'] for r in rows],[r[key] for r in rows],label=name)
        ax.set(xlabel='Optimizer update',ylabel=key,yscale='log');ax.legend()
    fig.tight_layout();fig.savefig(output/'fixed-sequence-fitting.png',dpi=150);plt.close(fig)
    result=dict(scope='One fixed real training sequence, file 131 frames 0–8, stride-4 spatial sampling to 16³, width 24; 12 blocks retained. 50 updates. Evaluation disables stochastic depth. Not evidence of full-size convergence.',curves=records)
    (output/'fixed-sequence-fitting.json').write_text(json.dumps(result,indent=2))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('/pscratch/sd/s/savu/fluid-vit/matey-memory/one-step-59880460'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    report=dict(status='running',source=str(args.source),torch=str(torch.__version__),device='cpu')
    try:
        report['models']=model_checks();print('Model equivalence checks passed',flush=True)
        report['data']=data_checks(args.source);print('Data traces passed',flush=True)
        report['artifacts']=artifacts(args.source,args.output)
        report['fixed_sequence']=fixed_sequence_check(args.source,args.output)
        report['status']='passed'
    except Exception as exc:
        report['status']='failed';report['error']=repr(exc);raise
    finally:
        (args.output/'audit.json').write_text(json.dumps(report,indent=2))
    print(args.output,flush=True)

if __name__=='__main__': main()
