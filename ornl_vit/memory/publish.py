"""Publish lightweight MATEY diagnostics in home runs; weights stay in scratch."""
import json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def publish(source,destination):
    source,destination=Path(source),Path(destination);destination.mkdir(parents=True,exist_ok=True)
    for p in source.iterdir():
        if p.is_file() and p.suffix in ('.json','.jsonl','.png','.npz','.csv'):
            shutil.copy2(p,destination/p.name)
    path=destination/'metrics.jsonl'
    if path.exists():
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        if rows:
            fig,ax=plt.subplots(figsize=(8,4));key=next(k for k in rows[0] if k.startswith('training_loss'))
            ax.plot([r['epoch'] for r in rows],[r[key] for r in rows],label='Training objective')
            ax.plot([r['epoch'] for r in rows],[r['validation']['mean_normalized_mse'] for r in rows],label='Validation five-lead mean')
            ax.set(xlabel='Epoch',ylabel='Normalized MSE');ax.legend();fig.tight_layout();fig.savefig(destination/'learning-curves.png',dpi=150);plt.close(fig)
    if (destination/'first-forecast.npz').exists():
        data=np.load(destination/'first-forecast.npz');a,p=data['actual'],data['predicted']
        a=np.linalg.norm(a.astype(float),axis=1)[:,:,:,a.shape[-1]//2]
        p=np.linalg.norm(p.astype(float),axis=1)[:,:,:,p.shape[-1]//2]
        limit=max(a.max(),p.max());err=p-a;emax=max(float(np.abs(err).max()),1e-12)
        fig,axes=plt.subplots(3,5,figsize=(15,9),constrained_layout=True)
        for row,values in enumerate([a,p,err]):
            for col in range(5):
                im=axes[row,col].imshow(values[col].T,origin='lower',vmin=-emax if row==2 else 0,vmax=emax if row==2 else limit,cmap='RdBu_r' if row==2 else 'viridis')
                axes[row,col].set_title(f'{["DNS","Prediction","Speed error"][row]}: lead {col+1}')
            fig.colorbar(im,ax=list(axes[row]),shrink=.7)
        fig.savefig(destination/'velocity-contours.png',dpi=150);plt.close(fig)
    (destination/'README.md').write_text('# MATEY experiment results\n\nScratch source: '+str(source)+'\n\nWeights remain in best.pt and last.pt under that path. acceptance.pt is disposable test weights.\n\nInspect acceptance.json for the three-update memory gate, failure-rank0.json for failures, and complete.json for actual epochs and time-budget stopping. Its absence does not mean the run completed. Metrics, energy plots, learning curves and first-case speed contours are saved here when available. Contours use one mid-z slice with shared limits; they are not population statistics.\n')
