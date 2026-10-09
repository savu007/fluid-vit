"""Physical units, per-sample energy penalties, rollout gradients and job integration."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from baseline_vit.dataset import CacheIndex
from forecasting.checkpoint import read_checkpoint
from forecasting.dataset import CrossFileSequenceDataset
from forecasting.rollout import forecast_loss
from forecasting.train import train as train_reference
from .config import load_config
from .energy_experiment import ROOT, digest, prepare, summarize, worker
from .energy_loss import energy_floor, kinetic_energy, rollout_loss
from .model import ARCHITECTURE, YalamanchiViT
from .test_baseline import TINY


class Scale(nn.Module):
    def __init__(self):super().__init__();self.scale=nn.Parameter(torch.tensor(2.));self.modes=[]
    def forward(self,x,dt):self.modes.append(torch.is_grad_enabled());return x*self.scale


class LossTests(unittest.TestCase):
    def test_energy_uses_all_components_and_physical_units(self):
        velocity=torch.tensor([1.,2.,3.]).reshape(1,3,1,1,1).expand(2,3,4,4,4)
        torch.testing.assert_close(kinetic_energy(velocity),torch.tensor([7.,7.]))
        mean=torch.tensor([3.,-1.,4.]).reshape(1,3,1,1,1)
        std=torch.tensor([2.,3.,5.]).reshape(1,3,1,1,1)
        normalized=(velocity-mean)/std
        torch.testing.assert_close(kinetic_energy(normalized*std+mean),torch.tensor([7.,7.]))
        self.assertAlmostEqual(energy_floor(dict(mean=[1.,2.,3.],std=[1.,1.,1.])),8.5e-8)

    def test_zero_weight_matches_existing_loss_and_gradient(self):
        torch.manual_seed(4)
        x=torch.randn(2,3,2,2,2);targets=torch.randn(2,5,3,2,2,2);times=torch.ones(2,5)
        a,b=Scale(),Scale()
        expected=forecast_loss(a,x,targets,times,'rollout')
        got,_,_=rollout_loss(b,x,targets,times,torch.zeros(1,3,1,1,1),torch.ones(1,3,1,1,1),0.,1e-8)
        torch.testing.assert_close(got,expected,rtol=0,atol=0)
        expected.backward();got.backward()
        torch.testing.assert_close(a.scale.grad,b.scale.grad)

    def test_opposite_energy_errors_do_not_cancel_between_samples(self):
        class Fixed(nn.Module):
            def forward(self,x,dt):return x
        x=torch.zeros(2,3,2,2,2);x[1]=(4/3)**.5
        targets=torch.full((2,1,3,2,2,2),(2/3)**.5)
        _,_,energy=rollout_loss(Fixed(),x,targets,torch.ones(2,1),torch.zeros(1,3,1,1,1),torch.ones(1,3,1,1,1),1.,1e-8)
        self.assertAlmostEqual(energy.item(),1.,places=6)

    def test_full_five_step_energy_gradient(self):
        model=Scale();x=torch.ones(1,3,2,2,2);targets=torch.ones(1,5,3,2,2,2)
        total,mse,energy=rollout_loss(model,x,targets,torch.ones(1,5),torch.zeros(1,3,1,1,1),torch.ones(1,3,1,1,1),1.,1e-8)
        total.backward()
        expected=sum(2*k*(2**k-1)*2**(k-1)+4*k*(2**(2*k)-1)*2**(2*k-1) for k in range(1,6))/5
        self.assertAlmostEqual(model.scale.grad.item()/expected,1.,places=6)
        self.assertEqual(model.modes,[True]*5)
        self.assertAlmostEqual(energy.item()/sum((2**(2*k)-1)**2 for k in range(1,6))*5,1.,places=6)
        self.assertGreater(mse.item(),0)

    def test_zero_target_finite_and_bad_weight_rejected(self):
        x=torch.zeros(1,3,2,2,2);targets=x[:,None];mean=torch.zeros(1,3,1,1,1);std=torch.ones_like(mean)
        total,_,energy=rollout_loss(Scale(),x,targets,torch.ones(1,1),mean,std,1.,1e-8)
        self.assertEqual(total.item(),0);self.assertEqual(energy.item(),0)
        for weight in (-1.,float('nan')):
            with self.assertRaises(ValueError):rollout_loss(Scale(),x,targets,torch.ones(1,1),mean,std,weight,1e-8)


class IntegrationTests(unittest.TestCase):
    def test_both_energy_weights_train_and_recreate_comparable_slides(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cache=root/'cache';cache.mkdir()
            cfg=load_config(ROOT/'yalamanchi_vit/config-runtime.yaml')
            cfg['model']=dict(TINY)
            cfg['data'].update(cache_dir=str(cache),train=[1],val=[3],test=[5],gaps=[])
            cfg['training'].update(output_dir=str(root/'reference-training'),device='cpu',epochs=1)
            spatial=np.random.default_rng(9).normal(size=(3,8,8,8)).astype(np.float32)
            times={}
            for fid,n in ((1,10),(3,11),(5,11)):
                times[str(fid)]=list(np.arange(n,dtype=float)+fid*20)
                np.save(cache/f'fluid_{fid}.npy',np.stack([spatial+.01*t for t in range(n)]))
            (cache/'manifest.json').write_text(json.dumps(dict(file_ids=[1,3,5],channels=['u','v','w'],
                target_grid=[8]*3,var_dims=['time','x','y','z'],time_coordinates=times)))
            with contextlib.redirect_stdout(io.StringIO()):train_reference(cfg,YalamanchiViT,ARCHITECTURE,save_last=True)
            checkpoint=root/'reference-training/best.pt';index=CacheIndex(cfg)
            validation=CrossFileSequenceDataset(index,'val',10)
            report=root/'reference-slides.json'
            report.write_text(json.dumps(dict(metadata=dict(checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),
                cache_fingerprint=index.fingerprint,split='val',horizon=10,start=validation.starts()[0],
                validation_windows=1,slice_axis='z',slice_index=4))))
            output=root/'energy'
            with contextlib.redirect_stdout(io.StringIO()):prepare(output,report,device='cpu')
            for arm,weight in (('energy-1',1.),('energy-10',10.)):
                with contextlib.redirect_stdout(io.StringIO()):worker(output,arm)
                for filename in ('best.pt','last.pt','best-objective.pt'):
                    saved=read_checkpoint(output/arm/filename,'yalamanchi_vit')
                    self.assertEqual(saved['energy_loss']['weight'],weight)
                    self.assertEqual(saved['config']['training']['objective'],'rollout_mse_kinetic_energy')
                    self.assertAlmostEqual(saved['validation']['combined_loss'],
                        saved['validation']['normalized_mse']+weight*saved['validation']['relative_energy_mse'],places=6)
                folder=output/arm/'validation-slides'
                self.assertTrue((folder/'forecast-slides.pdf').is_file())
                self.assertIn('relative KE error',json.loads((folder/'report.json').read_text())['metadata']['objective'])
                self.assertEqual((folder/'11-summary.png').read_bytes()[:8],b'\x89PNG\r\n\x1a\n')
            summarize(output)
            comparison=json.loads((output/'comparison.json').read_text())
            self.assertTrue(comparison['complete'])
            self.assertEqual(len((output/'comparison.csv').read_text().splitlines()),31)
            self.assertEqual(digest(checkpoint),json.loads(report.read_text())['metadata']['checkpoint_sha256'])
            for name in ('full_validation-comparison.png','selected_trajectory-comparison.png'):
                self.assertEqual((output/name).read_bytes()[:8],b'\x89PNG\r\n\x1a\n')
            with self.assertRaises(FileExistsError),contextlib.redirect_stdout(io.StringIO()):prepare(output,report,device='cpu')
            reference=json.loads(report.read_text());reference['metadata']['checkpoint_sha256']='wrong';report.write_text(json.dumps(reference))
            with self.assertRaisesRegex(ValueError,'Reference checkpoint'):prepare(root/'bad',report,device='cpu')

    def test_incomplete_summary(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(RuntimeError,'Incomplete'):summarize(root)
            self.assertFalse(json.loads((Path(root)/'comparison.json').read_text())['complete'])


if __name__=='__main__':unittest.main()
