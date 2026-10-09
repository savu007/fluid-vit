"""Parseval, analytic spectra, shared forecast origins and two-model integration."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn
import yaml

from ornl_vit.config import load_config as load_matey
from ornl_vit.test_baseline import TINY as MATEY_TINY
from yalamanchi_vit.config import load_config as load_yala
from yalamanchi_vit.test_baseline import TINY as YALA_TINY
from .compare_matey import ROOT, prepare, predict, summarize, worker
from .energy import energy_spectrum


class SpectrumTests(unittest.TestCase):
    def test_parseval_on_even_and_odd_grids(self):
        for side in (5, 8, 16):
            with self.subTest(side=side):
                x = np.random.default_rng(4).normal(size=(3,side,side,side))
                energy, spectrum = energy_spectrum(x)
                self.assertAlmostEqual(energy, .5 * np.mean(np.sum(x**2,axis=0)), places=12)
                self.assertAlmostEqual(spectrum.sum(), energy, places=12)

    def test_constant_mean_flow_retained(self):
        energy, spectrum = energy_spectrum(np.ones((3,8,8,8)))
        self.assertAlmostEqual(energy, 1.5)
        self.assertAlmostEqual(spectrum[0], 1.5)
        self.assertAlmostEqual(spectrum[1:].sum(), 0)

    def test_single_fourier_mode_and_nyquist_weights(self):
        side=8
        x=np.zeros((3,side,side,side))
        x[0]=np.sin(2*np.pi*2*np.arange(side)/side)[:,None,None]
        energy,spectrum=energy_spectrum(x)
        self.assertAlmostEqual(energy,.25)
        self.assertAlmostEqual(spectrum[2],.25)
        self.assertLess(np.delete(spectrum,2).sum(),1e-25)
        # RFFT Nyquist endpoint must not be doubled.
        x.fill(0);x[0]=(-1.)**np.arange(side)[None,None,:]
        energy,spectrum=energy_spectrum(x)
        self.assertAlmostEqual(energy,.5)
        self.assertAlmostEqual(spectrum[4],.5)

    def test_bad_fields_rejected(self):
        for x in (np.zeros((3,4,4,2)),np.zeros((2,4,4,4)),np.full((3,4,4,4),np.nan)):
            with self.assertRaises(ValueError):energy_spectrum(x)


class PredictionTests(unittest.TestCase):
    def test_yalamanchi_uses_last_history_and_recorded_intervals(self):
        class Increment(nn.Module):
            def __init__(self):super().__init__();self.inputs=[]
            def forward(self,x,dt):
                self.inputs.append((x.clone(),dt.clone()))
                return x+dt[:,None,None,None,None]
        history=torch.arange(4.).reshape(1,4,1,1,1,1).expand(1,4,3,2,2,2)
        intervals=torch.tensor([1.,1.,1.,1.002,1.,1.,1.,1.],dtype=torch.float64)
        times=torch.cat((torch.zeros(1,dtype=torch.float64),intervals.cumsum(0)))[None]
        stats=dict(mean=[1.,1.,1.],std=[2.,2.,2.])
        model=Increment()
        result=predict(model,'yalamanchi',history,times,stats,1.)
        torch.testing.assert_close(model.inputs[0][0],torch.ones_like(history[:,-1]))
        torch.testing.assert_close(result[0,:,0,0,0,0],torch.tensor([5.004,7.004,9.004,11.004,13.004]))
        self.assertAlmostEqual(model.inputs[0][1].item(),1.002,places=6)


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);cache=self.root/'cache';cache.mkdir()
        spatial=np.random.default_rng(7).normal(size=(3,16,16,16)).astype(np.float32)
        times={}
        for fid,start in ((1,10),(3,30),(5,50),(6,59)):
            steps=np.arange(start,start+10,dtype=float)
            times[str(fid)]=(steps*.03).tolist()
            np.save(cache/f'fluid_{fid}.npy',np.stack([spatial+.01*step for step in steps]).astype(np.float32))
        (cache/'manifest.json').write_text(json.dumps(dict(file_ids=[1,3,5,6],channels=['u','v','w'],
            target_grid=[16]*3,var_dims=['time','x','y','z'],time_coordinates=times)))
        matey=load_matey(ROOT/'ornl_vit/config.yaml')
        matey['model']=dict(MATEY_TINY)
        matey['data'].update(cache_dir=str(cache),train=[1],val=[3],test=[5,6],gaps=[])
        matey['training'].update(epochs=1,device='cpu')
        yala=load_yala(ROOT/'yalamanchi_vit/config-runtime.yaml')
        yala['model']={**YALA_TINY,'grid_size':16}
        yala['data']=matey['data'].copy()
        yala['training'].update(epochs=1,device='cpu')
        self.matey=self.root/'matey.yaml';self.yala=self.root/'yala.yaml'
        self.matey.write_text(yaml.safe_dump(matey));self.yala.write_text(yaml.safe_dump(yala))

    def test_fresh_train_common_origins_metrics_and_all_plots(self):
        output=self.root/'comparison'
        with contextlib.redirect_stdout(io.StringIO()):
            prepare(output,self.matey,self.yala,device='cpu')
        spec=json.loads((output/'experiment.json').read_text())
        self.assertEqual(spec['test_windows'],11)
        self.assertEqual(spec['models']['matey']['training_windows'],2)
        self.assertEqual(spec['models']['yalamanchi']['training_windows'],5)
        self.assertEqual(spec['test_starts'][0]['time'],53*.03)
        for name in ('matey','yalamanchi'):
            with contextlib.redirect_stdout(io.StringIO()):worker(output,name)
            self.assertTrue((output/f'{name}-smoke.json').is_file())
            for filename in ('best.pt','last.pt','evaluation-test-5.json','first-forecast.npz',
                             'energy-data.npz','energy-trajectories.csv','energy-spectra.csv'):
                self.assertTrue((output/name/filename).is_file(),filename)
            for filename in ('energy-evolution.png','energy-example.png','energy-spectra.png'):
                self.assertEqual((output/name/filename).read_bytes()[:8],b'\x89PNG\r\n\x1a\n')
            report=json.loads((output/name/'evaluation-test-5.json').read_text())
            with np.load(output/name/'energy-data.npz') as arrays:
                self.assertEqual(arrays['actual_energy'].shape,(11,6))
                np.testing.assert_allclose(arrays['predicted_energy'][:,0],arrays['actual_energy'][:,0])
                for lead,row in enumerate(report['rollout']['horizons'],1):
                    self.assertAlmostEqual(row['predicted_kinetic_energy'],arrays['predicted_energy'][:,lead].mean(),places=10)
                    self.assertAlmostEqual(row['target_kinetic_energy'],arrays['actual_energy'][:,lead].mean(),places=10)
                    self.assertAlmostEqual(arrays['actual_spectrum'][lead].sum(),arrays['actual_energy'][:,lead].mean(),places=10)
        summarize(output)
        result=json.loads((output/'comparison.json').read_text())
        self.assertTrue(result['complete'])
        self.assertEqual(len((output/'comparison.csv').read_text().splitlines()),11)
        self.assertEqual((output/'energy-comparison.png').read_bytes()[:8],b'\x89PNG\r\n\x1a\n')
        with self.assertRaises(FileExistsError),contextlib.redirect_stdout(io.StringIO()):
            prepare(output,self.matey,self.yala,device='cpu')
        # Same starts alone are insufficient: reject divergent actual-field energy.
        path=output/'yalamanchi'/'energy-data.npz'
        with np.load(path) as data:arrays={key:data[key].copy() for key in data.files}
        arrays['actual_energy'][0,0]+=1
        np.savez_compressed(path,**arrays)
        with self.assertRaisesRegex(ValueError,'Actual evaluation data differ'):summarize(output)

    def test_incomplete_summary_preserves_status(self):
        output=self.root/'empty';output.mkdir()
        with self.assertRaisesRegex(RuntimeError,'Incomplete'):summarize(output)
        self.assertFalse(json.loads((output/'comparison.json').read_text())['complete'])


if __name__=='__main__':
    unittest.main()
