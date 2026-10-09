"""CPU tests of actual TurbT, forecast causality, normalized loss and checkpoints."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import load_config
from .dataset import HistoryDataset
from .engine import evaluate, normalized_mse, restore, train
from .matey.compat import attention_backend, flash_attn_func
from .model import MateyForecast
from .rollout import forecast_five

TINY = dict(grid_size=16, in_channels=3, history=4, patch_size=2, embed_dim=24,
            num_heads=3, processor_blocks=3, nlevels=3, filtersize=2,
            upsampling='learned', drop_path=0., attention_backend='sdpa')


class ForecastTests(unittest.TestCase):
    def test_recursive_history_and_gradients(self):
        class Scale(nn.Module):
            def __init__(self):
                super().__init__(); self.weight = nn.Parameter(torch.tensor(2.)); self.inputs = []
            def forward(self, history, times, target_time):
                self.inputs.append((history.detach().clone(), times.detach().clone(), target_time.detach().clone()))
                return history[:, -1] * self.weight
        model = Scale()
        x = torch.arange(1., 5.).reshape(1,4,1,1,1,1).expand(1,4,3,2,2,2).clone()
        times = torch.arange(9,dtype=torch.float64)[None]*0.03
        out = forecast_five(model,x,times)
        torch.testing.assert_close(out[0,:,0,0,0,0],torch.tensor([8.,16.,32.,64.,128.]))
        torch.testing.assert_close(model.inputs[1][0][0,:,0,0,0,0],torch.tensor([2.,3.,4.,8.]))
        torch.testing.assert_close(model.inputs[-1][1],times[:,4:8])
        out[:,-1].sum().backward()
        self.assertAlmostEqual(model.weight.grad.item(), 3*8*4*5*2**4)

    def test_loss_matches_baseline_normalization(self):
        p,t = torch.randn(2,5,3,2,2,2),torch.randn(2,5,3,2,2,2)
        stats = {'mean':[2.,-3.,7.], 'std':[1.,2.,4.]}
        mean = torch.tensor(stats['mean']).reshape(1,1,3,1,1,1)
        std = torch.tensor(stats['std']).reshape(1,1,3,1,1,1)
        torch.testing.assert_close(normalized_mse(p,t,stats),((p-mean)/std-(t-mean)/std).square().mean())

    def test_bad_times_and_nonfinite_forecast(self):
        class Bad(nn.Module):
            def forward(self,x,*_): return x[:,-1]*float('nan')
        x=torch.ones(1,4,3,2,2,2)
        with self.assertRaises(ValueError): forecast_five(Bad(),x,torch.zeros(1,9))
        with self.assertRaises(FloatingPointError): forecast_five(Bad(),x,torch.arange(9)[None].double())

    def test_sdpa_layout_against_explicit_attention(self):
        q,k,v=[torch.randn(2,7,3,8,dtype=torch.float64) for _ in range(3)]
        a,b,c=[x.transpose(1,2) for x in (q,k,v)]
        expected=((a@b.transpose(-1,-2)/8**0.5).softmax(-1)@c).transpose(1,2)
        with attention_backend('sdpa'): actual=flash_attn_func(q,k,v)
        torch.testing.assert_close(actual,expected)

    def test_upstream_history_dependence_and_backward(self):
        torch.manual_seed(5)
        model=MateyForecast(**TINY,dt_reference=.03,dt_rtol=.003,dt_atol=1e-10).eval()
        x=torch.randn(1,4,3,16,16,16,requires_grad=True)
        times=torch.arange(9,dtype=torch.float64)[None]*.03
        output=forecast_five(model,x,times)
        self.assertEqual(output.shape,(1,5,3,16,16,16))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad[:,:3].abs().sum().item(),0)
        for level in range(3):
            grad=model.core.module_blocks[str(level)][0].spatialtemporal.input_head.weight.grad
            self.assertIsNotNone(grad)
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.abs().sum().item(),0)
        with torch.no_grad():
            swapped=x.detach().clone();swapped[:,0]=swapped[:,0].flip(-1)
            one=model(x.detach(),times[:,:4],times[:,4])
            other=model(swapped,times[:,:4],times[:,4])
            self.assertFalse(torch.equal(one,other))
            torch.testing.assert_close(one,model(x.detach(),times[:,:4],times[:,4]),rtol=0,atol=0)


class DataTests(unittest.TestCase):
    def test_cross_file_overlap_and_training_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for fid,start in [(1,0),(2,9)]:
                arr=np.broadcast_to(np.arange(start,start+10,dtype=np.float32)[:,None,None,None,None],(10,3,2,2,2)).copy()
                np.save(root/f'fluid_{fid}.npy',arr)
            index=SimpleNamespace(groups={'train':[1,2],'val':[1,2]},
                                  times={1:np.arange(10,dtype=float),2:np.arange(9,19,dtype=float)},
                                  dt_reference=1.,path=lambda i:root/f'fluid_{i}.npy')
            within=HistoryDataset(index,'train')
            self.assertEqual(len(within),4)
            cross=HistoryDataset(index,'val',cross_file=True)
            self.assertEqual(len(cross),11)
            x,y,t=cross[6]
            np.testing.assert_array_equal(t.numpy(),np.arange(6,15))
            np.testing.assert_array_equal(x[:,0,0,0,0].numpy(),np.arange(6,10))
            np.testing.assert_array_equal(y[:,0,0,0,0].numpy(),np.arange(10,15))
            with self.assertRaises(ValueError):HistoryDataset(index,'train',cross_file=True)
            arr=np.load(root/'fluid_2.npy');arr[0]+=1;np.save(root/'fluid_2.npy',arr)
            with self.assertRaisesRegex(ValueError,'Conflicting repeated'):HistoryDataset(index,'val',cross_file=True)

    def test_train_restore_evaluate(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cache=root/'cache';cache.mkdir()
            rng=np.random.default_rng(8)
            times={}
            for fid in [1,3,5]:
                times[str(fid)]=((np.arange(10)+10*fid)*.03).tolist()
                np.save(cache/f'fluid_{fid}.npy',rng.normal(size=(10,3,16,16,16)).astype(np.float32))
            manifest=dict(channels=['u','v','w'],target_grid=[16]*3,var_dims=['time','x','y','z'],
                          file_ids=[1,3,5],time_coordinates=times)
            (cache/'manifest.json').write_text(json.dumps(manifest))
            cfg=load_config(Path(__file__).with_name('config.yaml'))
            cfg['model']=TINY.copy()
            cfg['data'].update(cache_dir=str(cache),train=[1],val=[3],test=[5],gaps=[])
            cfg['training'].update(output_dir=str(root/'run'),device='cpu',epochs=1,max_train_windows=1)
            with contextlib.redirect_stdout(io.StringIO()):train(cfg)
            model,saved,index=restore(root/'run/best.pt',torch.device('cpu'))
            self.assertEqual(saved['epoch'],1)
            report=evaluate(model,DataLoader(HistoryDataset(index,'test'),batch_size=1),saved['normalization'],torch.device('cpu'))
            self.assertEqual(report['samples'],2)
            self.assertEqual(len(report['horizons']),5)
            self.assertTrue(np.isfinite(report['mean_normalized_mse']))
            self.assertTrue((root/'run/last.pt').is_file())
            # Enforce provenance: changing metadata invalidates the saved checkpoint.
            (cache/'manifest.json').write_text(json.dumps(manifest)+'\n')
            with self.assertRaisesRegex(ValueError,'Cache metadata changed'):restore(root/'run/best.pt',torch.device('cpu'))


if __name__=='__main__':
    unittest.main()
