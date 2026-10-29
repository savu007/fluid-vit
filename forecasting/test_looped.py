"""Gradient, geometry, gate and full tiny-pipeline tests for finest-stage loops."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from ornl_vit.test_baseline import TINY as MATEY
from yalamanchi_vit.test_baseline import TINY as YALA
from .looped import FinestStageForecast, restore_weights, tie_parameters
from .looped_experiment import profile, train, rollout_loss


def tiny(name):
    model = dict(MATEY, processor_blocks=12, drop_path=.2) if name=='matey' else dict(YALA)
    return dict(seed=42, model=model, data=dict(dt_rtol=.003, dt_atol=1e-10),
                training=dict(grad_clip=1., learning_rate=.0001, weight_decay=.01,
                              epochs=1,batch_size=1,num_workers=0))


def inputs(name):
    if name=='matey':
        return torch.randn(1,4,3,16,16,16), torch.arange(4)[None].double(), torch.tensor([4.]).double()
    return torch.randn(1,3,8,8,8), torch.ones(1)


class LoopTests(unittest.TestCase):
    def test_shared_gradient_equals_sum_of_independent_copies(self):
        first, second = nn.Linear(3,3), nn.Linear(3,3)
        second.load_state_dict(first.state_dict())
        tied1, tied2 = nn.Linear(3,3), nn.Linear(3,3)
        tied1.load_state_dict(first.state_dict()); tie_parameters(tied2,tied1)
        x=torch.randn(4,3)
        second(first(x)).square().mean().backward()
        tied2(tied1(x)).square().mean().backward()
        torch.testing.assert_close(tied1.weight.grad, first.weight.grad+second.weight.grad)
        torch.testing.assert_close(tied1.bias.grad, first.bias.grad+second.bias.grad)
        self.assertIs(tied1.weight,tied2.weight)

    def test_control_is_original_and_shared_geometry_is_preserved(self):
        for name in ('matey','yalamanchi'):
            cfg=tiny(name); torch.manual_seed(1)
            control=FinestStageForecast(name,cfg,1.,looped=False,checkpointing=False).eval()
            args=inputs(name)
            torch.testing.assert_close(control(*args),control.model(*args),rtol=0,atol=0)
            loop=FinestStageForecast(name,cfg,1.,looped=True,checkpointing=False)
            if name=='matey':
                blocks=loop.model.core.module_blocks['2']
                self.assertIs(blocks[0].spatialtemporal.input_head.weight,blocks[2].spatialtemporal.input_head.weight)
                self.assertNotEqual(type(blocks[0].spatialtemporal.drop_path),type(blocks[2].spatialtemporal.drop_path))
            else:
                blocks=loop.model.decoder0
                self.assertIs(blocks[0].attention.qkv.weight,blocks[1].attention.qkv.weight)
                self.assertEqual((blocks[0].shift,blocks[1].shift),(0,1))
            self.assertLess(sum(p.numel() for p in loop.parameters()),sum(p.numel() for p in control.parameters()))

    def test_auxiliary_matches_independent_prefix_decode_and_reload(self):
        class Identity(nn.Module):
            def forward(self,x,*args,**kwargs): return x
        for name in ('matey','yalamanchi'):
            cfg=tiny(name)
            model=FinestStageForecast(name,cfg,1.,looped=True,checkpointing=False).eval()
            args=inputs(name)
            auxiliary,final=model.supervised(*args)
            torch.testing.assert_close(final,model(*args),rtol=0,atol=0)
            reference=FinestStageForecast(name,cfg,1.,looped=True,checkpointing=False).eval()
            restore_weights(reference,model.state_dict())
            torch.testing.assert_close(reference(*args),final,rtol=0,atol=0)
            if name=='matey':
                blocks=reference.model.core.module_blocks['2']
                blocks[2]=Identity();blocks[3]=Identity()
            else: reference.model.decoder0[1]=Identity()
            torch.testing.assert_close(auxiliary,reference(*args),rtol=0,atol=0)
            bad={key:value.clone() for key,value in model.state_dict().items()}
            key=('model.core.module_blocks.2.2.spatialtemporal.input_head.weight' if name=='matey'
                 else 'model.decoder0.1.attention.qkv.weight')
            bad[key].add_(1.)
            with self.assertRaisesRegex(ValueError,'aliases'): restore_weights(model,bad)
            self.assertIsNone(model._mid); self.assertIsNone(model._candidate)

    def test_checkpoint_outputs_gradients_and_gate_learning(self):
        for name in ('matey','yalamanchi'):
            cfg=tiny(name)
            plain=FinestStageForecast(name,cfg,1.,looped=True,checkpointing=False).train()
            checked=FinestStageForecast(name,cfg,1.,looped=True,checkpointing=True).train()
            restore_weights(checked,plain.state_dict())
            if name=='yalamanchi':
                # Production head starts at zero, so first-step internal gradients
                # vanish. A nonzero head tests propagation rather than that init.
                nn.init.normal_(plain.model.head.weight,std=.02)
                restore_weights(checked,plain.state_dict())
            args=inputs(name); outputs=[]
            for model in (plain,checked):
                torch.manual_seed(9)
                a,f=model.supervised(*args)
                (.5*(a.square().mean()+f.square().mean())).backward()
                outputs.append((a.detach(),f.detach()))
            for a,b in zip(*outputs): torch.testing.assert_close(a,b,rtol=0,atol=0)
            gate_gradient=0.
            for (key,p),(_,q) in zip(plain.named_parameters(),checked.named_parameters()):
                if p.grad is None:self.assertIsNone(q.grad,key)
                else:
                    torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0,msg=key)
                    if 'loop_gate' in key: gate_gradient+=p.grad.abs().sum().item()
            self.assertGreater(gate_gradient,0.)
            self.assertIsNone(checked._mid)  # recomputation must not retain capture

    def test_full_rollout_auxiliary_supervision_and_final_state_recursion(self):
        class Scale(nn.Module):
            name='yalamanchi'; looped=True
            def __init__(self):super().__init__();self.w=nn.Parameter(torch.tensor(2.));self.states=[]
            def supervised(self,x,dt):
                self.states.append(x.detach().clone())
                return x*self.w*3, x*self.w
        model=Scale(); x=torch.ones(1,3,2,2,2);targets=torch.zeros(1,5,3,2,2,2)
        loss,a,f=rollout_loss(model,(x,targets,torch.ones(1,5)),{'mean':[0.]*3,'std':[1.]*3},torch.device('cpu'))
        loss.backward()
        expected=sum(4**k for k in range(1,6))/5
        self.assertAlmostEqual(f.item(),expected,places=3)
        self.assertAlmostEqual(a.item(),9*expected,places=3)
        self.assertAlmostEqual(loss.item(),5*expected,places=3)
        self.assertAlmostEqual(model.w.grad.item(),sum(5*k*4**k for k in range(1,6))/5,places=2)
        self.assertEqual([s.mean().item() for s in model.states],[1,2,4,8,16])

    def test_end_to_end_both_models(self):
        for name in ('matey','yalamanchi'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root=Path(directory);cache=root/'cache';cache.mkdir()
                cfg=tiny(name);side=cfg['model']['grid_size'];rng=np.random.default_rng(2)
                for fid in (1,3,5):
                    np.save(cache/f'fluid_{fid}.npy',rng.normal(size=(10,3,side,side,side)).astype('float32'))
                manifest=dict(channels=['u','v','w'],target_grid=[side]*3,var_dims=['time','x','y','z'],
                    file_ids=[1,3,5],time_coordinates={str(fid):(np.arange(10)+fid*10).tolist() for fid in (1,3,5)})
                (cache/'manifest.json').write_text(json.dumps(manifest))
                cfg['data'].update(cache_dir=str(cache),channels=['u','v','w'],train=[1],val=[3],test=[5],gaps=[])
                case=name+'-looped'
                with contextlib.redirect_stdout(io.StringIO()):
                    gate=profile(case,cfg,torch.device('cpu'),root/'profile')
                    train(case,cfg,torch.device('cpu'),root/'report',root/'weights',gate)
                for filename in ('complete.json','learning-curves.png','energy-spectra.png','velocity-contours.png','energy-evolution.png'):
                    self.assertTrue((root/'report'/filename).exists(),filename)
                self.assertTrue((root/'weights'/'last.pt').exists())


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
