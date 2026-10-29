"""Full physical-input dependency, checkpoint gradients and tiny training tests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import torch
from torch import nn
from .causal_model import OrderedMateyForecast,prefix_normalize
from .looped import restore_weights
from .test_looped import tiny,inputs
from ornl_vit.matey.data_utils.shared_utils import normalize_spatiotemporal_persample


class OrderedModelTests(unittest.TestCase):
    def test_prefix_statistics_match_upstream_latest_and_do_not_change_earlier(self):
        x=torch.randn(4,2,3,4,4,4)
        a,mean,std=prefix_normalize(x)
        b,m,s=normalize_spatiotemporal_persample(x)
        torch.testing.assert_close(a[-1],b[-1],rtol=0,atol=0)
        torch.testing.assert_close(mean[-1:],m,rtol=0,atol=0)
        torch.testing.assert_close(std[-1:],s,rtol=0,atol=0)
        altered=x.clone();altered[2:]=altered[2:]*2+8
        torch.testing.assert_close(prefix_normalize(altered)[0][:2],a[:2],rtol=0,atol=0)

    def test_full_model_prefix_invariance_at_every_block_and_level(self):
        cfg=tiny('matey');args=inputs('matey')
        for causal in (True,False):
            torch.manual_seed(4)
            model=OrderedMateyForecast(cfg,1.,causal=causal,checkpointing=False).eval()
            snapshots={};handles=[]
            for level,blocks in model.model.core.module_blocks.items():
                for i,block in enumerate(blocks):
                    # Amplify residuals to make the negative control informative.
                    block.spatialtemporal.gamma_att.data.fill_(1.)
                    block.spatialtemporal.gamma_mlp.data.fill_(1.)
                    def capture(module,values,result,key=(level,i)):
                        snapshots[key]=result.detach().clone()
                    handles.append(block.register_forward_hook(capture))
            with torch.no_grad():model(*args)
            original=dict(snapshots)
            found_noncausal=False
            for prefix in (1,2,3):
                changed=args[0].clone();changed[:,prefix:]=changed[:,prefix:]*3+7
                with torch.no_grad():model(changed,*args[1:])
                for key,before in original.items():
                    length=before.shape[-1]//4*prefix
                    difference=(before[...,:length]-snapshots[key][...,:length]).abs().max().item()
                    if causal:self.assertEqual(difference,0.,(key,prefix,difference))
                    else:found_noncausal |= difference>1e-5
            if not causal:self.assertTrue(found_noncausal)
            for h in handles:h.remove()

    def test_checkpoint_gradient_parity_and_control_initialization(self):
        cfg=tiny('matey');args=inputs('matey')
        torch.manual_seed(6);causal=OrderedMateyForecast(cfg,1.,causal=True,checkpointing=False)
        torch.manual_seed(6);control=OrderedMateyForecast(cfg,1.,causal=False,checkpointing=False)
        self.assertEqual(sum(p.numel() for p in causal.parameters()),sum(p.numel() for p in control.parameters()))
        for a,b in zip(causal.parameters(),control.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
        checked=OrderedMateyForecast(cfg,1.,causal=True,checkpointing=True)
        restore_weights(checked,causal.state_dict())
        for model in (causal,checked):
            torch.manual_seed(11);prediction=model(*args);prediction.square().mean().backward()
            if model is causal:expected=prediction.detach()
            else:torch.testing.assert_close(prediction,expected,rtol=0,atol=0)
        for (name,a),(_,b) in zip(causal.named_parameters(),checked.named_parameters()):
            if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0,msg=name)
        # No shared process-global normalization replacement.
        x=torch.randn(4,1,3,4,4,4)
        self.assertEqual(normalize_spatiotemporal_persample(x)[1].shape[0],1)
        reloaded=OrderedMateyForecast(cfg,1.,causal=True,checkpointing=False).eval()
        restore_weights(reloaded,causal.state_dict());causal.eval()
        torch.testing.assert_close(reloaded(*args),causal(*args),rtol=0,atol=0)



class PipelineTests(unittest.TestCase):
    def test_profile_segment_resume_evaluate_and_compare(self):
        import contextlib
        import io
        import numpy as np
        import shutil
        from .causal_experiment import profile,segment
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root=Path(directory);cache=root/'cache';cache.mkdir()
            cfg=tiny('matey');cfg['training']['epochs']=2
            side=cfg['model']['grid_size'];rng=np.random.default_rng(2)
            for fid in (1,3,5):
                np.save(cache/f'fluid_{fid}.npy',rng.normal(size=(10,3,side,side,side)).astype('float32'))
            manifest=dict(channels=['u','v','w'],target_grid=[side]*3,var_dims=['time','x','y','z'],
                file_ids=[1,3,5],time_coordinates={str(fid):(np.arange(10)+fid*10).tolist() for fid in (1,3,5)})
            (cache/'manifest.json').write_text(json.dumps(manifest))
            cfg['data'].update(cache_dir=str(cache),channels=['u','v','w'],train=[1],val=[3],test=[5],gaps=[])
            reports=root/'reports';weights=root/'weights';device=torch.device('cpu')
            for case in ('causal','unmasked'):
                profile(case,cfg,reports,device)
                segment(case,cfg,reports,weights,device,end_epoch=1)
                self.assertFalse((reports/case/'complete.json').exists())
                segment(case,cfg,reports,weights,device)
                for filename in ('complete.json','learning-curves.png','energy-spectra.png','velocity-contours.png','energy-evolution.png'):
                    self.assertTrue((reports/case/filename).exists(),filename)
                # Continuous and segmented runs must converge to exactly the same weights.
                other=root/('continuous-'+case);(other/case).mkdir(parents=True)
                shutil.copy2(reports/case/'profile.json',other/case/'profile.json')
                continuous=root/('continuous-weights-'+case)
                segment(case,cfg,other,continuous,device)
                a=torch.load(weights/case/'last.pt',weights_only=False)
                b=torch.load(continuous/case/'last.pt',weights_only=False)
                self.assertEqual(a['optimizer_updates'],b['optimizer_updates'])
                for key in a['model']:torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)
                # Provenance checks must reject a meaningful change on resume.
                changed=copy.deepcopy(cfg);changed['seed']+=1
                with self.assertRaisesRegex(ValueError,'mismatch'):
                    segment(case,changed,reports,weights,device)
            self.assertTrue((reports/'comparison.csv').exists())
            self.assertTrue((reports/'comparison.json').exists())

if __name__=='__main__':
    torch.set_num_threads(2);unittest.main()
