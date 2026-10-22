"""Analytical correction scores and exact instrumentation parity on actual TurbT."""
import tempfile
import unittest
import json
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch

from ..test_baseline import TINY
from .model import CheckpointedMatey
from .corrections import capture_hierarchy, correction_scores, decompose, render_report


class CorrectionTests(unittest.TestCase):
    def test_amplitude_alignment_and_wrong_direction(self):
        required=np.arange(1.,13.).reshape(3,2,2)
        weak=correction_scores(.5*required,required)
        self.assertAlmostEqual(weak['rms_ratio'],.5)
        self.assertAlmostEqual(weak['cosine'],1.)
        self.assertAlmostEqual(weak['oracle_alpha'],2.)
        self.assertAlmostEqual(weak['error_reduction'],.75)
        wrong=correction_scores(-required,required)
        self.assertAlmostEqual(wrong['cosine'],-1.)
        self.assertAlmostEqual(wrong['error_reduction'],-3.)
        zero=correction_scores(np.zeros_like(required),required)
        self.assertIsNone(zero['cosine']);self.assertIsNone(zero['oracle_alpha'])
        self.assertEqual(zero['error_reduction'],0.)

    def test_instrumentation_parity_and_actual_assembly(self):
        torch.manual_seed(31)
        model=CheckpointedMatey(**TINY,dt_reference=1.,dt_rtol=.003,dt_atol=1e-10).eval()
        history=torch.randn(1,4,3,16,16,16)
        times=torch.arange(4)[None].double();target=torch.tensor([4.],dtype=torch.float64)
        truth=torch.randn(1,3,16,16,16)
        original_keys=set(model.core.__dict__)
        with torch.inference_mode():
            reference=model(history,times,target)
            with capture_hierarchy(model.core) as records:
                predicted=model(history,times,target)
                torch.testing.assert_close(predicted,reference,rtol=0,atol=0)
                fields,rows=decompose(model.core,records,history,truth)
                torch.testing.assert_close(fields[2]['predicted'],predicted,rtol=0,atol=0)
                self.assertEqual(set(records),{0,1,2})
                self.assertEqual(len(rows),2)
                for row in rows:
                    self.assertAlmostEqual(row['mse_after'],row['actual_prediction_mse'],places=6)
            self.assertEqual(set(model.core.__dict__),original_keys)
            torch.testing.assert_close(model(history,times,target),reference,rtol=0,atol=0)
        snapshots={level:{k:v.float().cpu().numpy() for k,v in values.items()} for level,values in fields.items()}
        for row in rows:row.update(origin=0,lead=1)
        with tempfile.TemporaryDirectory() as tmp:
            render_report(Path(tmp),rows,{1:snapshots})
            self.assertEqual(len(list(Path(tmp).glob('*.png'))),3)
            self.assertTrue((Path(tmp)/'report.html').exists())

    def test_wrappers_removed_after_failure(self):
        class Core:
            def forward(self,*args,**kwargs):raise RuntimeError('expected')
            def get_spatiotemporalfromsequence(self,*args,**kwargs):pass
            def upsampeldata(self,*args,**kwargs):pass
        core=Core();before=dict(core.__dict__)
        with self.assertRaises(RuntimeError):
            with capture_hierarchy(core):raise RuntimeError('expected')
        self.assertEqual(core.__dict__,before)

    def test_cli_checkpoint_cache_rollout_and_outputs(self):
        from ..config import load_config
        from baseline_vit.dataset import CacheIndex, training_statistics
        from .run import Execution
        from .corrections import main
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cache=root/'cache';cache.mkdir();rng=np.random.default_rng(14)
            times={}
            for fid in (1,3,5):
                times[str(fid)]=(np.arange(10)+fid*10).astype(float).tolist()
                np.save(cache/f'fluid_{fid}.npy',rng.normal(size=(10,3,16,16,16)).astype('float32'))
            manifest=dict(channels=['u','v','w'],target_grid=[16]*3,var_dims=['time','x','y','z'],
                          file_ids=[1,3,5],time_coordinates=times)
            (cache/'manifest.json').write_text(json.dumps(manifest))
            cfg=load_config(Path(__file__).parents[1]/'config.yaml');cfg['model']=TINY.copy()
            cfg['data'].update(cache_dir=str(cache),train=[1],val=[3],test=[5],gaps=[])
            cfg['training']['objective']='rollout_normalized_mse'
            index=CacheIndex(cfg);stats=training_statistics(index)
            model=Execution('baseline',torch.device('cpu')).model(cfg,index.dt_reference)
            checkpoint=root/'best.pt'
            torch.save(dict(architecture='matey_checkpointed_rollout5_v1',format_version=1,
                            variant='baseline',world_size=1,config=cfg,model=model.state_dict(),
                            normalization=stats,dt_reference=index.dt_reference,
                            cache_fingerprint=index.fingerprint,epoch=1),checkpoint)
            args=['corrections','--checkpoint',str(checkpoint),'--device','cpu','--origins','1',
                  '--horizon','2','--output',str(root/'arrays'),'--report',str(root/'report')]
            with patch('sys.argv',args):main()
            complete=json.loads((root/'report/complete.json').read_text())
            self.assertEqual(complete['horizon'],2)
            self.assertEqual(len(list((root/'arrays').glob('*.npz'))),2)
            self.assertTrue((root/'report/first-origin-slices.npz').exists())
            self.assertEqual(len(json.loads((root/'report/metrics.json').read_text())),4)


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
