"""Objective, checkpoint recomputation, and complete tiny experiment tests."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from ..test_baseline import TINY
from ..config import load_config
from ..model import MateyForecast
from .model import CheckpointedMatey, pushforward_five
from .run import Execution, smoke, train, restore, update, selection_metric, load_experiment
from baseline_vit.dataset import CacheIndex, training_statistics


class MemoryTests(unittest.TestCase):
    def test_pushforward_gradient_and_history(self):
        class Scale(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.weight = torch.nn.Parameter(torch.tensor(2.)); self.calls = []
            def forward(self, history, times, target):
                self.calls.append((torch.is_grad_enabled(), history.detach().clone(), times.clone(), target.clone()))
                return history[:, -1] * self.weight
        model = Scale()
        history = torch.arange(1., 5.).reshape(1, 4, 1, 1, 1, 1).expand(1, 4, 3, 2, 2, 2)
        times = torch.arange(9)[None].double()
        pred = pushforward_five(model, history, times)
        pred.mean().backward()
        self.assertEqual([c[0] for c in model.calls], [False] * 4 + [True])
        self.assertEqual(pred.mean().item(), 128.)
        # Only the final multiply contributes: full rollout would give 320.
        self.assertEqual(model.weight.grad.item(), 64.)
        torch.testing.assert_close(model.calls[-1][1][0, :, 0, 0, 0, 0], torch.tensor([8., 16., 32., 64.]))
        torch.testing.assert_close(model.calls[-1][2], times[:, 4:8])
        with torch.no_grad():
            self.assertFalse(pushforward_five(model, history, times).requires_grad)

    def test_checkpoint_stochastic_outputs_gradients_and_state_keys(self):
        kwargs = dict(TINY, processor_blocks=6, drop_path=.2,
                      dt_reference=1., dt_rtol=.003, dt_atol=1e-10)
        torch.manual_seed(10)
        original = MateyForecast(**kwargs).train()
        checked = CheckpointedMatey(**kwargs).train()
        checked.load_state_dict(original.state_dict())
        history = torch.randn(1, 4, 3, 16, 16, 16)
        times = torch.arange(9)[None].double()
        predictions = []
        for model in (original, checked):
            torch.manual_seed(5)
            pred = pushforward_five(model, history, times)
            pred.square().mean().backward(); predictions.append(pred.detach())
        torch.testing.assert_close(*predictions, rtol=0, atol=0)
        for (name, p), (_, q) in zip(original.named_parameters(), checked.named_parameters()):
            if p.grad is None: self.assertIsNone(q.grad, name)
            else: torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0, msg=name)

    def test_one_step_uses_observed_history_first_target_and_gradient(self):
        class Scale(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.weight = torch.nn.Parameter(torch.tensor(2.)); self.calls = []
            def forward(self, history, times, target):
                self.calls.append((history.clone(), times.clone(), target.clone(), torch.is_grad_enabled()))
                return history[:, -1] * self.weight
        model = Scale()
        history = torch.ones(1, 4, 3, 2, 2, 2)
        targets = torch.full((1, 5, 3, 2, 2, 2), 99.)
        targets[:, 0] = 3.
        times = torch.arange(9)[None].double()
        cfg = {'training': {'objective': 'one_step_normalized_mse', 'grad_clip': 100.}}
        opt = torch.optim.SGD(model.parameters(), lr=.1)
        loss, _ = update(model, (history, targets, times), opt, {'std': [1., 1., 1.]},
                         cfg, Execution('baseline', torch.device('cpu')))
        self.assertAlmostEqual(loss, 1.)
        self.assertAlmostEqual(model.weight.item(), 2.2, places=6)
        self.assertEqual(len(model.calls), 1)
        torch.testing.assert_close(model.calls[0][0], history)
        torch.testing.assert_close(model.calls[0][1], times[:, :4])
        torch.testing.assert_close(model.calls[0][2], times[:, 4])
        self.assertTrue(model.calls[0][3])
        self.assertEqual(selection_metric({'horizons': [{'normalized_mse': .1}],
                                          'mean_normalized_mse': .9}, cfg), .1)
        loaded, _ = load_experiment(Path(__file__).with_name('one_step.yaml'))
        self.assertEqual(loaded['training']['epochs'], 25)
        self.assertEqual(loaded['training']['horizon'], 1)

    def test_rollout_supervises_all_leads_with_full_gradient(self):
        class Scale(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(torch.tensor(2.));self.grad_modes=[]
            def forward(self,history,times,target):
                self.grad_modes.append(torch.is_grad_enabled())
                return history[:,-1]*self.weight
        model=Scale();history=torch.full((1,4,3,2,2,2),4.)
        targets=torch.zeros(1,5,3,2,2,2);times=torch.arange(9)[None].double()
        cfg={'training':{'objective':'rollout_normalized_mse','grad_clip':1e10}}
        optimizer=torch.optim.SGD(model.parameters(),lr=0.)
        loss,_=update(model,(history,targets,times),optimizer,{'std':[1.,1.,1.]},cfg,Execution('baseline',torch.device('cpu')))
        expected_loss=sum(16*2**(2*k) for k in range(1,6))/5
        expected_gradient=sum(32*k*2**(2*k-1) for k in range(1,6))/5
        self.assertAlmostEqual(loss,expected_loss,places=2)
        self.assertAlmostEqual(model.weight.grad.item(),expected_gradient,places=2)
        self.assertEqual(model.grad_modes,[True]*5)

    def test_checkpointed_full_rollout_gradient_parity(self):
        from ..rollout import forecast_five
        kwargs=dict(TINY,processor_blocks=6,drop_path=.2,dt_reference=1.,dt_rtol=.003,dt_atol=1e-10)
        original=MateyForecast(**kwargs).train();checked=CheckpointedMatey(**kwargs).train()
        checked.load_state_dict(original.state_dict())
        x=torch.randn(1,4,3,16,16,16);times=torch.arange(9)[None].double();outputs=[]
        for model in (original,checked):
            torch.manual_seed(17);pred=forecast_five(model,x,times)
            pred.square().mean().backward();outputs.append(pred.detach())
        torch.testing.assert_close(*outputs,rtol=0,atol=0)
        for (name,p),(_,q) in zip(original.named_parameters(),checked.named_parameters()):
            if p.grad is None:self.assertIsNone(q.grad,name)
            else:torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0,msg=name)

    def test_rollout_end_to_end(self):
        self.run_end_to_end('rollout_normalized_mse')

    def test_one_step_end_to_end(self):
        self.run_end_to_end('one_step_normalized_mse')

    def test_end_to_end_acceptance_train_reload_energy(self):
        self.run_end_to_end('pushforward_final_normalized_mse')

    def run_end_to_end(self, objective):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); cache = root / 'cache'; cache.mkdir(); output = root / 'run'; output.mkdir()
            times = {}; rng = np.random.default_rng(7)
            for fid in (1, 3, 5):
                times[str(fid)] = (np.arange(10) + fid * 10).astype(float).tolist()
                np.save(cache / f'fluid_{fid}.npy', rng.normal(size=(10, 3, 16, 16, 16)).astype('float32'))
            manifest = dict(channels=['u', 'v', 'w'], target_grid=[16]*3,
                var_dims=['time', 'x', 'y', 'z'], file_ids=[1, 3, 5], time_coordinates=times)
            (cache / 'manifest.json').write_text(json.dumps(manifest))
            cfg = load_config(Path(__file__).parents[1] / 'config.yaml')
            cfg['model'] = TINY.copy()
            cfg['data'].update(cache_dir=str(cache), train=[1], val=[3], test=[5], gaps=[])
            cfg['training'].update(output_dir=str(output), epochs=1, max_train_windows=1,
                                   objective=objective, horizon=1 if objective == 'one_step_normalized_mse' else 5, device='cpu')
            settings = dict(smoke_updates=2, wall_budget_seconds=120)
            index = CacheIndex(cfg); stats = training_statistics(index)
            execution = Execution('baseline', torch.device('cpu'))
            with contextlib.redirect_stdout(io.StringIO()):
                smoke(cfg, settings, index, stats, output, execution)
                train(cfg, settings, index, stats, output, execution)
            for filename in ('acceptance.json', 'best.pt', 'last.pt', 'test.json', 'complete.json',
                             'energy-data.npz', 'energy-evolution.png', 'energy-spectra.png'):
                self.assertTrue((output / filename).is_file(), filename)
            _, saved, _ = restore(output / 'best.pt', execution)
            self.assertEqual(saved['optimizer_updates'], 1)
            self.assertEqual(saved['config']['training']['objective'], objective)
            if objective == 'rollout_normalized_mse':
                self.assertEqual(saved['architecture'], 'matey_checkpointed_rollout5_v1')
                from .publish import publish
                publish(output,root/'home-report')
                self.assertTrue((root/'home-report/velocity-contours.png').is_file())
                self.assertTrue((root/'home-report/learning-curves.png').is_file())
            if objective == 'one_step_normalized_mse':
                self.assertEqual(saved['architecture'], 'matey_checkpointed_onestep_v1')
                self.assertEqual(saved['validation']['horizons'][0]['normalized_mse'],
                                 json.loads((output / 'complete.json').read_text())['best_validation'])


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
