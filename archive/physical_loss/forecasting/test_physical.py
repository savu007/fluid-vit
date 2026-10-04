"""Analytic spectral checks and temporary-cache physical-loss training."""
import contextlib
import copy
import importlib
import io
import json
from pathlib import Path
import unittest
import torch
import yaml
from .physical_loss import PhysicalForecastLoss
from .checkpoint import components, read_checkpoint, restore_model
from .train_physical import main as physical_main
from yalamanchi_vit import test_baseline as fixtures

SETTINGS = {"name": "mse_log_spectrum", "spectral_weight": .01, "energy_floor_fraction": 1e-6}
STATS = {"mean": [1., -2., .5], "std": [2., 3., 4.]}


class SpectralTests(unittest.TestCase):
    def test_parseval_with_mean_and_nyquist(self):
        loss = PhysicalForecastLoss(8, STATS, SETTINGS).double()
        field = torch.randn(2, 3, 8, 8, 8, dtype=torch.float64)
        torch.testing.assert_close(loss.spectrum(field).sum(1),
                                   .5 * field.square().sum(1).mean((1, 2, 3)))
        spec = loss.spectrum(torch.ones_like(field)*2)
        torch.testing.assert_close(spec[:, 0], torch.full((2,), 6., dtype=torch.float64))
        self.assertEqual(spec[:, 1:].abs().sum().item(), 0)
        checker = ((-1.)**torch.arange(8, dtype=torch.float64))[None, None, :, None, None]
        field.zero_()
        field[:, :1] = checker.expand(2, 1, 8, 8, 8)
        torch.testing.assert_close(loss.spectrum(field)[:, 4], torch.full((2,), .5, dtype=torch.float64))

    def test_single_mode_amplitude_and_phase(self):
        loss = PhysicalForecastLoss(8, STATS, SETTINGS).double()
        angle = torch.arange(8, dtype=torch.float64)*(2*torch.pi/8)
        field = torch.zeros(1, 3, 8, 8, 8, dtype=torch.float64)
        field[:, 0] = (2*torch.cos(2*angle))[:, None, None]
        self.assertAlmostEqual(loss.spectrum(field)[0, 2].item(), 1., places=12)
        x = (field-loss.mean)/loss.std
        y = (torch.roll(field, 1, dims=2)-loss.mean)/loss.std
        mse, spectral = loss.per_example(x, y)
        self.assertGreater(mse.item(), .01)
        self.assertLess(spectral.item(), 1e-15)
        _, changed = loss.per_example(x, (2*field-loss.mean)/loss.std)
        self.assertGreater(changed.item(), 0)

    def test_zero_identity_and_normalization(self):
        loss = PhysicalForecastLoss(4, STATS, SETTINGS).double()
        x = (-loss.mean/loss.std).expand(2, 3, 4, 4, 4).clone().requires_grad_()
        mse, spectral = loss.per_example(x, x.detach())
        (mse.mean()+spectral.mean()).backward()
        self.assertEqual(spectral.sum().item(), 0)
        self.assertTrue(torch.isfinite(x.grad).all())
        raw = torch.randn_like(x)
        rebuilt = ((raw-loss.mean)/loss.std)*loss.std+loss.mean
        torch.testing.assert_close(loss.spectrum(raw), loss.spectrum(rebuilt))

    def test_spectral_autograd(self):
        loss = PhysicalForecastLoss(4, STATS, SETTINGS).double()
        x = torch.randn(1, 3, 4, 4, 4, dtype=torch.float64, requires_grad=True)
        target = torch.randn_like(x)
        self.assertTrue(torch.autograd.gradcheck(
            lambda z: loss.per_example(z, target)[1], (x,), fast_mode=True))

    def test_rollout_keeps_all_steps_connected(self):
        class Scale(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.1))
                self.states = []
            def forward(self, x, dt):
                y = x*self.weight
                y.retain_grad()
                self.states.append(y)
                return y
        criterion = PhysicalForecastLoss(4, STATS, SETTINGS)
        model = Scale()
        x = torch.randn(1, 3, 4, 4, 4)
        targets = torch.randn(1, 3, 3, 4, 4, 4)
        total, mse, spectral = criterion.rollout(model, x, targets, torch.ones(1, 3))
        torch.testing.assert_close(total, mse+.01*spectral)
        total.backward()
        self.assertTrue(all(s.grad.abs().sum() > 0 for s in model.states))
        self.assertTrue(torch.isfinite(model.weight.grad))

    def test_invalid_loss_rejected(self):
        for key, value in (("spectral_weight", -1), ("energy_floor_fraction", 0),
                           ("spectral_weight", float("nan")), ("name", "navier_stokes")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                PhysicalForecastLoss(4, STATS, {**SETTINGS, key: value})


class PhysicalIntegrationTests(unittest.TestCase):
    setUp = fixtures.IntegrationTests.setUp
    save_config = fixtures.IntegrationTests.save_config

    def test_configs_matched_and_mse_entry_rejects_physical(self):
        from .compare import validate_matched_specs
        root = Path(__file__).resolve().parents[1]
        configs = []
        for name in ("baseline_vit", "yalamanchi_vit"):
            loader, _, _ = components(name)
            path = root/name/"config-physical.yaml"
            configs.append(loader(path, allow_physical_loss=True))
            with self.assertRaises(ValueError):
                loader(path)
        self.assertEqual(configs[0]["loss"], configs[1]["loss"])
        validate_matched_specs(configs)

    def test_both_trainers_save_load_and_evaluate(self):
        for name in ("baseline_vit", "yalamanchi_vit"):
            with self.subTest(architecture=name):
                cfg = copy.deepcopy(self.cfg)
                if name == "baseline_vit":
                    cfg["model"] = dict(grid_size=8, in_channels=3, patch_size=2,
                                        embed_dim=12, depth=1, num_heads=3, mlp_ratio=2, dropout=0.)
                cfg["loss"] = dict(SETTINGS)
                cfg["training"].update(horizon=5, objective="rollout", output_dir=str(self.root/name))
                self.path.write_text(yaml.safe_dump(cfg))
                with contextlib.redirect_stdout(io.StringIO()):
                    physical_main(name, ["--config", str(self.path)])
                output = self.root/name
                for filename, metric in (("best.pt", "mean_combined_loss"),
                                          ("best-mse.pt", "mean_normalized_mse")):
                    saved = read_checkpoint(output/filename, name)
                    self.assertEqual(saved["checkpoint_selection"], metric)
                    restored = restore_model(saved, torch.device("cpu"), name)
                    specialized = importlib.import_module(name+".model_physical").PhysicalViT(**cfg["model"])
                    specialized.load_state_dict(saved["model"])
                    x = torch.randn(1, 3, 8, 8, 8)
                    torch.testing.assert_close(restored(x, torch.ones(1)), specialized(x, torch.ones(1)))
                record = json.loads((output/"metrics.jsonl").read_text())
                self.assertAlmostEqual(record["training_loss"], record["training_mse"]+
                                       .01*record["training_spectral_loss"], places=6)
                report = json.loads((output/"evaluation-val.json").read_text())
                self.assertEqual(report["rollout"]["horizon"], 5)
                self.assertEqual(report["physical_loss"]["samples"], 2)
                self.assertEqual((output/"velocity-val.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(FileExistsError):
                    physical_main(name, ["--config", str(self.path)])


if __name__ == "__main__":
    unittest.main()
