"""CPU-only tests of indexing, topology, attention, and model gradients."""

import math
import unittest

import torch

from .model import WindowAttention3D
from .model import SwinBlock3D, ConvSkip3D
from .model import YalamanchiViT
from .model import space_to_depth, depth_to_space, partition, reverse

TINY = dict(grid_size=8, in_channels=3, patch_size=2, embed_dim=6,
            depths=[2, 2, 2, 2, 2], num_heads=[1, 2, 4],
            window_size=2, mlp_ratio=2, dropout=0.0, boundary="periodic")


class SpatialTests(unittest.TestCase):
    def test_child_order_and_inverse(self):
        x = torch.arange(1 * 4 * 4 * 4 * 2).reshape(1, 4, 4, 4, 2)
        packed = space_to_depth(x, 2)
        expected = torch.stack([x[0, a, b, c] for a in range(2)
                                for b in range(2) for c in range(2)]).flatten()
        torch.testing.assert_close(packed[0, 0, 0, 0], expected)
        torch.testing.assert_close(depth_to_space(packed, 2), x)

    def test_window_roundtrip_and_shift(self):
        x = torch.arange(2 * 8**3 * 3).reshape(2, 8, 8, 8, 3)
        for window in (1, 2, 4, 8):
            shifted = torch.roll(x, (-1, -1, -1), (1, 2, 3))
            restored = reverse(partition(shifted, window), window, 8)
            torch.testing.assert_close(torch.roll(restored, (1, 1, 1), (1, 2, 3)), x)

    def test_periodic_face_neighbors_share_shifted_windows(self):
        side = 4
        ids = torch.arange(side**3).reshape(1, side, side, side, 1)
        groups = partition(torch.roll(ids, (-1, -1, -1), (1, 2, 3)), 2).squeeze(-1)
        for axis in range(3):
            first, last = [0, 0, 0], [0, 0, 0]
            last[axis] = side - 1
            a, b = ids[(0, *first, 0)], ids[(0, *last, 0)]
            self.assertTrue(any(bool((g == a).any() and (g == b).any()) for g in groups))
        periodic = SwinBlock3D(6, 2, 4, 2, True, 2, 0, "periodic")
        finite = SwinBlock3D(6, 2, 4, 2, True, 2, 0, "finite")
        self.assertIsNone(periodic.mask)
        self.assertTrue(torch.isneginf(finite.mask).any())

    def test_periodic_convolution_wraps_each_face(self):
        module = ConvSkip3D(1, "periodic")
        x = torch.arange(4**3).float().reshape(1, 1, 4, 4, 4)
        for axis in range(3):
            with torch.no_grad():
                module.depthwise.weight.zero_()
                module.depthwise.bias.zero_()
                offset = [1, 1, 1]
                offset[axis] = 0
                module.depthwise.weight[(0, 0, *offset)] = 1
            torch.testing.assert_close(module.depthwise(x), torch.roll(x, 1, axis + 2))

    def test_invalid_partition(self):
        with self.assertRaises(ValueError):
            partition(torch.zeros(1, 5, 5, 5, 3), 2)


class AttentionTests(unittest.TestCase):
    def test_explicit_cosine_reference(self):
        torch.manual_seed(4)
        module = WindowAttention3D(4, 2, 2).double()
        x = torch.randn(2, 8, 4, dtype=torch.float64, requires_grad=True)
        output = module(x)
        # Independent per-head/per-query loops, including affine Q/V biases.
        qkv = x @ module.qkv.weight.T
        q, k, v = qkv.split(4, dim=-1)
        q, v = q + module.q_bias, v + module.v_bias
        bias = module.relative_bias()
        references = []
        for batch in range(2):
            tokens = []
            for i in range(8):
                heads = []
                for head in range(2):
                    sl = slice(2 * head, 2 * head + 2)
                    qi = q[batch, i, sl]
                    keys = k[batch, :, sl]
                    cosine = (keys @ qi) / (keys.norm(dim=-1) * qi.norm())
                    scores = cosine * module.logit_scale[head].exp().squeeze() + bias[head, i]
                    heads.append(scores.softmax(0) @ v[batch, :, sl])
                tokens.append(torch.cat(heads))
            references.append(torch.stack(tokens))
        expected = module.projection(torch.stack(references))
        torch.testing.assert_close(output, expected)
        output.square().sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.abs().sum().item(), 0)

    def test_full_domain_minimum_image_bias(self):
        module = WindowAttention3D(6, 2, 4, periodic_full=True)
        # x=0 to x=3 in the periodic domain uses displacement +1, not -3.
        got = module.relative_coordinates[module.relative_index[0, 3 * 4 * 4]]
        expected = math.log2(1 + 8 / 3) / math.log2(8)
        torch.testing.assert_close(got, torch.tensor([expected, 0, 0]))
        self.assertTrue(torch.isfinite(WindowAttention3D(6, 2, 1, periodic_full=True).relative_bias()).all())

    def test_masked_attention_is_finite(self):
        block = SwinBlock3D(6, 2, 4, 2, True, 2, 0, "finite")
        x = torch.randn(1, 4, 4, 4, 6, requires_grad=True)
        block(x).square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())


class ModelTests(unittest.TestCase):
    def test_initial_persistence_and_gradient_flow(self):
        torch.manual_seed(3)
        model = YalamanchiViT(**TINY)
        x = torch.randn(2, 3, 8, 8, 8)
        prediction = model(x, torch.ones(2))
        torch.testing.assert_close(prediction, x, rtol=0, atol=0)
        (prediction - x - 0.2).square().mean().backward()
        self.assertGreater(model.head.weight.grad.abs().sum().item(), 0)
        with torch.no_grad():
            model.head.weight.normal_(std=0.01)
        model.zero_grad(set_to_none=True)
        model(x, torch.ones(2)).square().mean().backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(model.patch_embed.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.encoder0[0].attention.qkv.weight.grad.abs().sum().item(), 0)

    def test_unequal_batch_intervals_and_shape_errors(self):
        model = YalamanchiViT(**TINY)
        x = torch.randn(2, 3, 8, 8, 8)
        self.assertEqual(model(x, torch.tensor([1., 2.])).shape, x.shape)
        with self.assertRaises(ValueError):
            model(x, torch.ones(1))
        with self.assertRaises(ValueError):
            model(x[:, :, :4], torch.ones(2))

    def test_bad_model_configuration(self):
        for change in ({"grid_size": 10}, {"num_heads": [4, 2, 4]},
                       {"window_size": 3}, {"depths": [2, 2]}, {"boundary": "guess"},
                       {"dropout": float("nan")}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                YalamanchiViT(**{**TINY, **change})



"""Synthetic temporary-cache integration tests; no production training."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

from baseline_vit.engine import evaluate
from baseline_vit.dataset import PairDataset
from functools import partial
from forecasting.checkpoint import read_checkpoint as read_shared, check_cache, restore_model as restore_shared
read_checkpoint = partial(read_shared, architecture="yalamanchi_vit")
restore_model = partial(restore_shared, architecture="yalamanchi_vit")
from .config import load_config
from forecasting.dataset import CacheIndex, SequenceDataset, training_statistics
from .evaluate import main as evaluate_main
from .model import YalamanchiViT
from forecasting.rollout import forecast_loss, evaluate_rollout
from .train import main as train_main


class ObjectiveTests(unittest.TestCase):
    class Scale(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(2.0))
            self.grad_modes = []

        def forward(self, x, dt):
            self.grad_modes.append(torch.is_grad_enabled())
            return self.weight * x

    def test_full_rollout_gradients(self):
        model = self.Scale()
        loss = forecast_loss(model, torch.ones(1, 1), torch.zeros(1, 3, 1),
                             torch.ones(1, 3), "rollout")
        loss.backward()
        self.assertAlmostEqual(model.weight.grad.item(), 76., places=5)
        self.assertEqual(model.grad_modes, [True, True, True])
        with self.assertRaises(ValueError):
            forecast_loss(model, torch.ones(1, 1), torch.zeros(1, 3, 1),
                          torch.ones(1, 3), "unknown")

    def test_single_step_objective(self):
        model = self.Scale()
        loss = forecast_loss(model, torch.ones(1, 1), torch.zeros(1, 1, 1),
                             torch.ones(1, 1), "single_step")
        self.assertEqual(loss.item(), 4)
        with self.assertRaises(ValueError):
            forecast_loss(model, torch.ones(1, 1), torch.zeros(1, 2, 1),
                          torch.ones(1, 2), "single_step")


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        cfg = load_config(Path(__file__).with_name("config.yaml"))
        cfg["model"] = dict(TINY)
        cfg["data"].update(cache_dir=str(self.root), train=[1], val=[3], test=[5], gaps=[2, 4])
        cfg["training"].update(device="cpu", epochs=1, output_dir=str(self.root / "run"),
                               max_train_pairs=2)
        self.cfg = cfg
        times = {str(i): [i * 10.0 + t for t in range(7)] for i in range(1, 6)}
        manifest = {"file_ids": [1, 2, 3, 4, 5], "channels": ["u", "v", "w"],
                    "target_grid": [8]*3, "var_dims": ["time", "x", "y", "z"],
                    "time_coordinates": times}
        (self.root / "manifest.json").write_text(json.dumps(manifest))
        spatial = np.linspace(-1, 1, 3 * 8**3, dtype=np.float32).reshape(3, 8, 8, 8)
        for file_id in range(1, 6):
            np.save(self.root / f"fluid_{file_id}.npy",
                    np.stack([spatial + t for t in range(7)]))
        self.path = self.root / "experiment.yaml"
        self.save_config()

    def save_config(self):
        self.path.write_text(yaml.safe_dump(self.cfg))

    def test_sequence_order_and_no_cross_file_windows(self):
        index = CacheIndex(self.cfg)
        dataset = SequenceDataset(index, "train", 5)
        self.assertEqual(dataset.pairs, [(1, 0), (1, 1)])
        x, targets, dt = dataset[1]
        for step in range(5):
            torch.testing.assert_close(targets[step] - x, torch.full_like(x, step + 1))
        torch.testing.assert_close(dt, torch.ones(5))
        with self.assertRaises(ValueError):
            SequenceDataset(index, "train", 7)

    def test_known_rollout_and_fixed_initial_persistence(self):
        class Increment(nn.Module):
            def forward(self, x, dt):
                return x + dt[:, None, None, None, None]

        index = CacheIndex(self.cfg)
        dataset = SequenceDataset(index, "val", 5)
        report = evaluate_rollout(Increment(), DataLoader(dataset, batch_size=2),
                                  {"mean": [0]*3, "std": [1]*3}, torch.device("cpu"))
        self.assertLess(report["mean_normalized_mse"], 1e-12)
        for k, row in enumerate(report["horizons"], 1):
            self.assertAlmostEqual(row["persistence_mse"], k*k, places=5)
            self.assertAlmostEqual(row["mse_skill_vs_persistence"], 1, places=6)

    def test_horizon_one_agrees_with_existing_metrics(self):
        index = CacheIndex(self.cfg)
        model = YalamanchiViT(**TINY)
        stats = training_statistics(index)
        sequence_report = evaluate_rollout(model, DataLoader(SequenceDataset(index, "val"), batch_size=2),
                                           stats, torch.device("cpu"))["horizons"][0]
        reference = evaluate(model, DataLoader(PairDataset(index, "val"), batch_size=2),
                             stats, torch.device("cpu"))
        for key, value in reference.items():
            self.assertAlmostEqual(sequence_report[key], value, places=10, msg=key)

    def test_config_rejects_incompatible_objective_and_split(self):
        self.cfg["training"]["horizon"] = 5
        self.save_config()
        with self.assertRaisesRegex(ValueError, "single_step"):
            load_config(self.path)
        self.cfg["training"]["horizon"] = 1
        self.cfg["data"]["val"] = [1]
        self.save_config()
        with self.assertRaisesRegex(ValueError, "disjoint"):
            load_config(self.path)

    def test_train_checkpoint_evaluate_plot_and_rollout(self):
        with contextlib.redirect_stdout(io.StringIO()):
            train_main(["--config", str(self.path)])
        checkpoint = self.root / "run" / "best.pt"
        saved = read_checkpoint(checkpoint)
        index = CacheIndex(self.cfg)
        check_cache(saved, index)
        model = restore_model(saved, torch.device("cpu"))
        stats = saved["normalization"]
        x = torch.randn(1, 3, 8, 8, 8)
        second = restore_model(saved, torch.device("cpu"))
        torch.testing.assert_close(model(x, torch.ones(1)), second(x, torch.ones(1)), rtol=0, atol=0)
        for key in ("mean", "std", "file_ids"):
            self.assertEqual(stats[key], training_statistics(index)[key])
        plot = self.root / "forecast.png"
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            evaluate_main(["--checkpoint", str(checkpoint), "--device", "cpu", "--horizon", "5",
                           "--plot", str(plot)])
        report = json.loads(output.getvalue())
        self.assertEqual(report["one_step"]["samples"], 6)
        self.assertEqual(report["rollout"]["samples"], 2)
        self.assertEqual(plot.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        self.cfg["training"].update(output_dir=str(self.root / "fine-tune"),
                                     objective="rollout", horizon=5)
        self.save_config()
        with contextlib.redirect_stdout(io.StringIO()):
            train_main(["--config", str(self.path)])
        self.assertTrue((self.root / "fine-tune" / "best.pt").exists())
        with self.assertRaises(FileExistsError), contextlib.redirect_stdout(io.StringIO()):
            train_main(["--config", str(self.path)])
        manifest = self.root / "manifest.json"
        manifest.write_text(manifest.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "metadata"):
            check_cache(saved, CacheIndex(self.cfg))

    def test_wrong_architecture_checkpoint_rejected(self):
        path = self.root / "wrong.pt"
        torch.save({"architecture": "plain_vit", "format_version": 1}, path)
        with self.assertRaisesRegex(ValueError, "supported"):
            read_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
