"""Training-strategy gradients, stitched trajectories, and worker integration."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn
import yaml

from forecasting.dataset import CacheIndex, CrossFileSequenceDataset
from forecasting.rollout import forecast_loss
from forecasting.compare_training import worker, summarize, write_json
from forecasting.checkpoint import read_checkpoint
from yalamanchi_vit.test_baseline import IntegrationTests as Fixture


class GradientTests(unittest.TestCase):
    def test_detached_pushforward_and_full_rollout(self):
        class Scale(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(2.))
                self.modes = []
            def forward(self, x, dt):
                self.modes.append(torch.is_grad_enabled())
                return x * self.weight
        x = torch.ones(1, 1)
        targets = torch.zeros(1, 5, 1)
        dt = torch.ones(1, 5)
        push = Scale()
        loss = forecast_loss(push, x, targets, dt, "pushforward")
        loss.backward()
        self.assertEqual(loss.item(), 1024.)
        self.assertEqual(push.weight.grad.item(), 1024.)
        self.assertEqual(push.modes, [False, False, False, False, True])
        full = Scale()
        loss = forecast_loss(full, x, targets, dt, "rollout")
        loss.backward()
        expected = sum(2*k*2**(2*k-1) for k in range(1, 6))/5
        self.assertAlmostEqual(full.weight.grad.item(), expected, places=3)
        self.assertEqual(full.modes, [True]*5)
        with self.assertRaises(ValueError):
            forecast_loss(push, x, targets[:, :1], dt[:, :1], "pushforward")


class CrossFileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.times = {1: np.arange(0., 61.), 2: np.arange(60., 121.)}
        for i, times in self.times.items():
            np.save(self.root/f"fluid_{i}.npy",
                    np.broadcast_to(times[:, None, None, None, None],
                                    (len(times), 3, 2, 2, 2)).astype(np.float32).copy())
        self.index = SimpleNamespace(times=self.times, groups={"test": [1, 2]},
                                     dt_reference=1., path=lambda i: self.root/f"fluid_{i}.npy")

    def test_100_step_deduplication_and_alignment(self):
        data = CrossFileSequenceDataset(self.index, "test", 100)
        self.assertEqual(len(data.frames), 121)
        self.assertEqual(len(data), 21)
        x, targets, dt = data[20]
        self.assertEqual(x[0, 0, 0, 0], 20.)
        torch.testing.assert_close(targets[:, 0, 0, 0, 0], torch.arange(21., 121.))
        torch.testing.assert_close(dt, torch.ones(100))
        self.assertEqual(data.starts()[-1]["end_time"], 120.)

    def test_gap_and_overlap_rejected(self):
        for offset in (2., -2.):
            self.times[2] = np.arange(60., 121.) + offset
            with self.assertRaisesRegex(ValueError, "Noncontiguous"):
                CrossFileSequenceDataset(self.index, "test", 100)

    def test_conflicting_duplicate_rejected(self):
        p = self.index.path(2)
        a = np.load(p); a[0] += 1; np.save(p, a)
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            CrossFileSequenceDataset(self.index, "test", 100)

    def test_short_or_training_split_rejected(self):
        with self.assertRaises(ValueError):
            CrossFileSequenceDataset(self.index, "test", 121)
        with self.assertRaises(ValueError):
            CrossFileSequenceDataset(self.index, "train", 5)


class WorkerTests(unittest.TestCase):
    setUp = Fixture.setUp
    save_config = Fixture.save_config

    def test_both_train_save_and_test_100(self):
        # Extend held-out test only; training and validation stay tiny.
        manifest_path = self.root/"manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["time_coordinates"]["5"] = [50.+t for t in range(101)]
        manifest_path.write_text(json.dumps(manifest))
        field = np.load(self.root/"fluid_5.npy")[0]
        np.save(self.root/"fluid_5.npy", np.stack([field + .01*t for t in range(101)]))
        output = self.root/"experiment"
        output.mkdir()
        cfg = copy.deepcopy(self.cfg)
        cfg["training"].update(objective="rollout", horizon=5)
        index = CacheIndex(cfg)
        data = CrossFileSequenceDataset(index, "test", 100)
        write_json(output/"experiment.json",
                   {"test_horizon": 100, "test_starts": data.starts(),
                    "cache_fingerprint": index.fingerprint})
        for strategy in ("rollout", "pushforward"):
            cfg["training"].update(objective=strategy, output_dir=str(output/strategy))
            (output/(strategy+".yaml")).write_text(yaml.safe_dump(cfg))
            with contextlib.redirect_stdout(io.StringIO()):
                worker(output, strategy)
            for filename in ("best.pt", "last.pt"):
                saved = read_checkpoint(output/strategy/filename, "yalamanchi_vit")
                self.assertEqual(saved["epoch"], 1)
                self.assertEqual(saved["config"]["training"]["objective"], strategy)
            report = json.loads((output/strategy/"evaluation-test-100.json").read_text())
            self.assertEqual(report["rollout"]["horizon"], 100)
            self.assertEqual(report["rollout"]["samples"], 1)
            self.assertEqual(len(report["rollout"]["horizons"]), 100)
        summarize(output)
        result = json.loads((output/"comparison.json").read_text())
        self.assertTrue(result["complete"])
        self.assertEqual(len((output/"comparison.csv").read_text().splitlines()), 201)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(FileExistsError):
            worker(output, "rollout")

    def test_incomplete_summary_retains_status(self):
        output = self.root/"empty"; output.mkdir()
        with self.assertRaisesRegex(RuntimeError, "Incomplete"):
            summarize(output)
        self.assertFalse(json.loads((output/"comparison.json").read_text())["complete"])


del Fixture  # Do not rediscover the imported fixture class as another test suite.

if __name__ == "__main__":
    unittest.main()
