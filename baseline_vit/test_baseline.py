"""Small synthetic checks for users to run explicitly; no production data needed."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import yaml

from .config import load_config
from .dataset import CacheIndex, PairDataset, training_statistics
from .engine import evaluate
from .model import BaselineViT


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = BaselineViT(grid_size=8, patch_size=2, embed_dim=24,
                                 depth=2, num_heads=3)

    def test_initial_prediction_is_persistence_and_head_learns(self):
        x = torch.randn(2, 3, 8, 8, 8)
        dt = torch.ones(2)
        prediction = self.model(x, dt)
        torch.testing.assert_close(prediction, x, rtol=0, atol=0)
        loss = (prediction - (x + 0.1)).square().mean()
        loss.backward()
        self.assertGreater(self.model.head.bias.grad.abs().sum().item(), 0)
        # A zero head blocks backbone gradients on the first step by design.
        with torch.no_grad():
            self.model.head.weight.normal_(std=0.01)
        self.model.zero_grad(set_to_none=True)
        self.model(x, dt).square().mean().backward()
        self.assertGreater(self.model.patch_embed.weight.grad.abs().sum().item(), 0)

    def test_patch_layout_roundtrip(self):
        x = torch.arange(3 * 8 ** 3).reshape(1, 3, 8, 8, 8).float()
        patches = x.unfold(2, 2, 2).unfold(3, 2, 2).unfold(4, 2, 2)
        tokens = patches.permute(0, 2, 3, 4, 1, 5, 6, 7).reshape(1, 64, 24)
        torch.testing.assert_close(self.model.unpatchify(tokens), x, rtol=0, atol=0)

    def test_bad_shape_rejected(self):
        with self.assertRaises(ValueError):
            self.model(torch.zeros(1, 3, 8, 8, 4), torch.ones(1))


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        cfg = load_config(Path(__file__).with_name("config.yaml"))
        cfg["data"].update(cache_dir=str(root), train=[1], val=[3], test=[5], gaps=[2, 4])
        cfg["model"].update(grid_size=8, patch_size=2, embed_dim=24, depth=2, num_heads=3)
        self.cfg = cfg
        self.root = root
        self.manifest = {
            "file_ids": [1, 2, 3, 4, 5], "channels": ["u", "v", "w"],
            "target_grid": [8, 8, 8], "var_dims": ["time", "x", "y", "z"],
            "time_coordinates": {str(i): [i * 10.0, i * 10.0 + 1, i * 10.0 + 2]
                                 for i in range(1, 6)},
        }
        for i in range(1, 6):
            arr = np.stack([np.full((3, 8, 8, 8), i * 10 + t, dtype=np.float32)
                            for t in range(3)])
            np.save(root / f"fluid_{i}.npy", arr)
        self.write_manifest()

    def write_manifest(self):
        (self.root / "manifest.json").write_text(json.dumps(self.manifest))

    def test_pairs_and_training_only_statistics(self):
        index = CacheIndex(self.cfg)
        dataset = PairDataset(index, "train")
        self.assertEqual(dataset.pairs, [(1, 0), (1, 1)])
        x, y, dt = dataset[1]
        self.assertEqual(x.mean().item(), 11)
        self.assertEqual(y.mean().item(), 12)
        self.assertEqual(dt.item(), 1)
        stats = training_statistics(index)
        np.testing.assert_allclose(stats["mean"], [11] * 3)
        np.testing.assert_allclose(stats["std"], [np.sqrt(2 / 3)] * 3)
        self.assertEqual(stats["file_ids"], [1])

    def test_time_overlap_rejected(self):
        self.manifest["time_coordinates"]["3"] = [10, 11, 12]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "overlap"):
            CacheIndex(self.cfg)

    def test_time_spacing_rejected(self):
        self.manifest["time_coordinates"]["5"] = [50, 51, 53]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "spacing"):
            CacheIndex(self.cfg)

    def test_config_overlap_rejected(self):
        self.cfg["data"]["test"] = [1]
        path = self.root / "config.yaml"
        path.write_text(yaml.safe_dump(self.cfg))
        with self.assertRaisesRegex(ValueError, "disjoint"):
            load_config(path)

    def test_evaluation_matches_persistence_initially(self):
        index = CacheIndex(self.cfg)
        loader = torch.utils.data.DataLoader(PairDataset(index, "val"), batch_size=2)
        model = BaselineViT(**self.cfg["model"])
        metrics = evaluate(model, loader, training_statistics(index), torch.device("cpu"))
        self.assertAlmostEqual(metrics["physical_mse"], metrics["persistence_mse"], places=4)
        self.assertAlmostEqual(metrics["mse_skill_vs_persistence"], 0, places=4)


if __name__ == "__main__":
    unittest.main()
