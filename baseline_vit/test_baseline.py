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


    def test_touching_files_within_train_are_allowed_without_cross_file_pairs(self):
        self.cfg["data"].update(train=[1, 2], gaps=[4])
        self.manifest["time_coordinates"]["2"] = [12, 13, 14]
        self.write_manifest()
        index = CacheIndex(self.cfg)
        dataset = PairDataset(index, "train")
        self.assertEqual(dataset.pairs, [(1, 0), (1, 1), (2, 0), (2, 1)])
        self.assertEqual(index.dt_reference, 1)
        self.assertTrue(all(dataset[i][2].item() == 1 for i in range(len(dataset))))

    def test_touching_train_validation_boundary_is_rejected(self):
        self.manifest["time_coordinates"]["3"] = [12, 13, 14]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "chronological train before val"):
            CacheIndex(self.cfg)

    def test_touching_validation_test_boundary_is_rejected(self):
        self.manifest["time_coordinates"]["5"] = [32, 33, 34]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "chronological val before test"):
            CacheIndex(self.cfg)

    def test_plot_contains_physical_velocity_and_correct_slices(self):
        from unittest.mock import patch
        from matplotlib.figure import Figure
        from .visualize import plot_prediction

        # Spatial variation exposes axis-order mistakes; the zero head predicts x.
        field = np.arange(3 * 8 ** 3, dtype=np.float32).reshape(3, 8, 8, 8)
        np.save(self.root / "fluid_3.npy", np.stack([field, field + 1, field + 2]))
        index = CacheIndex(self.cfg)
        dataset = PairDataset(index, "val")
        model = BaselineViT(**self.cfg["model"])
        stats = training_statistics(index)
        savefig = Figure.savefig
        figures = []

        def capture(figure, *args, **kwargs):
            figures.append(figure)
            return savefig(figure, *args, **kwargs)

        for axis_number, axis in enumerate(("x", "y", "z")):
            with self.subTest(axis=axis), patch.object(Figure, "savefig", capture):
                output = self.root / f"{axis}.png"
                plot_prediction(model, dataset, stats, torch.device("cpu"), output,
                                axis=axis, slice_index=2)
                self.assertEqual(output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
                panels = [p for p in figures[-1].axes if p.images]
                self.assertEqual(len(panels), 9)
                plane = np.take(field, 2, axis=axis_number + 1)
                for channel in range(3):
                    truth, predicted, error = panels[3 * channel:3 * channel + 3]
                    np.testing.assert_allclose(truth.images[0].get_array(), plane[channel].T + 1)
                    np.testing.assert_allclose(predicted.images[0].get_array(),
                                               plane[channel].T, atol=2e-4)
                    np.testing.assert_allclose(error.images[0].get_array(), -1, atol=2e-4)
                    self.assertEqual(truth.images[0].get_clim(), predicted.images[0].get_clim())

    def test_plot_rejects_invalid_selection(self):
        from .visualize import plot_prediction

        index = CacheIndex(self.cfg)
        dataset = PairDataset(index, "val")
        model = BaselineViT(**self.cfg["model"])
        for options in ({"sample_index": -1}, {"sample_index": len(dataset)},
                        {"slice_index": 8}, {"slice_index": -1}, {"axis": "time"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                plot_prediction(model, dataset, training_statistics(index),
                                torch.device("cpu"), self.root / "invalid.png", **options)


if __name__ == "__main__":
    unittest.main()
