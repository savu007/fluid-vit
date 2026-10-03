"""End-to-end matched rollout comparison on temporary synthetic data."""
import contextlib
import copy
import io
import json
from pathlib import Path
import unittest

import torch
import yaml

from baseline_vit.config import load_config as baseline_config
from baseline_vit.train import main as baseline_train
from baseline_vit.evaluate import main as baseline_evaluate
from yalamanchi_vit.config import load_config as yalamanchi_config
from yalamanchi_vit import test_baseline as fixtures
from .compare import main as compare, validate_comparison
from .checkpoint import read_checkpoint


class ComparisonTests(unittest.TestCase):
    setUp = fixtures.IntegrationTests.setUp
    save_config = fixtures.IntegrationTests.save_config

    def configs(self):
        yala = copy.deepcopy(self.cfg)
        yala["training"].update(objective="rollout", horizon=5)
        baseline = copy.deepcopy(yala)
        baseline["model"] = dict(grid_size=8, in_channels=3, patch_size=2,
                                 embed_dim=12, depth=1, num_heads=3, mlp_ratio=2, dropout=0.)
        return baseline, yala

    def test_defaults_match_except_architecture_and_output(self):
        root = Path(__file__).resolve().parents[1]
        configs = [baseline_config(root / "baseline_vit/config-rollout.yaml"),
                   yalamanchi_config(root / "yalamanchi_vit/config-rollout.yaml")]
        validate_comparison(configs)

    def test_comparison_rejects_mismatches(self):
        for section, key, value in (("training", "batch_size", 1), ("training", "horizon", 3),
                                     ("data", "val", [4]), ("model", "dropout", .1)):
            configs = self.configs()
            configs[1][section][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_comparison(configs)

    def test_both_config_loaders_reject_pushforward(self):
        for cfg, loader in zip(self.configs(), (baseline_config, yalamanchi_config)):
            cfg["training"]["objective"] = "pushforward"
            self.path.write_text(yaml.safe_dump(cfg))
            with self.assertRaisesRegex(ValueError, "objective"):
                loader(self.path)

    def test_complete_comparison(self):
        paths = []
        for i, cfg in enumerate(self.configs()):
            path = self.root / f"config-{i}.yaml"
            path.write_text(yaml.safe_dump(cfg))
            paths.append(path)
        output = self.root / "comparison"
        args = ["--baseline-config", str(paths[0]), "--yalamanchi-config", str(paths[1]),
                "--output", str(output)]
        with contextlib.redirect_stdout(io.StringIO()):
            compare(args)
        results = json.loads((output / "comparison.json").read_text())
        self.assertEqual(set(results), {"baseline_vit", "yalamanchi_vit"})
        records, checkpoints = [], []
        for name in results:
            run = output / name
            checkpoint = read_checkpoint(run / "best.pt", name)
            checkpoints.append(checkpoint)
            report = results[name]["val"]
            self.assertEqual(report["rollout"]["horizon"], 5)
            self.assertEqual(report["rollout"]["samples"], 2)
            self.assertEqual(report["one_step"]["samples"], 6)
            self.assertEqual((run / "velocity-val.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            record = json.loads((run / "metrics.jsonl").read_text())
            records.append(record)
            self.assertEqual(record["validation"], checkpoint["validation"])
            self.assertTrue(torch.isfinite(torch.tensor(record["training_loss"])))
            self.assertEqual(checkpoint["environment"]["initialization"], None)
            self.assertNotIn("test", results[name])
        for key in ("normalization", "cache_fingerprint", "dt_reference"):
            self.assertEqual(checkpoints[0][key], checkpoints[1][key])
        for key in ("optimizer_updates", "training_forward_examples", "horizon", "objective"):
            self.assertEqual(records[0][key], records[1][key])
        self.assertEqual(records[0]["training_forward_examples"], 10)
        self.assertEqual(len((output / "comparison.csv").read_text().splitlines()), 11)
        with self.assertRaises(FileExistsError):
            compare(args)

    def test_legacy_single_step_checkpoint_and_rollout_evaluation(self):
        cfg, _ = self.configs()
        cfg["training"].pop("objective")
        cfg["training"].pop("horizon")
        self.path.write_text(yaml.safe_dump(cfg))
        with contextlib.redirect_stdout(io.StringIO()):
            baseline_train(["--config", str(self.path)])
        checkpoint = self.root / "run/best.pt"
        saved = read_checkpoint(checkpoint, "baseline_vit")
        self.assertNotIn("architecture", saved)
        with self.assertRaises(ValueError):
            read_checkpoint(checkpoint, "yalamanchi_vit")
        for extra, key in (([], "metrics"), (["--horizon", "5"], "rollout")):
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                baseline_evaluate(["--checkpoint", str(checkpoint), "--device", "cpu",
                                   "--split", "val", *extra])
            self.assertIn(key, json.loads(stream.getvalue()))

    def test_matching_package_file_names(self):
        root = Path(__file__).resolve().parents[1]
        for suffix in ("*.py", "*.yaml"):
            self.assertEqual({p.name for p in (root / "baseline_vit").glob(suffix)},
                             {p.name for p in (root / "yalamanchi_vit").glob(suffix)})
        self.assertTrue((root / "yalamanchi_vit/docs/architecture-for-dns.md").is_file())

    def test_slurm_workers_use_distinct_nodes(self):
        from unittest.mock import patch
        from .compare import slurm_prefixes
        with patch.dict("os.environ", {"SLURM_JOB_ID": "123",
                                     "SLURM_JOB_NODELIST": "nid[1-2]",
                                     "SLURM_CPUS_PER_TASK": "8"}):
            with patch("forecasting.compare.subprocess.check_output", return_value="nid1\nnid2\n"):
                commands = slurm_prefixes()
            self.assertIn("--nodelist=nid1", commands[0])
            self.assertIn("--nodelist=nid2", commands[1])
            for command in commands:
                self.assertIn("--gpus-per-task=1", command)
                self.assertIn("--nodes=1", command)
                self.assertIn("--exclusive", command)
            with patch("forecasting.compare.subprocess.check_output", return_value="nid1\n"):
                with self.assertRaisesRegex(ValueError, "two allocated"):
                    slurm_prefixes()

    def test_parallel_workers_overlap_and_preserve_success_on_failure(self):
        from unittest.mock import patch
        import threading
        configs = self.configs()
        paths = []
        for i, cfg in enumerate(configs):
            path = self.root / f"parallel-{i}.yaml"
            path.write_text(yaml.safe_dump(cfg))
            paths.append(path)
        barrier = threading.Barrier(2, timeout=10)
        def worker(name, cfg, output, splits, prefix):
            barrier.wait()
            if name == "baseline_vit":
                raise RuntimeError("intentional worker failure")
            return {"val": {"checkpoint_epoch": 1, "rollout": {"horizons": [
                {"lead_steps": 1, "physical_mse": 0.1}]}}}
        output = self.root / "partial"
        with patch("forecasting.compare.slurm_prefixes", return_value=[[], []]), \
             patch("forecasting.compare.run_model", side_effect=worker), \
             contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                compare(["--baseline-config", str(paths[0]), "--yalamanchi-config", str(paths[1]),
                         "--output", str(output), "--launcher", "slurm"])
        self.assertEqual(set(json.loads((output / "comparison.json").read_text())), {"yalamanchi_vit"})
        self.assertEqual(set(json.loads((output / "failures.json").read_text())), {"baseline_vit"})


    def test_runtime_configuration_and_retired_loss_rejection(self):
        from .compare import validate_comparison, validate_matched_specs
        root = Path(__file__).resolve().parents[1]
        configs = [baseline_config(root / "baseline_vit/config-runtime.yaml"),
                   yalamanchi_config(root / "yalamanchi_vit/config-runtime.yaml")]
        validate_comparison(configs)
        self.assertEqual(configs[0]["model"]["depth"], 10)
        self.assertEqual(sum(configs[1]["model"]["depths"]), 10)
        self.assertEqual(configs[0]["model"]["embed_dim"], 240)
        self.assertEqual(configs[0]["model"]["num_heads"], 10)
        self.assertEqual(configs[0]["model"]["patch_size"], 8)
        with self.assertRaisesRegex(ValueError, "patch_size"):
            validate_matched_specs(configs)
        for name, loader in (("baseline_vit", baseline_config),
                             ("yalamanchi_vit", yalamanchi_config)):
            with self.assertRaisesRegex(ValueError, "Top-level"):
                loader(root / "archive/physical_loss" / name / "config-physical.yaml")

    def test_matched_specs_configuration(self):
        from .compare import validate_matched_specs
        root = Path(__file__).resolve().parents[1]
        configs = [baseline_config(root / "baseline_vit/config-matched.yaml"),
                   yalamanchi_config(root / "yalamanchi_vit/config-matched.yaml")]
        report = validate_matched_specs(configs)
        self.assertEqual(report["transformer_blocks"], 10)
        self.assertEqual(report["head_dimension"], 24)
        self.assertEqual(report["patch_size"], 4)
        self.assertEqual(report["parameters"],
                         {"baseline_vit": 8020272, "yalamanchi_vit": 8313683})
        self.assertLess(report["parameter_gap_fraction_of_yalamanchi"], .05)

    def test_matched_specs_rejects_architecture_capacity_mismatches(self):
        from .compare import validate_matched_specs
        root = Path(__file__).resolve().parents[1]
        original = [baseline_config(root / "baseline_vit/config-matched.yaml"),
                    yalamanchi_config(root / "yalamanchi_vit/config-matched.yaml")]
        for changes, message in (({"patch_size": 8}, "patch_size"),
                                 ({"depth": 6}, "blocks"),
                                 ({"num_heads": 8}, "head dimensions"),
                                 ({"mlp_ratio": 3}, "mlp_ratio"),
                                 ({"embed_dim": 312, "num_heads": 13}, "parameter counts")):
            configs = copy.deepcopy(original)
            configs[0]["model"].update(changes)
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                validate_matched_specs(configs)

    def test_matched_models_rollout_gradients_on_small_grid(self):
        from .checkpoint import components
        from .rollout import forecast_loss
        root = Path(__file__).resolve().parents[1]
        for name in ("baseline_vit", "yalamanchi_vit"):
            with self.subTest(architecture=name):
                loader, model_class, _ = components(name)
                cfg = loader(root / name / "config-matched.yaml")
                cfg["model"]["grid_size"] = 16
                model = model_class(**cfg["model"])
                x = torch.randn(1, 3, 16, 16, 16)
                torch.testing.assert_close(model(x, torch.ones(1)), x, rtol=0, atol=0)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                targets = torch.stack((x + .1, x + .2), dim=1)
                for step in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    loss = forecast_loss(model, x, targets, torch.ones(1, 2), "rollout")
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                                        for p in model.parameters()))
                    if step == 1:
                        self.assertGreater(model.patch_embed.weight.grad.abs().sum().item(), 0)
                    optimizer.step()



if __name__ == "__main__":
    unittest.main()
