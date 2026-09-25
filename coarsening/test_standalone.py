"""Small synthetic checks; never read production data."""

import tempfile
import unittest
from pathlib import Path

import yaml

from coarsening.config import CoarsenConfig
from coarsening.standalone import discover_files, source_signature


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = {
            "root": str(self.root), "file_glob": "*.nc",
            "channels": ["u"], "var_dims": ["time", "x", "y", "z"],
            "source_grid": [16, 16, 16], "snapshots_per_file": 2,
            "periodic": True, "splits": {"train": [25]},
            "coarsen": {"method": "spectral_sharp", "target_grid": [8, 8, 8],
                        "cache_dir": str(self.root / "cache")},
        }

    def parse(self, data):
        path = self.root / "test.yaml"
        path.write_text(yaml.safe_dump(data))
        return CoarsenConfig.from_yaml(path)

    def test_nested_and_top_level(self):
        self.assertEqual(self.parse(self.config), self.parse({"data": self.config}))

    def test_overlap_rejected(self):
        self.config["splits"]["val"] = [25]
        with self.assertRaisesRegex(ValueError, "disjoint"):
            self.parse(self.config)

    def test_bad_grid_rejected(self):
        for target in ([7, 7, 7], [8, 8, 4], [32, 32, 32]):
            self.config["coarsen"]["target_grid"] = target
            with self.assertRaises(ValueError):
                self.parse(self.config)

    def test_missing_physics_rejected(self):
        del self.config["periodic"]
        with self.assertRaisesRegex(ValueError, "periodic"):
            self.parse(self.config)

    def test_file_discovery_and_signature(self):
        path = self.root / "fluid_25.nc"
        path.write_bytes(b"fixture")
        self.assertEqual(discover_files(self.root, "*.nc", self.parse(self.config).file_id_pattern), {25: path})
        before = source_signature([path])
        path.write_bytes(b"changed fixture")
        self.assertNotEqual(before, source_signature([path]))
        (self.root / "fluid_025.nc").touch()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            discover_files(self.root, "*.nc", self.parse(self.config).file_id_pattern)


class OperatorTests(unittest.TestCase):
    def test_netcdf_pass_and_cache(self):
        import numpy as np
        import xarray as xr
        import torch
        from coarsening.coarsen import run_coarsen_pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            coords = {"time": [0., 0.5], "x": np.arange(16),
                      "y": np.arange(16), "z": np.arange(16)}
            # Store axes out of order to exercise explicit transposition.
            ds = xr.Dataset({"u": (("z", "time", "y", "x"),
                                   np.full((16, 2, 16, 16), 3.25))}, coords=coords)
            ds.time.attrs["units"] = "seconds since 2000-01-01"
            ds.to_netcdf(root / "fluid_25.nc")
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump({
                "root": str(root), "file_glob": "*.nc", "channels": ["u"],
                "var_dims": ["time", "x", "y", "z"], "source_grid": [16]*3,
                "snapshots_per_file": 2, "periodic": True,
                "splits": {"train": [25]},
                "coarsen": {"method": "spectral_sharp", "target_grid": [8]*3,
                            "cache_dir": str(root / "cache")}}))
            config = CoarsenConfig.from_yaml(config_path)
            manifest = run_coarsen_pass(config, device=torch.device("cpu"))
            output = root / "cache" / "fluid_25.npy"
            arr = np.load(output)
            self.assertEqual(arr.shape, (2, 1, 8, 8, 8))
            self.assertEqual(arr.dtype, np.float32)
            np.testing.assert_allclose(arr, 3.25)
            self.assertEqual(manifest["time_coordinates"]["25"], [0., 0.5])
            self.assertEqual(manifest["time_attributes"]["25"]["units"],
                             "seconds since 2000-01-01")
            modified = output.stat().st_mtime_ns
            self.assertEqual(run_coarsen_pass(config), manifest)
            self.assertEqual(output.stat().st_mtime_ns, modified)

    def test_fourier_modes_and_nesting(self):
        try:
            import torch
            from coarsening.coarsen import spectral_coarsen, _verification_residuals
        except ModuleNotFoundError as error:
            if error.name not in {"torch", "xarray", "jaxtyping"}:
                raise
            self.skipTest(f"Missing test dependency: {error.name}")
        n, nc = 32, 8
        x = torch.arange(n, dtype=torch.float64) * (2 * torch.pi / n)
        field = (2 + torch.cos(2 * x) + torch.cos(7 * x))[None, :, None, None]
        field = field.expand(1, n, n, n)
        coarse = spectral_coarsen(field, nc)
        xc = torch.arange(nc, dtype=torch.float64) * (2 * torch.pi / nc)
        expected = (2 + torch.cos(2 * xc))[None, :, None, None].expand_as(coarse)
        torch.testing.assert_close(coarse, expected, atol=1e-12, rtol=1e-12)
        residuals = _verification_residuals(field, nc)
        self.assertLess(residuals["mean_abs_diff"], 1e-12)
        self.assertLess(residuals["idempotency_max_abs_err"], 1e-12)
        self.assertFalse(residuals["energy_spectrum_checked"])


if __name__ == "__main__":
    unittest.main()
