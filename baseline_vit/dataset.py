"""Timestamp-checked within-file pairs and training-only normalization."""

from functools import lru_cache
from pathlib import Path
import hashlib
import json

import numpy as np
import torch
from torch.utils.data import Dataset


class CacheIndex:
    def __init__(self, cfg):
        data = cfg["data"]
        self.root = Path(data["cache_dir"])
        manifest_bytes = (self.root / "manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        if manifest["channels"] != data["channels"]:
            raise ValueError("Cache channel names/order differ from the forecasting configuration")
        grid = cfg["model"]["grid_size"]
        if manifest["target_grid"] != [grid] * 3:
            raise ValueError("Cache grid differs from the model grid")
        if "var_dims" not in manifest or len(manifest["var_dims"]) != 4:
            raise ValueError("Manifest must record the time and three spatial dimension names")
        self.times = {}
        self.groups = {}
        signatures = []
        time_attrs = []
        for split in ("train", "val", "test", "gaps"):
            for file_id in data[split]:
                if file_id not in manifest["file_ids"]:
                    raise ValueError(f"File {file_id} is absent from the completed cache manifest")
                times = np.asarray(manifest["time_coordinates"][str(file_id)], dtype=np.float64)
                if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
                        or not np.all(np.diff(times) > 0)):
                    raise ValueError(f"File {file_id}: need at least two strictly increasing finite times")
                self.times[file_id] = times
                time_attrs.append(manifest.get("time_attributes", {}).get(str(file_id), {}))
                path = self.path(file_id)
                arr = np.load(path, mmap_mode="r", allow_pickle=False)
                expected = (len(times), len(data["channels"]), grid, grid, grid)
                if arr.shape != expected or arr.dtype != np.float32:
                    raise ValueError(f"{path}: expected float32 {expected}, got {arr.dtype} {arr.shape}")
                del arr
                stat = path.stat()
                signatures.append((file_id, stat.st_size, stat.st_mtime_ns))
            self.groups[split] = sorted(data[split], key=lambda i: self.times[i][0])
        if any(attrs != time_attrs[0] for attrs in time_attrs):
            raise ValueError("Time units/calendars must match across selected files")
        # Consecutive files may repeat the boundary timestamp. Pairs stay within
        # files, so touching endpoints introduce no zero-duration example.
        # Split boundaries below remain strictly separated to prevent leakage.
        chronological = sorted(self.times, key=lambda i: self.times[i][0])
        for left, right in zip(chronological, chronological[1:]):
            if self.times[left][-1] > self.times[right][0]:
                raise ValueError(f"Files {left} and {right} overlap in time; resolve trajectory membership first")
        for left, right in (("train", "val"), ("val", "test")):
            if (max(self.times[i][-1] for i in self.groups[left])
                    >= min(self.times[i][0] for i in self.groups[right])):
                raise ValueError(f"Expected chronological {left} before {right}")
        train_steps = np.concatenate([np.diff(self.times[i]) for i in self.groups["train"]])
        self.dt_reference = float(np.median(train_steps))
        for file_id, times in self.times.items():
            if not np.allclose(np.diff(times), self.dt_reference,
                               rtol=data["dt_rtol"], atol=data["dt_atol"]):
                raise ValueError(f"File {file_id}: variable saved-time spacing; this baseline assumes a fixed interval")
        self.manifest = manifest
        self.fingerprint = hashlib.sha256(
            manifest_bytes + json.dumps(sorted(signatures)).encode()
        ).hexdigest()

    def path(self, file_id):
        return self.root / f"fluid_{file_id}.npy"


def training_statistics(index):
    """Stable per-channel population moments, using ONLY training snapshots."""
    count = 0
    mean = None
    m2 = None
    for file_id in index.groups["train"]:
        arr = np.load(index.path(file_id), mmap_mode="r", allow_pickle=False)
        for snapshot in arr:
            values = np.asarray(snapshot, dtype=np.float64).reshape(snapshot.shape[0], -1)
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite training data in file {file_id}")
            n = values.shape[1]
            local_mean = values.mean(axis=1)
            local_m2 = values.var(axis=1) * n
            if count == 0:
                mean, m2 = local_mean, local_m2
            else:
                delta = local_mean - mean
                total = count + n
                m2 += local_m2 + delta ** 2 * count * n / total
                mean += delta * n / total
            count += n
    std = np.sqrt(np.maximum(m2 / count, 0))
    std = np.where(std < 1e-12, 1.0, std)
    return {"mean": mean.tolist(), "std": std.tolist(), "count_per_channel": count,
            "file_ids": index.groups["train"]}


class PairDataset(Dataset):
    def __init__(self, index, split, max_pairs=None):
        self.index = index
        self.pairs = [(i, t) for i in index.groups[split]
                      for t in range(len(index.times[i]) - 1)]
        if max_pairs is not None:
            self.pairs = self.pairs[:max_pairs]

    def __len__(self):
        return len(self.pairs)

    @lru_cache(maxsize=2)
    def _array(self, file_id):
        return np.load(self.index.path(file_id), mmap_mode="r", allow_pickle=False)

    def __getitem__(self, item):
        file_id, t = self.pairs[item]
        arr = self._array(file_id)
        # Copies avoid exposing read-only memory maps to PyTorch operations.
        x, y = np.array(arr[t], copy=True), np.array(arr[t + 1], copy=True)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"Nonfinite data in file {file_id}, pair {t}")
        times = self.index.times[file_id]
        dt = np.float32((times[t + 1] - times[t]) / self.index.dt_reference)
        return torch.from_numpy(x), torch.from_numpy(y), torch.tensor(dt)
