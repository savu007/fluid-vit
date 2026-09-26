"""Use the existing cache contract; add within-file temporal windows only."""

import numpy as np
import torch

from baseline_vit.dataset import CacheIndex, PairDataset, training_statistics


class SequenceDataset(PairDataset):
    def __init__(self, index, split, horizon=1, max_pairs=None):
        if type(horizon) is not int or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        super().__init__(index, split)
        self.horizon = horizon
        for file_id in index.groups[split]:
            if len(index.times[file_id]) <= horizon:
                raise ValueError(f"File {file_id} has too few snapshots for horizon {horizon}")
        self.pairs = [(i, t) for i in index.groups[split]
                      for t in range(len(index.times[i]) - horizon)]
        if max_pairs is not None:
            if type(max_pairs) is not int or max_pairs < 1:
                raise ValueError("max_pairs must be positive")
            self.pairs = self.pairs[:max_pairs]
        if not self.pairs:
            raise ValueError("Selected split contains no temporal windows")

    def __getitem__(self, item):
        file_id, start = self.pairs[item]
        arr = self._array(file_id)
        x = np.array(arr[start], copy=True)
        targets = np.array(arr[start + 1:start + self.horizon + 1], copy=True)
        if not np.isfinite(x).all() or not np.isfinite(targets).all():
            raise ValueError(f"Nonfinite sequence in file {file_id}, start {start}")
        times = self.index.times[file_id][start:start + self.horizon + 1]
        dt = (np.diff(times) / self.index.dt_reference).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(targets), torch.from_numpy(dt)
