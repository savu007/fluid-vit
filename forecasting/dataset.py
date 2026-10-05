"""Within-file training windows and strictly checked cross-file evaluation windows."""

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


class CrossFileSequenceDataset(PairDataset):
    """Evaluation only: join a chronological split, checking every boundary.

    Repeated boundary times are deduplicated only if the fields are identical.
    Missing intervals are rejected rather than interpreted as one model step.
    """
    def __init__(self, index, split, horizon=100, *, dt_rtol=0.003, dt_atol=1e-10):
        if split not in ("val", "test"):
            raise ValueError("Cross-file windows are evaluation-only")
        if type(horizon) is not int or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        super().__init__(index, split)
        self.horizon = horizon
        self.frames, times = [], []
        for file_id in index.groups[split]:
            for frame, timestamp in enumerate(index.times[file_id]):
                if times and timestamp == times[-1]:
                    left_id, left_frame = self.frames[-1]
                    left, right = self._array(left_id)[left_frame], self._array(file_id)[frame]
                    if not np.isfinite(left).all() or not np.array_equal(left, right):
                        raise ValueError(f"Conflicting repeated timestamp at files {left_id}, {file_id}")
                    continue
                if times and not np.isclose(timestamp - times[-1], index.dt_reference,
                                             rtol=dt_rtol, atol=dt_atol):
                    raise ValueError(f"Noncontiguous evaluation time before file {file_id}, frame {frame}")
                self.frames.append((file_id, frame))
                times.append(float(timestamp))
        self.times = np.asarray(times, dtype=np.float64)
        self.pairs = list(range(len(self.frames) - horizon))
        if not self.pairs:
            raise ValueError("Too few contiguous snapshots for evaluation horizon")

    def __getitem__(self, item):
        start = self.pairs[item]
        refs = self.frames[start:start + self.horizon + 1]
        x = np.array(self._array(refs[0][0])[refs[0][1]], copy=True)
        targets = np.stack([self._array(file_id)[frame] for file_id, frame in refs[1:]])
        if not np.isfinite(x).all() or not np.isfinite(targets).all():
            raise ValueError(f"Nonfinite evaluation sequence starting at {refs[0]}")
        intervals = np.diff(self.times[start:start + self.horizon + 1]) / self.index.dt_reference
        return torch.from_numpy(x), torch.from_numpy(targets), torch.from_numpy(intervals.astype(np.float32))

    def starts(self):
        return [{"file_id": self.frames[i][0], "frame": self.frames[i][1],
                 "time": float(self.times[i]), "end_time": float(self.times[i + self.horizon])}
                for i in self.pairs]
