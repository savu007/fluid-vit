"""Observed histories and future targets from the existing velocity cache."""
import numpy as np
import torch
from baseline_vit.dataset import PairDataset
from forecasting.dataset import CrossFileSequenceDataset


class HistoryDataset(PairDataset):
    """Four observed frames plus five future frames; no split-crossing windows.

    Training remains within files, matching the baseline's selection policy.
    Validation/test may use checked, deduplicated cross-file sequences.
    Times are returned for every frame, not replaced by a nominal constant dt.
    """
    def __init__(self, index, split, *, history=4, horizon=5, cross_file=False,
                 dt_rtol=0.003, dt_atol=1e-10, max_windows=None):
        if type(history) is not int or history != 4:
            raise ValueError('This experiment requires four observed frames')
        if type(horizon) is not int or horizon != 5:
            raise ValueError('This experiment requires five future frames')
        if split not in ('train', 'val', 'test'):
            raise ValueError('Unknown split')
        if max_windows is not None and (type(max_windows) is not int or max_windows < 1):
            raise ValueError('max_windows must be a positive integer or None')
        super().__init__(index, split)
        self.history, self.horizon = history, horizon
        self.continuous = None
        span = history + horizon - 1
        if cross_file:
            self.continuous = CrossFileSequenceDataset(index, split, span,
                                                       dt_rtol=dt_rtol, dt_atol=dt_atol)
            self.pairs = self.continuous.pairs.copy()
        else:
            self.pairs = [(file_id, start) for file_id in index.groups[split]
                          for start in range(len(index.times[file_id]) - span)]
        if max_windows is not None:
            self.pairs = self.pairs[:max_windows]
        if not self.pairs:
            raise ValueError('No complete four-history/five-future windows in selected split')

    def __getitem__(self, item):
        if self.continuous is None:
            file_id, start = self.pairs[item]
            end = start + self.history + self.horizon
            fields = np.array(self._array(file_id)[start:end], copy=True)
            times = self.index.times[file_id][start:end].copy()
        else:
            start = self.pairs[item]
            refs = self.continuous.frames[start:start + self.history + self.horizon]
            fields = np.stack([self._array(file_id)[frame] for file_id, frame in refs])
            times = self.continuous.times[start:start + self.history + self.horizon].copy()
        if not np.isfinite(fields).all() or not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
            raise ValueError('Nonfinite fields or non-increasing sequence times')
        return (torch.from_numpy(fields[:self.history]),
                torch.from_numpy(fields[self.history:]), torch.from_numpy(times))
