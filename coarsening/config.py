"""Configuration schema, YAML loading, and validation for spectral coarsening.

Edit config.yaml to change dataset paths or experiment settings. This module
converts those values into typed objects and checks the operator's requirements.
Both top-level settings and a wrapping `data:` mapping are supported. Relative
paths retain the original convention: resolve against the working directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import re

import yaml

Split = Literal["train", "val", "test"]


@dataclass(frozen=True)
class SplitsConfig:
    """File-id membership per split, plus the two gap files.

    Splitting is by FILE: every snapshot in a file moves with it, so
    file-level disjointness (checked when loading configuration) implies
    snapshot-level disjointness too. `gap_after_train` and `gap_after_val`
    file ids belong to no split at all — they exist only to widen the
    train/val and val/test boundaries.
    """

    train: tuple[int, ...]
    val: tuple[int, ...]
    test: tuple[int, ...]
    gap_after_train: tuple[int, ...] = ()
    gap_after_val: tuple[int, ...] = ()

    def ids_for(self, split: Split) -> tuple[int, ...]:
        """Return the declared file ids for `split` ("train" | "val" | "test")."""
        if split not in ("train", "val", "test"):
            raise ValueError(
                f"unknown split {split!r}, expected 'train', 'val', or 'test'"
            )
        return getattr(self, split)

    def all_groups(self) -> dict[str, tuple[int, ...]]:
        """All five declared groups, including the two gaps, by name."""
        return {
            "train": self.train,
            "gap_after_train": self.gap_after_train,
            "val": self.val,
            "gap_after_val": self.gap_after_val,
            "test": self.test,
        }


def assert_splits_disjoint(splits: SplitsConfig) -> None:
    """Raise `ValueError` if any file id appears in more than one group.

    Checked against all five declared groups (train, both gaps, val, test) —
    not just train/val/test — so a typo that puts a gap file back into a real
    split is caught the same way a train/val overlap would be.
    """
    seen: dict[int, str] = {}
    for name, ids in splits.all_groups().items():
        for file_id in ids:
            if file_id in seen:
                raise ValueError(
                    f"file id {file_id} appears in both '{seen[file_id]}' and '{name}' — "
                    "splits (including the gap groups) must be disjoint"
                )
            seen[file_id] = name


@dataclass(frozen=True)
class CoarsenSettings:
    enabled: bool
    method: str
    target_grid: tuple[int, int, int]
    cache_dir: Path


@dataclass(frozen=True)
class CoarsenConfig:
    root: Path
    file_glob: str
    channels: tuple[str, ...]
    var_dims: tuple[str, ...]
    source_grid: tuple[int, int, int]
    snapshots_per_file: int
    splits: SplitsConfig
    coarsen: CoarsenSettings
    file_id_pattern: str = r"fluid_(\d+)"

    @property
    def time_dim(self):
        return self.var_dims[0]

    @property
    def spatial_dims(self):
        return self.var_dims[1:]

    @classmethod
    def from_yaml(cls, path):
        with Path(path).open() as stream:
            raw = yaml.safe_load(stream)
        if not isinstance(raw, dict):
            raise ValueError("Configuration must be a YAML mapping")
        data = raw.get("data", raw)
        if not isinstance(data, dict):
            raise ValueError("data must be a mapping")
        required = ("root", "file_glob", "channels", "var_dims", "source_grid",
                    "snapshots_per_file", "splits", "coarsen", "periodic")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"Missing coarsening configuration keys: {', '.join(missing)}")
        if data["periodic"] is not True:
            raise ValueError("Spectral coarsening requires periodic: true")
        block = data["coarsen"]
        if not isinstance(block, dict):
            raise ValueError("coarsen must be a mapping")
        for key in ("target_grid", "cache_dir", "method"):
            if key not in block:
                raise ValueError(f"Missing coarsen.{key}")
        if block["method"] != "spectral_sharp":
            raise ValueError("Only coarsen.method: spectral_sharp is supported")
        enabled = block.get("enabled", True)
        if type(enabled) is not bool:
            raise ValueError("coarsen.enabled must be a boolean")

        def grid(value, name):
            if (not isinstance(value, (list, tuple)) or len(value) != 3
                    or any(type(n) is not int or n < 4 for n in value)
                    or len(set(value)) != 1):
                raise ValueError(f"{name} must contain three equal integers >= 4")
            return tuple(value)

        source = grid(data["source_grid"], "source_grid")
        target = grid(block["target_grid"], "coarsen.target_grid")
        if target[0] % 2 or target[0] > source[0]:
            raise ValueError("Target grid must be even and no larger than the source")
        channels, dims = data["channels"], data["var_dims"]
        for values, name in ((channels, "channels"), (dims, "var_dims")):
            if (not isinstance(values, list) or not values
                    or any(not isinstance(v, str) or not v for v in values)
                    or len(set(values)) != len(values)):
                raise ValueError(f"{name} must be a list of unique nonempty names")
        if len(dims) != 4:
            raise ValueError("var_dims must name time followed by three spatial axes")
        count = data["snapshots_per_file"]
        if type(count) is not int or count < 1:
            raise ValueError("snapshots_per_file must be a positive integer")
        groups = data["splits"]
        if not isinstance(groups, dict):
            raise ValueError("splits must be a mapping")
        names = ("train", "val", "test", "gap_after_train", "gap_after_val")
        if set(groups) - set(names):
            raise ValueError("Unknown split names in configuration")
        for name, ids in groups.items():
            if not isinstance(ids, list) or any(type(i) is not int for i in ids):
                raise ValueError(f"splits.{name} must be a list of integer file IDs")
        splits = SplitsConfig(**{name: tuple(groups.get(name, ())) for name in names})
        assert_splits_disjoint(splits)
        if not any(splits.all_groups().values()):
            raise ValueError("At least one file ID must be declared in splits")
        pattern = data.get("file_id_pattern", r"fluid_(\d+)")
        if re.compile(pattern).groups != 1:
            raise ValueError("file_id_pattern must have exactly one capture group for the ID")
        return cls(Path(data["root"]), data["file_glob"], tuple(channels), tuple(dims),
                   source, count, splits,
                   CoarsenSettings(enabled, block["method"], target,
                                   Path(block["cache_dir"])), pattern)
