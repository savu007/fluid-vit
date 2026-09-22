"""Typed parsing of the `data:` block of an experiment config.

This module owns nothing about how bytes are read off disk — see
`mae.data.dataset` and `mae.data.normalization` for that. It exists so that
every other module reads config values through one validated path instead of
indexing raw dicts, and so a malformed config fails at load time with a
message naming the missing key, not three layers deep inside a `KeyError`
during training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

Split = Literal["train", "val", "test"]


@dataclass(frozen=True)
class NormalizationConfig:
    """Per-channel normalization settings, computed on the train split only.

    Attributes:
        mode: normalization strategy. Only "per_channel_global" is implemented
            by `mae.data.normalization`.
        stats_path: where frozen mean/std are read from (by the DataModule)
            and written to (by the explicit stats-generation entry point).
        recompute: must be `false` in every experiment config. `true` is
            rejected at load time — statistics are frozen and regenerated
            only via `python -m mae.data.normalization`, never implicitly by
            a training run. See CLAUDE.md §Required properties.
    """

    mode: str
    stats_path: Path
    recompute: bool


@dataclass(frozen=True)
class LoaderConfig:
    """`torch.utils.data.DataLoader` knobs, sourced from `data.loader`."""

    batch_size: int
    num_workers: int
    pin_memory: bool
    persistent_workers: bool


@dataclass(frozen=True)
class SplitsConfig:
    """File-id membership per split, plus the two gap files.

    Splitting is by FILE: every snapshot in a file moves with it, so
    file-level disjointness (checked in `DataConfig.__post_init__`) implies
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
class DataConfig:
    """Parsed `data:` block of an experiment config, plus the top-level seed.

    Axis order on disk (asserted at open in `mae.data.dataset`, never
    assumed): `var_dims[0]` is the per-snapshot ("time") axis; `var_dims[1:]`
    are the three spatial axes, in the same order as `grid_size` and
    `crop_size`. For `mae-3d-base`, `var_dims = [time, x, y, z]`, so a raw
    on-disk variable is `(time, x, y, z)` and a returned crop is
    `(C, X, Y, Z)` — channel first (in `channels` order), then the spatial
    axes in that same source order. There is no axis reversal anywhere in
    this data path.
    """

    root: Path
    file_glob: str
    channels: tuple[str, ...]
    var_dims: tuple[str, ...]
    snapshots_per_file: int
    total_snapshots: int
    grid_size: tuple[int, int, int]
    domain_length: float
    grid_centering: str
    periodic: bool
    crop_size: tuple[int, int, int]
    crops_per_snapshot: int
    splits: SplitsConfig
    normalization: NormalizationConfig
    loader: LoaderConfig
    seed: int

    def __post_init__(self) -> None:
        if len(self.var_dims) != 4:
            raise ValueError(
                f"data.var_dims must have exactly 4 entries, got {self.var_dims}"
            )
        if len(self.grid_size) != 3:
            raise ValueError(
                f"data.grid_size must have exactly 3 entries, got {self.grid_size}"
            )
        if len(self.crop_size) != 3:
            raise ValueError(
                f"data.crop_size must have exactly 3 entries, got {self.crop_size}"
            )
        if any(c > g for c, g in zip(self.crop_size, self.grid_size, strict=True)):
            raise ValueError(
                f"data.crop_size {self.crop_size} exceeds data.grid_size {self.grid_size}"
            )
        if not self.channels:
            raise ValueError("data.channels must not be empty")
        if self.normalization.recompute:
            raise ValueError(
                "data.normalization.recompute=true is not supported here. Statistics are frozen; "
                "regenerate explicitly with `python -m mae.data.normalization --config <path> --force`."
            )
        assert_splits_disjoint(self.splits)

    @property
    def time_dim(self) -> str:
        """Name of the per-snapshot axis (`var_dims[0]`)."""
        return self.var_dims[0]

    @property
    def spatial_dims(self) -> tuple[str, str, str]:
        """Names of the three spatial axes (`var_dims[1:]`), in `grid_size` order."""
        return (self.var_dims[1], self.var_dims[2], self.var_dims[3])

    @classmethod
    def from_yaml(cls, path: str | Path) -> DataConfig:
        """Parse `data:` and top-level `seed` out of an experiment yml.

        Raises `FileNotFoundError` if `path` does not exist, `ValueError` if
        it does not use the `.yml` extension, and `KeyError` naming the exact
        missing key (e.g. `'data.normalization.stats_path'`) if a required
        field is absent. Never silently defaults a physical parameter.
        """
        path = Path(path)
        if path.suffix != ".yml":
            raise ValueError(f"config must use the .yml extension, got: {path}")
        if not path.exists():
            raise FileNotFoundError(f"config not found: {path}")
        with path.open() as f:
            raw = yaml.safe_load(f)

        def get(d: dict, key: str, ctx: str):
            if key not in d:
                full = f"{ctx}.{key}" if ctx else key
                raise KeyError(f"{path}: missing required key '{full}'")
            return d[key]

        seed = get(raw, "seed", "")
        d = get(raw, "data", "")

        splits_raw = get(d, "splits", "data")
        splits = SplitsConfig(
            train=tuple(get(splits_raw, "train", "data.splits")),
            val=tuple(get(splits_raw, "val", "data.splits")),
            test=tuple(get(splits_raw, "test", "data.splits")),
            gap_after_train=tuple(splits_raw.get("gap_after_train", ())),
            gap_after_val=tuple(splits_raw.get("gap_after_val", ())),
        )

        norm_raw = get(d, "normalization", "data")
        normalization = NormalizationConfig(
            mode=get(norm_raw, "mode", "data.normalization"),
            stats_path=Path(get(norm_raw, "stats_path", "data.normalization")),
            recompute=bool(norm_raw.get("recompute", False)),
        )

        loader_raw = get(d, "loader", "data")
        loader = LoaderConfig(
            batch_size=int(get(loader_raw, "batch_size", "data.loader")),
            num_workers=int(get(loader_raw, "num_workers", "data.loader")),
            pin_memory=bool(get(loader_raw, "pin_memory", "data.loader")),
            persistent_workers=bool(
                get(loader_raw, "persistent_workers", "data.loader")
            ),
        )

        return cls(
            root=Path(get(d, "root", "data")),
            file_glob=get(d, "file_glob", "data"),
            channels=tuple(get(d, "channels", "data")),
            var_dims=tuple(get(d, "var_dims", "data")),
            snapshots_per_file=int(get(d, "snapshots_per_file", "data")),
            total_snapshots=int(get(d, "total_snapshots", "data")),
            grid_size=tuple(int(v) for v in get(d, "grid_size", "data")),
            domain_length=float(get(d, "domain_length", "data")),
            grid_centering=get(d, "grid_centering", "data"),
            periodic=bool(get(d, "periodic", "data")),
            crop_size=tuple(int(v) for v in get(d, "crop_size", "data")),
            crops_per_snapshot=int(get(d, "crops_per_snapshot", "data")),
            splits=splits,
            normalization=normalization,
            loader=loader,
            seed=int(seed),
        )