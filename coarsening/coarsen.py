"""One-time spectral coarsening pass: 384^3 -> `data.coarsen.target_grid` (64^3).

    PYTHONPATH=src conda run -n mae --no-capture-output --live-stream \\
        python -m mae.data.coarsen --config configs/march-3d-c64-2step-p4.yml

Reads every snapshot declared anywhere in `data.splits` (train, both gaps, val,
test — the full 20 files / 200 snapshots, since the cache is shared across
splits and across the p2/p4 arms) and writes one `fluid_<id>.npy` per file,
shape `(snapshots_per_file, C, *target_grid)` float32, plus a `manifest.json`
recording the source file ids, each file's `time` coordinate, the coarsen
method/target grid, and the operator's verification residuals (computed once,
on the first snapshot processed — see `_verification_residuals`). Skips the
whole pass if the cache already exists and its manifest matches the config
(logged, not silent).

`spectral_coarsen` below is the validated operator from the experiment spec —
E_coarse(k) == E_fine(k) for k <= 31 to ~3.6e-16, idempotent under nesting to
~5.5e-16, mean preserved to ~6.2e-18 on the real dataset. It is reproduced
here EXACTLY as specified and must not be "improved": the Nyquist-zeroing and
the `Nc^3 / Nf^3` normalization are both easy to get subtly, silently wrong.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from jaxtyping import Float
from torch import Tensor

from mae.data.dataset import assert_schema, discover_files, log_raw_file_stat_checksum
from mae.data.spectral import energy_spectrum
from mae.data.trajectory_config import TrajectoryDataConfig

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "manifest.json"


def spectral_coarsen(x: torch.Tensor, Nc: int) -> torch.Tensor:
    """Sharp Fourier low-pass to |k_i| <= Nc/2-1, then exact subsample to Nc^3.

    Reproduced verbatim from the `march-3d-c64-2step-p4` spec — do not modify.
    Run in float64 (the caller promotes before calling this); the caller
    downcasts the *stored* result to float32 after this returns.
    """
    Nf = x.shape[-1]
    h = Nc // 2
    F = torch.fft.rfftn(x, dim=(-3, -2, -1))
    Fc = torch.zeros(x.shape[:-3] + (Nc, Nc, h + 1), dtype=F.dtype, device=x.device)
    lo, hi = slice(0, h), slice(-(h - 1), None)  # k in [0, h-1] and [-(h-1), -1]
    for a in (lo, hi):
        for b in (lo, hi):
            Fc[..., a, b, :h] = F[..., a, b, :h]
    return torch.fft.irfftn(Fc, s=(Nc,) * 3, dim=(-3, -2, -1)) * (Nc**3 / Nf**3)


def cache_path(cache_dir: Path, file_id: int) -> Path:
    """Path of the cached `.npy` for `file_id` under `cache_dir`."""
    return Path(cache_dir) / f"fluid_{file_id}.npy"


def _manifest_path(cache_dir: Path) -> Path:
    return Path(cache_dir) / MANIFEST_FILENAME


def load_manifest(cache_dir: Path) -> dict | None:
    """Load `manifest.json` from `cache_dir`, or `None` if it does not exist."""
    path = _manifest_path(cache_dir)
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _all_declared_file_ids(config: TrajectoryDataConfig) -> tuple[int, ...]:
    ids: set[int] = set()
    for group_ids in config.splits.all_groups().values():
        ids.update(group_ids)
    return tuple(sorted(ids))


def _manifest_matches(
    manifest: dict, config: TrajectoryDataConfig, file_ids: tuple[int, ...]
) -> bool:
    """Whether an existing manifest was produced by this exact config's coarsen block."""
    return (
        manifest.get("file_ids") == list(file_ids)
        and manifest.get("file_glob") == config.file_glob
        and manifest.get("source_grid") == list(config.source_grid)
        and manifest.get("target_grid") == list(config.coarsen.target_grid)
        and manifest.get("coarsen_method") == config.coarsen.method
        and manifest.get("channels") == list(config.channels)
    )


def _cache_complete(cache_dir: Path, file_ids: tuple[int, ...]) -> bool:
    return all(cache_path(cache_dir, fid).exists() for fid in file_ids)


def _verification_residuals(x_fine64: Float[Tensor, "C Nf Nf Nf"], nc: int) -> dict:
    """The three exact identities from the spec, computed on ONE fp64 snapshot.

    Never sees the float32-downcast cache — every quantity here is derived
    straight from `x_fine64` in float64, matching CLAUDE.md §Precision
    ("physics diagnostics reported from reduced-precision outputs are invalid").
    """
    coarse64 = spectral_coarsen(x_fine64, nc)

    k_max = nc // 2 - 1
    e_fine = energy_spectrum(x_fine64)
    e_coarse = energy_spectrum(coarse64)
    denom = e_fine[: k_max + 1].clamp_min(torch.finfo(torch.float64).tiny)
    energy_max_rel_err = float(
        ((e_coarse[: k_max + 1] - e_fine[: k_max + 1]).abs() / denom).max()
    )

    nf = x_fine64.shape[-1]
    mid = nc * 2
    if mid < nf:
        nested = spectral_coarsen(spectral_coarsen(x_fine64, mid), nc)
        idempotency_max_abs_err = float((coarse64 - nested).abs().max())
    else:
        idempotency_max_abs_err = (
            None  # not checkable: no room for an intermediate grid
        )

    mean_abs_diff = float((x_fine64.mean() - coarse64.mean()).abs())

    return {
        "k_max_checked": k_max,
        "energy_spectrum_max_rel_err": energy_max_rel_err,
        "idempotency_intermediate_grid": mid if mid < nf else None,
        "idempotency_max_abs_err": idempotency_max_abs_err,
        "mean_abs_diff": mean_abs_diff,
        "output_finite": bool(torch.isfinite(coarse64).all()),
    }


def run_coarsen_pass(
    config: TrajectoryDataConfig,
    *,
    device: torch.device | None = None,
    force: bool = False,
) -> dict:
    """Run (or skip) the one-time coarsening pass declared by `config.coarsen`.

    Returns the manifest dict (freshly written, or the pre-existing one if the
    pass was skipped). Raises `RuntimeError` if a file id declared anywhere in
    `config.splits` is missing on disk.
    """
    cache_dir = Path(config.coarsen.cache_dir)
    file_ids = _all_declared_file_ids(config)

    existing = load_manifest(cache_dir)
    if (
        not force
        and existing is not None
        and _manifest_matches(existing, config, file_ids)
        and _cache_complete(cache_dir, file_ids)
    ):
        logger.info(
            "coarsen cache up to date at %s (file_ids=%s) — skipping the pass",
            cache_dir,
            file_ids,
        )
        return existing

    if not config.coarsen.enabled:
        raise RuntimeError(
            "data.coarsen.enabled=false but no valid cache exists at "
            f"{cache_dir} — nothing to run and nothing to read"
        )

    file_paths = discover_files(config.root, config.file_glob)
    missing = [i for i in file_ids if i not in file_paths]
    if missing:
        raise RuntimeError(
            f"declared file ids not found under {config.root}: {missing}"
        )
    log_raw_file_stat_checksum([file_paths[i] for i in file_ids])

    if device is None:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
    logger.info(
        "coarsen pass: device=%s target_grid=%s", device, config.coarsen.target_grid
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    nc = config.coarsen.target_grid[0]
    time_coordinates: dict[str, list[float]] = {}
    residuals: dict | None = None

    for file_id in file_ids:
        path = file_paths[file_id]
        ds = xr.open_dataset(path)
        try:
            assert_schema(ds, path, config)
            time_coordinates[str(file_id)] = [
                float(v) for v in np.asarray(ds[config.time_dim].values).reshape(-1)
            ]

            coarse_snapshots = []
            for t in range(config.snapshots_per_file):
                raw = np.stack(
                    [
                        ds[ch].isel({config.time_dim: t}).values
                        for ch in config.channels
                    ],
                    axis=0,
                ).astype(np.float32)
                x64 = torch.from_numpy(raw).to(device=device, dtype=torch.float64)

                if residuals is None:
                    residuals = _verification_residuals(x64, nc)
                    residuals["computed_on_file_id"] = file_id
                    residuals["computed_on_time_index"] = t

                xc64 = spectral_coarsen(x64, nc)
                coarse_snapshots.append(xc64.to(torch.float32).cpu().numpy())

            arr = np.stack(coarse_snapshots, axis=0)  # (snapshots_per_file, C, *target)
            np.save(cache_path(cache_dir, file_id), arr)
            logger.info("coarsen pass: wrote file_id=%d shape=%s", file_id, arr.shape)
        finally:
            ds.close()

    manifest = {
        "source_root": str(config.root),
        "file_glob": config.file_glob,
        "file_ids": list(file_ids),
        "time_coordinates": time_coordinates,
        "coarsen_method": config.coarsen.method,
        "source_grid": list(config.source_grid),
        "target_grid": list(config.coarsen.target_grid),
        "channels": list(config.channels),
        "verification_residuals": residuals,
    }
    with _manifest_path(cache_dir).open("w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(
        "coarsen pass: wrote manifest to %s: %s", _manifest_path(cache_dir), manifest
    )
    return manifest


def main(argv: list[str] | None = None) -> None:
    """CLI: run (or skip) the coarsening pass for one experiment config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-run even if a matching cache already exists",
    )
    args = parser.parse_args(argv)

    config = TrajectoryDataConfig.from_yaml(args.config)
    run_coarsen_pass(config, force=args.force)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()