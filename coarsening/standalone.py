"""NetCDF discovery, source fingerprints, and dataset schema checks.

Configuration is loaded and validated by config.py; these helpers check actual
files against that configuration without defining experiment settings.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import logging
import re

import numpy as np


def discover_files(root, file_glob, file_id_pattern):
    """Match the configured regex against the entire filename without extension."""
    files = {}
    for path in sorted(Path(root).glob(file_glob)):
        if not path.is_file():
            continue
        match = re.fullmatch(file_id_pattern, path.stem)
        if match is None:
            raise ValueError(f"Filename does not match file_id_pattern: {path.name}")
        file_id = int(match.group(1))
        if file_id in files:
            raise ValueError(f"Duplicate file ID {file_id}: {files[file_id]} and {path}")
        files[file_id] = path
    return files


def source_signature(paths):
    """Metadata fingerprint, not a checksum of file contents."""
    records = []
    for path in paths:
        stat = path.stat()
        records.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
    digest = hashlib.sha256(json.dumps(records).encode()).hexdigest()
    logging.getLogger(__name__).info("Source metadata SHA256: %s", digest)
    return digest


def assert_schema(ds, path, config):
    expected = dict(zip(config.var_dims,
                        (config.snapshots_per_file, *config.source_grid)))
    for dim, size in expected.items():
        if ds.sizes.get(dim) != size:
            raise ValueError(f"{path}: expected {dim} size {size}, got {ds.sizes.get(dim)}")
    for channel in config.channels:
        if channel not in ds or set(ds[channel].dims) != set(config.var_dims):
            raise ValueError(f"{path}: {channel} must have dimensions {config.var_dims}")
        if ds[channel].dtype.kind not in "fiu":
            raise ValueError(f"{path}: {channel} must contain real numeric data")
    for dim in config.var_dims:
        if dim not in ds.coords or ds[dim].dims != (dim,):
            raise ValueError(f"{path}: expected one-dimensional coordinate {dim}")
        values = np.asarray(ds[dim].values, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"{path}: nonfinite coordinate {dim}")
        delta = np.diff(values)
        if dim == config.time_dim:
            if not np.all(delta > 0):
                raise ValueError(f"{path}: time must increase strictly")
        elif (not (np.all(delta > 0) or np.all(delta < 0))
              or not np.allclose(delta, delta[0], rtol=1e-4, atol=0)):
            raise ValueError(f"{path}: {dim} must be a uniform monotonic grid")
