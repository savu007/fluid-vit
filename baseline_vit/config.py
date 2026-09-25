"""Load experiment settings without accessing the dataset or running computation."""

from pathlib import Path
import math

import yaml


def load_config(path):
    path = Path(path).resolve()
    with path.open() as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a mapping")
    required = {
        "data": {"cache_dir", "channels", "train", "val", "test", "gaps", "dt_rtol", "dt_atol"},
        "model": {"grid_size", "in_channels", "patch_size", "embed_dim", "depth",
                  "num_heads", "mlp_ratio", "dropout"},
        "training": {"output_dir", "device", "epochs", "batch_size", "num_workers",
                     "learning_rate", "weight_decay", "grad_clip", "max_train_pairs"},
    }
    if set(cfg) != {"seed", *required}:
        raise ValueError("Top-level keys must be seed, data, model, training")
    for section, keys in required.items():
        if not isinstance(cfg[section], dict) or set(cfg[section]) != keys:
            raise ValueError(f"{section} must contain exactly these keys: {sorted(keys)}")
    if type(cfg["seed"]) is not int or cfg["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    data, model, training = (cfg[k] for k in ("data", "model", "training"))
    seen = set()
    for split in ("train", "val", "test", "gaps"):
        ids = data[split]
        if not isinstance(ids, list) or any(type(i) is not int for i in ids):
            raise ValueError(f"data.{split} must be a list of integer file IDs")
        if split != "gaps" and not ids:
            raise ValueError(f"data.{split} must not be empty")
        if len(ids) != len(set(ids)) or seen.intersection(ids):
            raise ValueError("All split and gap file IDs must be unique and disjoint")
        seen.update(ids)
    channels = data["channels"]
    if (not isinstance(channels, list) or not channels
            or any(not isinstance(c, str) or not c for c in channels)
            or len(set(channels)) != len(channels)):
        raise ValueError("data.channels must contain unique variable names")
    for key in ("grid_size", "in_channels", "patch_size", "embed_dim", "depth", "num_heads", "mlp_ratio"):
        if type(model[key]) is not int or model[key] < 1:
            raise ValueError(f"model.{key} must be a positive integer")
    if model["in_channels"] != len(channels):
        raise ValueError("model.in_channels must equal the number of configured channels")
    if model["grid_size"] % model["patch_size"]:
        raise ValueError("grid_size must be divisible by patch_size")
    if model["embed_dim"] % model["num_heads"]:
        raise ValueError("embed_dim must be divisible by num_heads")
    for mapping, key, lower, strict in (
        (data, "dt_rtol", 0, False), (data, "dt_atol", 0, False),
        (model, "dropout", 0, False), (training, "learning_rate", 0, True),
        (training, "weight_decay", 0, False), (training, "grad_clip", 0, True),
    ):
        value = mapping[key]
        if (type(value) not in (int, float) or not math.isfinite(value)
                or value < lower or (strict and value == lower)):
            raise ValueError(f"Invalid numerical setting: {key}")
    if model["dropout"] >= 1:
        raise ValueError("dropout must be < 1")
    for key in ("epochs", "batch_size", "num_workers"):
        if type(training[key]) is not int or training[key] < (0 if key == "num_workers" else 1):
            raise ValueError(f"Invalid training.{key}")
    limit = training["max_train_pairs"]
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("max_train_pairs must be null or a positive integer")
    if training["device"] not in ("cpu", "cuda"):
        raise ValueError("training.device must be cpu or cuda")
    for mapping, key in ((data, "cache_dir"), (training, "output_dir")):
        if not isinstance(mapping[key], str) or not mapping[key]:
            raise ValueError(f"{key} must be a nonempty path")
        value = Path(mapping[key]).expanduser()
        mapping[key] = str(value if value.is_absolute() else (path.parent / value).resolve())
    return cfg
