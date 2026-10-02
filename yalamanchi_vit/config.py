"""Strict configuration validation; paths resolve relative to this YAML."""

from pathlib import Path
import math

import yaml

from .model import validate_model


def load_config(path, *, allow_physical_loss=False):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    physical_loss = None
    if allow_physical_loss:
        if not isinstance(cfg, dict) or "loss" not in cfg:
            raise ValueError("Physical experiment requires an explicit loss section")
        physical_loss = cfg.pop("loss")
        from forecasting.physical_loss import validate_loss
        validate_loss(physical_loss)
    required = {
        "data": {"cache_dir", "channels", "train", "val", "test", "gaps", "dt_rtol", "dt_atol"},
        "model": {"grid_size", "in_channels", "patch_size", "embed_dim", "depths", "num_heads",
                  "window_size", "mlp_ratio", "dropout", "boundary"},
        "training": {"output_dir", "device", "epochs", "batch_size", "num_workers", "learning_rate",
                     "weight_decay", "grad_clip", "max_train_pairs", "objective", "horizon"},
    }
    if not isinstance(cfg, dict) or set(cfg) != {"seed", *required}:
        raise ValueError("Top-level keys must be seed, data, model, training")
    for section, keys in required.items():
        if not isinstance(cfg[section], dict) or set(cfg[section]) != keys:
            raise ValueError(f"{section} must contain exactly {sorted(keys)}")
    if type(cfg["seed"]) is not int or cfg["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    data, model, training = cfg["data"], cfg["model"], cfg["training"]
    validate_model(**model)
    if data["channels"] != ["u", "v", "w"] or model["in_channels"] != 3:
        raise ValueError("This velocity baseline requires channels [u, v, w] and in_channels=3")
    seen = set()
    for split in ("train", "val", "test", "gaps"):
        ids = data[split]
        if not isinstance(ids, list) or any(type(i) is not int for i in ids):
            raise ValueError(f"data.{split} must be a list of integer IDs")
        if split != "gaps" and not ids:
            raise ValueError(f"data.{split} must be nonempty")
        if len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError("Split and gap IDs must be unique and disjoint")
        seen.update(ids)
    for mapping, key, positive in (
        (data, "dt_rtol", False), (data, "dt_atol", False),
        (training, "learning_rate", True), (training, "weight_decay", False),
        (training, "grad_clip", True),
    ):
        value = mapping[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (positive and value == 0):
            raise ValueError(f"Invalid {key}")
    for key in ("epochs", "batch_size", "num_workers", "horizon"):
        value = training[key]
        if type(value) is not int or value < (0 if key == "num_workers" else 1):
            raise ValueError(f"Invalid training.{key}")
    limit = training["max_train_pairs"]
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("max_train_pairs must be null or a positive integer")
    if training["device"] not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    from forecasting.config import validate_objective
    validate_objective(training)
    for mapping, key in ((data, "cache_dir"), (training, "output_dir")):
        if not isinstance(mapping[key], str) or not mapping[key]:
            raise ValueError(f"{key} must be a nonempty path")
        value = Path(mapping[key]).expanduser()
        mapping[key] = str(value if value.is_absolute() else (path.parent / value).resolve())
    if physical_loss is not None:
        cfg["loss"] = physical_loss
    return cfg
