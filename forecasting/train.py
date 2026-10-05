"""Identical training loop for both velocity architectures."""

import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from baseline_vit.engine import normalization_tensors, select_device
from .dataset import CacheIndex, SequenceDataset, training_statistics
from .rollout import forecast_loss, evaluate_rollout


def train(cfg, model_class, architecture, *, save_last=False):
    options = cfg["training"]
    device = select_device(options["device"])
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    index = CacheIndex(cfg)
    train = SequenceDataset(index, "train", options["horizon"], options["max_train_pairs"])
    val = SequenceDataset(index, "val", options["horizon"])
    output = Path(options["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(cfg, indent=2))
    stats = training_statistics(index)
    (output / "normalization.json").write_text(json.dumps(stats, indent=2))
    model = model_class(**cfg["model"]).to(device)
    metadata = {
        "architecture": architecture, "torch": str(torch.__version__), "numpy": np.__version__,
        "device": str(device), "parameters": sum(p.numel() for p in model.parameters()),
        "initialization": None,
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    (output / "environment.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({**metadata, "training_windows": len(train), "validation_windows": len(val),
                      "reference_dt": index.dt_reference, "objective": options["objective"],
                      "horizon": options["horizon"]}), flush=True)
    common = dict(batch_size=options["batch_size"], num_workers=options["num_workers"],
                  pin_memory=device.type == "cuda")
    train_loader = DataLoader(train, shuffle=True, generator=torch.Generator().manual_seed(cfg["seed"]), **common)
    val_loader = DataLoader(val, shuffle=False, **common)
    optimizer = torch.optim.AdamW(model.parameters(), lr=options["learning_rate"],
                                  weight_decay=options["weight_decay"])
    mean, std = normalization_tensors(stats, device)
    best = float("inf")
    for epoch in range(1, options["epochs"] + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        model.train()
        loss_sum, examples = 0.0, 0
        for x, targets, intervals in train_loader:
            x, targets, intervals = x.to(device), targets.to(device), intervals.to(device)
            x = (x - mean) / std
            targets = (targets - mean[:, None]) / std[:, None]
            optimizer.zero_grad(set_to_none=True)
            loss = forecast_loss(model, x, targets, intervals, options["objective"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), options["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            loss_sum += loss.item() * x.shape[0]
            examples += x.shape[0]
        report = evaluate_rollout(model, val_loader, stats, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        record = {
            "epoch": epoch, "objective": options["objective"], "horizon": options["horizon"],
            "training_loss": loss_sum / examples, "validation": report,
            "optimizer_updates": epoch * len(train_loader),
            "training_forward_examples": epoch * len(train) * options["horizon"],
            "seconds": time.perf_counter() - start,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        }
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record, allow_nan=False), flush=True)
        improved = report["mean_normalized_mse"] < best
        if improved or save_last:
            checkpoint = {
                "format_version": 1, "architecture": architecture,
                "model": model.state_dict(), "config": cfg, "normalization": stats,
                "epoch": epoch, "validation": report, "dt_reference": index.dt_reference,
                "cache_fingerprint": index.fingerprint, "environment": metadata,
            }
            filenames = (["best.pt"] if improved else []) + (["last.pt"] if save_last else [])
            for filename in filenames:
                temporary = output / (filename + ".tmp")
                torch.save(checkpoint, temporary)
                temporary.replace(output / filename)
        if improved:
            best = report["mean_normalized_mse"]
    print(f"Finished. Best mean validation normalized MSE={best:g}; checkpoint={output / 'best.pt'}", flush=True)
    print("Test field values were not evaluated. No optimizer-resume checkpoint is saved.", flush=True)

