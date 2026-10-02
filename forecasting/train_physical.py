"""Separate MSE-plus-spectrum experiment; existing MSE training stays unchanged."""

import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from baseline_vit.engine import normalization_tensors, select_device
from .dataset import CacheIndex, SequenceDataset, training_statistics
from .rollout import evaluate_rollout
from .physical_loss import PhysicalForecastLoss, evaluate_physical


def train(cfg, model_class, architecture):
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
    criterion = PhysicalForecastLoss(cfg["model"]["grid_size"], stats, cfg["loss"]).to(device)
    metadata = {
        "architecture": architecture, "torch": str(torch.__version__), "numpy": np.__version__,
        "device": str(device), "parameters": sum(p.numel() for p in model.parameters()),
        "initialization": None, "loss": cfg["loss"],
        "spectral_energy_scale": criterion.energy_scale.item(),
        "spectral_shells": criterion.active_shells.nonzero().flatten().tolist(),
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
    best = {"best.pt": float("inf"), "best-mse.pt": float("inf")}
    for epoch in range(1, options["epochs"] + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        model.train()
        loss_sum, mse_sum, spectral_sum, examples = 0.0, 0.0, 0.0, 0
        for x, targets, intervals in train_loader:
            x, targets, intervals = x.to(device), targets.to(device), intervals.to(device)
            x = (x - mean) / std
            targets = (targets - mean[:, None]) / std[:, None]
            optimizer.zero_grad(set_to_none=True)
            loss, mse, spectral = model.rollout_loss(x, targets, intervals, criterion)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), options["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            loss_sum += loss.item() * x.shape[0]
            mse_sum += mse.item() * x.shape[0]
            spectral_sum += spectral.item() * x.shape[0]
            examples += x.shape[0]
        report = evaluate_physical(model, val_loader, criterion, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        record = {
            "epoch": epoch, "objective": options["objective"], "horizon": options["horizon"],
            "training_loss": loss_sum / examples, "training_mse": mse_sum / examples,
            "training_spectral_loss": spectral_sum / examples,
            "weighted_training_spectral_loss": criterion.weight * spectral_sum / examples,
            "validation": report,
            "optimizer_updates": epoch * len(train_loader),
            "training_forward_examples": epoch * len(train) * options["horizon"],
            "seconds": time.perf_counter() - start,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        }
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record, allow_nan=False), flush=True)
        for filename, metric in (("best.pt", "mean_combined_loss"),
                                  ("best-mse.pt", "mean_normalized_mse")):
            if report[metric] < best[filename]:
                best[filename] = report[metric]
                checkpoint = {
                    "format_version": 1, "architecture": architecture,
                    "model": model.state_dict(), "config": cfg, "normalization": stats,
                    "epoch": epoch, "validation": report, "dt_reference": index.dt_reference,
                    "cache_fingerprint": index.fingerprint, "environment": metadata,
                    "checkpoint_selection": metric,
                }
                temporary = output / (filename + ".tmp")
                torch.save(checkpoint, temporary)
                temporary.replace(output / filename)
    saved = torch.load(output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(saved["model"])
    report = evaluate_rollout(model, val_loader, stats, device)
    (output / "evaluation-val.json").write_text(json.dumps({
        "checkpoint_epoch": saved["epoch"], "selection": saved["checkpoint_selection"],
        "physical_loss": saved["validation"], "rollout": report}, indent=2, allow_nan=False))
    from baseline_vit.dataset import PairDataset
    from baseline_vit.visualize import plot_prediction
    plot_prediction(model, PairDataset(index, "val"), stats, device, output / "velocity-val.png")
    print(f"Finished physical-loss experiment: {output}; best losses={best}", flush=True)
    print("Test values were not evaluated. No optimizer-resume state is saved.", flush=True)


def main(architecture, argv=None):
    import argparse
    import importlib
    from .checkpoint import components
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parents[1] / architecture / "config-physical.yaml")
    parser.add_argument("--output", type=Path, help="Override with a NEW run directory")
    args = parser.parse_args(argv)
    load_config, _, tag = components(architecture)
    cfg = load_config(args.config, allow_physical_loss=True)
    if cfg["training"]["objective"] != "rollout":
        raise ValueError("Physical experiment requires full-gradient rollout training")
    if args.output:
        cfg["training"]["output_dir"] = str(args.output.resolve())
    model_class = importlib.import_module(architecture + ".model_physical").PhysicalViT
    train(cfg, model_class, tag)
