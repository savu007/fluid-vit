"""Explicit training command. Importing this module starts no work."""

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import load_config
from .dataset import CacheIndex, PairDataset, training_statistics
from .engine import evaluate, normalization_tensors, select_device
from .model import BaselineViT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    options = cfg["training"]
    device = select_device(options["device"])
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    index = CacheIndex(cfg)
    train = PairDataset(index, "train", options["max_train_pairs"])
    val = PairDataset(index, "val")
    print(f"device={device}, training pairs={len(train)}, validation pairs={len(val)}, "
          f"reference dt={index.dt_reference:g}", flush=True)
    output = Path(options["output_dir"])
    # Refuse to overwrite an earlier experiment. Choose a new directory per run.
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(cfg, indent=2))
    print("Computing normalization from training files only...", flush=True)
    stats = training_statistics(index)
    (output / "normalization.json").write_text(json.dumps(stats, indent=2))
    common = dict(batch_size=options["batch_size"], num_workers=options["num_workers"],
                  pin_memory=device.type == "cuda")
    generator = torch.Generator().manual_seed(cfg["seed"])
    train_loader = DataLoader(train, shuffle=True, generator=generator, **common)
    val_loader = DataLoader(val, shuffle=False, **common)
    model = BaselineViT(**cfg["model"]).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=options["learning_rate"],
                                  weight_decay=options["weight_decay"])
    mean, std = normalization_tensors(stats, device)
    best = float("inf")
    for epoch in range(1, options["epochs"] + 1):
        start = time.perf_counter()
        model.train()
        loss_sum, elements = 0.0, 0
        for x, y, dt in train_loader:
            x, y, dt = x.to(device), y.to(device), dt.to(device)
            x, y = (x - mean) / std, (y - mean) / std
            optimizer.zero_grad(set_to_none=True)
            loss = (model(x, dt) - y).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), options["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            loss_sum += loss.item() * y.numel()
            elements += y.numel()
        metrics = evaluate(model, val_loader, stats, device)
        record = {"epoch": epoch, "training_mse": loss_sum / elements,
                  "validation": metrics, "seconds": time.perf_counter() - start}
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record, allow_nan=False), flush=True)
        if metrics["normalized_mse"] < best:
            best = metrics["normalized_mse"]
            checkpoint = {
                "model": model.state_dict(), "config": cfg, "normalization": stats,
                "epoch": epoch, "validation": metrics, "dt_reference": index.dt_reference,
                "cache_fingerprint": index.fingerprint,
            }
            temporary = output / "best.pt.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(output / "best.pt")
    print(f"Finished. Best validation MSE={best:g}. Checkpoint: {output / 'best.pt'}", flush=True)
    print("The test set has not been evaluated. Use baseline_vit.evaluate separately.", flush=True)


if __name__ == "__main__":
    main()
