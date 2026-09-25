"""Shared normalization, device selection, and one-step evaluation."""

import math

import torch


def select_device(name):
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Run inside your GPU allocation, or explicitly select cpu.")
    return torch.device(name)


def normalization_tensors(stats, device):
    mean = torch.tensor(stats["mean"], dtype=torch.float32, device=device)[None, :, None, None, None]
    std = torch.tensor(stats["std"], dtype=torch.float32, device=device)[None, :, None, None, None]
    return mean, std


@torch.no_grad()
def evaluate(model, loader, stats, device):
    model.eval()
    mean, std = normalization_tensors(stats, device)
    totals = {k: 0.0 for k in ("normalized", "error", "persistence", "target", "predicted")}
    count = 0
    for x, y, dt in loader:
        x, y, dt = x.to(device), y.to(device), dt.to(device)
        prediction = model((x - mean) / std, dt) * std + mean
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("Nonfinite evaluation prediction")
        # Accumulate diagnostics in float64; predictions themselves remain float32.
        error = prediction.double() - y.double()
        values = {
            "normalized": (error / std.double()).square().sum(),
            "error": error.square().sum(),
            "persistence": (x.double() - y.double()).square().sum(),
            "target": y.double().square().sum(),
            "predicted": prediction.double().square().sum(),
        }
        for key, value in values.items():
            totals[key] += value.item()
        count += y.numel()
    if not count:
        raise ValueError("Evaluation split has no pairs")
    reference = totals["target"]
    return {
        "normalized_mse": totals["normalized"] / count,
        "physical_mse": totals["error"] / count,
        "relative_l2": math.sqrt(totals["error"] / reference) if reference else None,
        "persistence_mse": totals["persistence"] / count,
        "persistence_relative_l2": math.sqrt(totals["persistence"] / reference) if reference else None,
        "mse_skill_vs_persistence": 1 - totals["error"] / totals["persistence"] if totals["persistence"] else None,
        # For the configured three velocity channels, volume-mean kinetic energy.
        "predicted_kinetic_energy": 0.5 * totals["predicted"] / count * len(stats["mean"]),
        "target_kinetic_energy": 0.5 * reference / count * len(stats["mean"]),
        "samples": len(loader.dataset),
    }
