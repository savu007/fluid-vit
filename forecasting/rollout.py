"""Training objectives and common-start autoregressive metrics."""

import math

import torch

from baseline_vit.engine import normalization_tensors


def forecast_loss(model, x, targets, intervals, objective):
    """Inputs/targets are already normalized; targets are [B,H,C,X,Y,Z]."""
    horizon = targets.shape[1]
    if objective not in ("single_step", "rollout"):
        raise ValueError("Unknown objective")
    if intervals.shape != (x.shape[0], horizon) or horizon < 1:
        raise ValueError("Need one interval per example per future step")
    if objective == "single_step" and horizon != 1:
        raise ValueError("single_step requires one target")
    losses = []
    for step in range(horizon):
        x = model(x, intervals[:, step])
        losses.append((x - targets[:, step]).square().mean())
    return torch.stack(losses).mean()


@torch.no_grad()
def evaluate_rollout(model, loader, stats, device):
    """Every lead uses the SAME starts; persistence holds the initial field fixed."""
    model.eval()
    mean, std = normalization_tensors(stats, device)
    horizon = loader.dataset.horizon
    keys = ("normalized", "error", "absolute", "persistence", "target", "prediction")
    totals = [{key: 0.0 for key in keys} for _ in range(horizon)]
    count = 0
    for initial, targets, intervals in loader:
        initial, targets, intervals = initial.to(device), targets.to(device), intervals.to(device)
        state = (initial - mean) / std
        for step in range(horizon):
            state = model(state, intervals[:, step])
            prediction = state * std + mean
            if not torch.isfinite(prediction).all():
                raise FloatingPointError(f"Nonfinite forecast at horizon {step + 1}")
            target = targets[:, step].double()
            error = prediction.double() - target
            values = {
                "normalized": (error / std.double()).square().sum(),
                "error": error.square().sum(),
                "absolute": error.abs().sum(),
                "persistence": (initial.double() - target).square().sum(),
                "target": target.square().sum(),
                "prediction": prediction.double().square().sum(),
            }
            for key, value in values.items():
                totals[step][key] += value.item()
        count += initial.numel()
    if not count:
        raise ValueError("No evaluation windows")
    metrics = []
    for lead, values in enumerate(totals, 1):
        metrics.append({
            "lead_steps": lead,
            "normalized_mse": values["normalized"] / count,
            "physical_mse": values["error"] / count,
            "physical_mae": values["absolute"] / count,
            "relative_l2": math.sqrt(values["error"] / values["target"]) if values["target"] else None,
            "persistence_mse": values["persistence"] / count,
            "persistence_relative_l2": math.sqrt(values["persistence"] / values["target"]) if values["target"] else None,
            "mse_skill_vs_persistence": 1 - values["error"] / values["persistence"] if values["persistence"] else None,
            "predicted_kinetic_energy": 0.5 * values["prediction"] / count * len(stats["mean"]),
            "target_kinetic_energy": 0.5 * values["target"] / count * len(stats["mean"]),
            "samples": len(loader.dataset),
        })
    return {"horizon": horizon, "samples": len(loader.dataset), "horizons": metrics,
            "mean_normalized_mse": sum(m["normalized_mse"] for m in metrics) / horizon}
