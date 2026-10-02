"""Differentiable original-scale spectral regularization; no PDE residual."""
import math
import torch
from torch import nn


def validate_loss(settings):
    if not isinstance(settings, dict) or set(settings) != {"name", "spectral_weight", "energy_floor_fraction"}:
        raise ValueError("loss requires name, spectral_weight, energy_floor_fraction")
    if settings["name"] != "mse_log_spectrum":
        raise ValueError("Unsupported physical loss")
    for key in ("spectral_weight", "energy_floor_fraction"):
        value = settings[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"loss.{key} must be finite and positive")


class PhysicalForecastLoss(nn.Module):
    """Mean per-field log shell-energy error plus normalized velocity MSE."""
    def __init__(self, grid_size, stats, settings):
        super().__init__()
        validate_loss(settings)
        if type(grid_size) is not int or grid_size < 4 or grid_size % 2:
            raise ValueError("Spectral loss requires an even cubic grid >=4")
        mean = torch.tensor(stats["mean"], dtype=torch.float32)
        std = torch.tensor(stats["std"], dtype=torch.float32)
        if mean.shape != (3,) or std.shape != (3,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("Need three finite channel means and positive standard deviations")
        self.grid_size = grid_size
        self.weight = settings["spectral_weight"]
        self.floor_fraction = settings["energy_floor_fraction"]
        self.register_buffer("mean", mean[None, :, None, None, None])
        self.register_buffer("std", std[None, :, None, None, None])
        # Training-derived total kinetic-energy scale, fixed across samples/leads.
        scale = .5 * (mean.square() + std.square()).sum()
        if not torch.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid training energy scale")
        self.register_buffer("energy_scale", scale)
        modes = (torch.fft.fftfreq(grid_size, dtype=torch.float64) * grid_size).round()
        axes = torch.meshgrid(modes, modes, modes, indexing="ij")
        shells = torch.sqrt(sum(axis.square() for axis in axes)).floor().long().flatten()
        self.register_buffer("shell_index", shells)
        self.shell_count = int(shells.max()) + 1
        present = torch.bincount(shells, minlength=self.shell_count) > 0
        present[0] = False
        self.register_buffer("active_shells", present)

    def spectrum(self, velocity):
        """Full complex FFT, shell SUM; sum of shells equals total kinetic energy."""
        if velocity.ndim != 5 or tuple(velocity.shape[1:]) != (3,) + (self.grid_size,) * 3:
            raise ValueError("Expected physical velocity [B,3,N,N,N]")
        coefficients = torch.fft.fftn(velocity, dim=(-3, -2, -1), norm="forward")
        energy = .5 * (coefficients.real.square() + coefficients.imag.square()).sum(dim=1)
        energy = energy.flatten(1)
        shells = energy.new_zeros((velocity.shape[0], self.shell_count))
        return shells.scatter_add(1, self.shell_index[None].expand(velocity.shape[0], -1), energy)

    def per_example(self, prediction, target):
        """Inputs are normalized; return unreduced batch MSE and spectral loss."""
        if prediction.shape != target.shape:
            raise ValueError("Prediction and target shapes differ")
        mse = (prediction - target).square().flatten(1).mean(1)
        pred_e = self.spectrum(prediction * self.std + self.mean)[:, self.active_shells]
        true_e = self.spectrum(target * self.std + self.mean)[:, self.active_shells]
        pred_log = (pred_e / self.energy_scale + self.floor_fraction).log()
        true_log = (true_e / self.energy_scale + self.floor_fraction).log()
        spectral = (pred_log - true_log).square().mean(1)
        return mse, spectral

    def rollout(self, model, initial, targets, intervals):
        if targets.ndim != 6 or targets.shape[1] < 1 or intervals.shape != targets.shape[:2]:
            raise ValueError("Need [B,H,C,N,N,N] targets and [B,H] intervals")
        mses, spectra = [], []
        state = initial
        for h in range(targets.shape[1]):
            state = model(state, intervals[:, h])
            mse, spectral = self.per_example(state, targets[:, h])
            mses.append(mse.mean())
            spectra.append(spectral.mean())
        mse = torch.stack(mses).mean()
        spectral = torch.stack(spectra).mean()
        return mse + self.weight * spectral, mse, spectral


@torch.no_grad()
def evaluate_physical(model, loader, criterion, device):
    model.eval()
    horizon = loader.dataset.horizon
    totals = torch.zeros(horizon, 2, dtype=torch.float64, device=device)
    samples = 0
    for initial, targets, intervals in loader:
        state = (initial.to(device) - criterion.mean) / criterion.std
        targets = (targets.to(device) - criterion.mean[:, None]) / criterion.std[:, None]
        intervals = intervals.to(device)
        for h in range(horizon):
            state = model(state, intervals[:, h])
            mse, spectral = criterion.per_example(state, targets[:, h])
            if not torch.isfinite(mse).all() or not torch.isfinite(spectral).all():
                raise FloatingPointError("Nonfinite validation loss")
            totals[h, 0] += mse.double().sum()
            totals[h, 1] += spectral.double().sum()
        samples += initial.shape[0]
    if not samples:
        raise ValueError("No validation windows")
    values = (totals / samples).cpu().tolist()
    leads = [{"lead_steps": i+1, "normalized_mse": row[0], "spectral_loss": row[1],
              "combined_loss": row[0] + criterion.weight * row[1]} for i, row in enumerate(values)]
    return {"samples": samples, "horizons": leads,
            "mean_combined_loss": sum(row["combined_loss"] for row in leads) / horizon,
            "mean_normalized_mse": sum(row["normalized_mse"] for row in leads) / horizon,
            "mean_spectral_loss": sum(row["spectral_loss"] for row in leads) / horizon}
