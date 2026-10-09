"""Differentiable global kinetic-energy matching for a separate rollout experiment."""
import math

import torch


def kinetic_energy(physical):
    """[B,3,X,Y,Z] -> [B]; includes mean flow, with no density/volume factor."""
    if physical.ndim != 5 or physical.shape[1] != 3:
        raise ValueError('Expected [B,3,X,Y,Z] physical velocity')
    return .5 * physical.square().sum(dim=1).mean(dim=(1, 2, 3))


def energy_floor(stats):
    """Numerical denominator floor: 1e-8 times training mean kinetic energy."""
    reference = .5 * sum(float(s)**2 + float(m)**2 for s, m in zip(stats['std'], stats['mean']))
    if not math.isfinite(reference) or reference <= 0:
        raise ValueError('Training energy scale must be positive and finite')
    return max(1e-8 * reference, 1e-12)


def rollout_loss(model, initial, targets, intervals, mean, std, weight, floor):
    """Normalized inputs/targets; mean/std restore physical velocity differentiably.

    All leads receive equal weight. No detach or target insertion between calls.
    Energy mismatch is squared per sample before averaging, preventing errors
    of opposite sign in different samples from cancelling.
    """
    if not math.isfinite(weight) or weight < 0 or not math.isfinite(floor) or floor <= 0:
        raise ValueError('Weight must be nonnegative and denominator floor positive')
    if targets.ndim != 6 or intervals.shape != targets.shape[:2] or targets.shape[1] < 1:
        raise ValueError('Need [B,H,3,X,Y,Z] targets and [B,H] recorded intervals')
    mse_terms, energy_terms = [], []
    state = initial
    for lead in range(targets.shape[1]):
        state = model(state, intervals[:, lead])
        target = targets[:, lead]
        mse_terms.append((state - target).square().mean())
        predicted_energy = kinetic_energy(state * std + mean)
        target_energy = kinetic_energy(target * std + mean)
        relative_error = (predicted_energy - target_energy) / target_energy.clamp_min(floor)
        energy_terms.append(relative_error.square().mean())
    mse = torch.stack(mse_terms).mean()
    energy = torch.stack(energy_terms).mean()
    return mse + weight * energy, mse, energy
