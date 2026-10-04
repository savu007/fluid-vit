"""Matched architecture with an explicit differentiable physical-loss interface."""
from .model import BaselineViT


class PhysicalViT(BaselineViT):
    # No added weights: checkpoints retain the original forward-model layout.
    def rollout_loss(self, initial, targets, intervals, criterion):
        return criterion.rollout(self, initial, targets, intervals)
