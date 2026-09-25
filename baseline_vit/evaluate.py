"""Evaluate a saved checkpoint on validation or the held-out test split."""

import argparse
import json

import torch
from torch.utils.data import DataLoader

from .dataset import CacheIndex, PairDataset
from .engine import evaluate, select_device
from .model import BaselineViT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    device = select_device(args.device)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = saved["config"]
    index = CacheIndex(cfg)
    if index.fingerprint != saved["cache_fingerprint"]:
        raise ValueError("Cache metadata changed since training; refusing an inconsistent evaluation")
    dataset = PairDataset(index, args.split)
    loader = DataLoader(dataset, batch_size=cfg["training"]["batch_size"],
                        num_workers=cfg["training"]["num_workers"], shuffle=False)
    model = BaselineViT(**cfg["model"]).to(device)
    model.load_state_dict(saved["model"])
    metrics = evaluate(model, loader, saved["normalization"], device)
    print(json.dumps({"split": args.split, "checkpoint_epoch": saved["epoch"],
                      "metrics": metrics}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
