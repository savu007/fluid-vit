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
    parser.add_argument("--plot", metavar="PNG", help="Save a target/prediction/error slice comparison")
    parser.add_argument("--sample-index", type=int, default=0, help="Pair index within the selected split")
    parser.add_argument("--slice-axis", choices=("x", "y", "z"), default="z")
    parser.add_argument("--slice-index", type=int, help="Voxel index along slice-axis (default: middle)")
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
    if args.plot:
        from .visualize import plot_prediction
        plot_prediction(model, dataset, saved["normalization"], device, args.plot,
                        sample_index=args.sample_index, axis=args.slice_axis,
                        slice_index=args.slice_index)
    metrics = evaluate(model, loader, saved["normalization"], device)
    print(json.dumps({"split": args.split, "checkpoint_epoch": saved["epoch"],
                      "metrics": metrics, **({"plot": args.plot} if args.plot else {})},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
