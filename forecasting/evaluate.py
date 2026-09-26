"""Shared one-step and common-start rollout evaluation."""
import argparse
import json
from torch.utils.data import DataLoader
from baseline_vit.dataset import CacheIndex, PairDataset
from baseline_vit.engine import evaluate, select_device
from .checkpoint import read_checkpoint, check_cache, components, restore_model
from .dataset import SequenceDataset
from .rollout import evaluate_rollout


def evaluate_checkpoint(args, architecture):
    device = select_device(args.device)
    saved = read_checkpoint(args.checkpoint, architecture)
    cfg = saved["config"]
    index = CacheIndex(cfg)
    check_cache(saved, index)
    sequences = SequenceDataset(index, args.split, args.horizon)
    pairs = PairDataset(index, args.split)
    _, _, tag = components(architecture)
    model = restore_model(saved, device, architecture)
    common = dict(batch_size=cfg["training"]["batch_size"],
                  num_workers=cfg["training"]["num_workers"])
    metrics = evaluate(model, DataLoader(pairs, shuffle=False, **common), saved["normalization"], device)
    report = evaluate_rollout(model, DataLoader(sequences, shuffle=False, **common),
                              saved["normalization"], device)
    if args.plot:
        from baseline_vit.visualize import plot_prediction
        plot_prediction(model, pairs, saved["normalization"], device, args.plot,
                        sample_index=args.sample_index, axis=args.slice_axis, slice_index=args.slice_index)
    result = {"architecture": tag, "checkpoint_epoch": saved["epoch"], "split": args.split,
              "one_step": metrics, "rollout": report}
    if args.plot:
        result["plot"] = str(args.plot)
    return result


def main(architecture, argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--plot", metavar="PNG")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--slice-axis", choices=("x", "y", "z"), default="z")
    parser.add_argument("--slice-index", type=int)
    args = parser.parse_args(argv)
    print(json.dumps(evaluate_checkpoint(args, architecture), indent=2, allow_nan=False))
