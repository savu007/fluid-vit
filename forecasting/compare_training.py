"""Yalamanchi-only, equal-epoch pushforward versus rollout experiment."""
import argparse
import csv
import json
import math
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from baseline_vit.engine import select_device
from .checkpoint import check_cache, read_checkpoint, restore_model
from .dataset import CacheIndex, CrossFileSequenceDataset, SequenceDataset
from .rollout import evaluate_rollout
from .train import train
from yalamanchi_vit.config import load_config
from yalamanchi_vit.model import ARCHITECTURE, YalamanchiViT

STRATEGIES = ("rollout", "pushforward")
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "yalamanchi_vit/config-runtime.yaml"


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def prepare(config, output):
    cfg = load_config(config)
    if cfg["training"]["objective"] != "rollout" or cfg["training"]["horizon"] != 5:
        raise ValueError("Source config must use five-step rollout")
    count = max(50, math.ceil(len(cfg["data"]["train"]) / 3))
    if count > len(cfg["data"]["train"]):
        raise ValueError("Not enough training files for requested subset")
    index = CacheIndex(cfg)
    selected = index.groups["train"][:count]
    cfg["data"]["train"] = selected
    index = CacheIndex(cfg)
    train_data = SequenceDataset(index, "train", 5)
    val_data = SequenceDataset(index, "val", 5)
    test = CrossFileSequenceDataset(index, "test", 100,
                                    dt_rtol=cfg["data"]["dt_rtol"], dt_atol=cfg["data"]["dt_atol"])
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    spec = {
        "source_config": str(Path(config).resolve()),
        "training_files": selected, "training_file_count": count,
        "epochs": cfg["training"]["epochs"], "seed": cfg["seed"],
        "training_horizon": 5, "test_horizon": 100,
        "training_windows": len(train_data), "validation_windows": len(val_data),
        "test_windows": len(test), "test_unique_snapshots": len(test.frames),
        "test_starts": test.starts(), "cache_fingerprint": index.fingerprint,
        "selection": "lowest mean five-step validation normalized MSE",
        "comparison": "equal epochs and updates; not equal elapsed time",
        "losses": {"rollout": "mean MSE across all five steps, full gradients",
                   "pushforward": "terminal MSE, first four steps without gradients"},
    }
    for strategy in STRATEGIES:
        cfg["training"]["objective"] = strategy
        cfg["training"]["output_dir"] = str(output / strategy)
        (output / (strategy + ".yaml")).write_text(yaml.safe_dump(cfg, sort_keys=False))
    write_json(output / "experiment.json", spec)
    print(json.dumps({key: value for key, value in spec.items()
                      if key not in ("test_starts", "training_files")}), flush=True)


def evaluate_test(checkpoint, spec, device):
    saved = read_checkpoint(checkpoint, "yalamanchi_vit")
    cfg = saved["config"]
    index = CacheIndex(cfg)
    check_cache(saved, index)
    if index.fingerprint != spec["cache_fingerprint"]:
        raise ValueError("Cache changed after experiment preparation")
    dataset = CrossFileSequenceDataset(index, "test", spec["test_horizon"],
                                       dt_rtol=cfg["data"]["dt_rtol"], dt_atol=cfg["data"]["dt_atol"])
    if dataset.starts() != spec["test_starts"]:
        raise ValueError("Test windows changed between workers")
    model = restore_model(saved, device, "yalamanchi_vit")
    # Batch one bounds memory for 100 target fields; no training graph retained.
    report = evaluate_rollout(model, DataLoader(dataset, batch_size=1, shuffle=False,
                                                num_workers=0), saved["normalization"], device)
    return {"checkpoint": str(checkpoint), "checkpoint_epoch": saved["epoch"],
            "objective": cfg["training"]["objective"], "split": "test",
            "test_starts": dataset.starts(), "rollout": report}


def worker(output, strategy):
    output = Path(output).resolve()
    cfg = load_config(output / (strategy + ".yaml"))
    spec = json.loads((output / "experiment.json").read_text())
    if cfg["training"]["objective"] != strategy:
        raise ValueError("Worker strategy differs from resolved config")
    run = output / strategy
    try:
        train(cfg, YalamanchiViT, ARCHITECTURE, save_last=True)
        print("Training complete; beginning held-out 100-step evaluation", flush=True)
        result = evaluate_test(run / "best.pt", spec, select_device(cfg["training"]["device"]))
        write_json(run / "evaluation-test-100.json", result)
        print("100-step test evaluation complete", flush=True)
    except Exception as error:
        write_json(output / (strategy + "-failure.json"),
                   {"type": type(error).__name__, "message": str(error)})
        raise


def summarize(output):
    output = Path(output).resolve()
    reports, missing, rows = {}, [], []
    for strategy in STRATEGIES:
        path = output / strategy / "evaluation-test-100.json"
        if not path.exists():
            missing.append(strategy)
            continue
        report = json.loads(path.read_text())
        reports[strategy] = report
        rows.extend({"strategy": strategy, "checkpoint_epoch": report["checkpoint_epoch"], **lead}
                    for lead in report["rollout"]["horizons"])
    if len(reports) == 2 and reports["rollout"]["test_starts"] != reports["pushforward"]["test_starts"]:
        raise ValueError("Cannot compare different evaluation starts")
    write_json(output / "comparison.json", {"complete": not missing, "missing": missing,
                                           "results": reports})
    if rows:
        with (output / "comparison.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if missing:
        raise RuntimeError("Incomplete experiment; retained available results: " + ", ".join(missing))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "worker", "summarize"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--strategy", choices=STRATEGIES)
    args = parser.parse_args(argv)
    if args.action == "prepare":
        prepare(args.config, args.output)
    elif args.action == "summarize":
        summarize(args.output)
    else:
        if args.strategy is None:
            parser.error("worker requires --strategy")
        worker(args.output, args.strategy)


if __name__ == "__main__":
    main()
