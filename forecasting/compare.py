"""Train both architectures with matched settings and save comparable results."""
import argparse
import copy
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import subprocess
import sys

import yaml

from .checkpoint import components
from .dataset import CacheIndex, SequenceDataset


def validate_comparison(configs):
    left, right = configs
    if left["seed"] != right["seed"] or left["data"] != right["data"]:
        raise ValueError("Comparison requires identical seed and data settings")
    settings = [{k: v for k, v in c["training"].items() if k != "output_dir"} for c in configs]
    if settings[0] != settings[1] or settings[0]["objective"] != "rollout":
        raise ValueError("Comparison requires identical rollout training settings")
    for key in ("grid_size", "in_channels", "dropout"):
        if left["model"][key] != right["model"][key]:
            raise ValueError(f"Comparison requires identical model.{key}")


def validate_matched_specs(configs):
    """Match declared capacity controls; preserve architecture-specific choices."""
    validate_comparison(configs)
    left, right = [cfg["model"] for cfg in configs]
    for key in ("patch_size", "mlp_ratio"):
        if left[key] != right[key]:
            raise ValueError(f"Matched specs require identical model.{key}")
    if left["depth"] != sum(right["depths"]):
        raise ValueError("Matched specs require equal total transformer blocks")
    head_dim = left["embed_dim"] // left["num_heads"]
    stage_head_dims = [right["embed_dim"] * 2**i // heads
                       for i, heads in enumerate(right["num_heads"])]
    if any(dim != head_dim for dim in stage_head_dims):
        raise ValueError("Matched specs require equal attention head dimensions")
    counts = {}
    for name, cfg in zip(("baseline_vit", "yalamanchi_vit"), configs):
        # Counting is independent of initialized values. Preserve the caller's RNG.
        import torch
        with torch.random.fork_rng(devices=[]):
            model = components(name)[1](**cfg["model"])
        counts[name] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        del model
    gap = abs(counts["baseline_vit"] - counts["yalamanchi_vit"]) / counts["yalamanchi_vit"]
    if gap > 0.05:
        raise ValueError("Matched specs require parameter counts within 5% of Yalamanchi")
    return {"patch_size": left["patch_size"], "transformer_blocks": left["depth"],
            "head_dimension": head_dim, "mlp_ratio": left["mlp_ratio"],
            "parameters": counts, "parameter_gap_fraction_of_yalamanchi": gap,
            "parameter_gap_limit": 0.05,
            "compute_matched": False, "architecture_matched": False}


def slurm_prefixes():
    """Pin the two independent workers to distinct nodes in this allocation."""
    nodelist = os.environ.get("SLURM_JOB_NODELIST")
    if not os.environ.get("SLURM_JOB_ID") or not nodelist:
        raise ValueError("--launcher slurm requires a Slurm allocation")
    nodes = subprocess.check_output(["scontrol", "show", "hostnames", nodelist], text=True).split()
    if len(set(nodes)) < 2:
        raise ValueError("The parallel comparison requires at least two allocated nodes")
    cpus = os.environ.get("SLURM_CPUS_PER_TASK", "8")
    return [["srun", "--exclusive", "--exact", "--nodes=1", "--ntasks=1",
             "--ntasks-per-node=1", "--gpus-per-task=1", "--cpu-bind=cores",
             f"--cpus-per-task={cpus}", f"--nodelist={node}"] for node in nodes[:2]]


def run_model(architecture, original, output, splits, prefix):
    cfg = copy.deepcopy(original)
    run = output / architecture
    cfg["training"]["output_dir"] = str(run)
    config_path = output / f"{architecture}.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"Training {architecture}; log: {output / (architecture + '.log')}", flush=True)
    with (output / f"{architecture}.log").open("w") as log:
        subprocess.run([*prefix, sys.executable, "-u", "-m", f"{architecture}.train",
                        "--config", str(config_path)], stdout=log, stderr=subprocess.STDOUT, check=True)
    reports = {}
    for split in splits:
        report_path = run / f"evaluation-{split}.json"
        with report_path.open("w") as stream, (output / f"{architecture}.log").open("a") as log:
            subprocess.run([*prefix, sys.executable, "-m", f"{architecture}.evaluate",
                            "--checkpoint", str(run / "best.pt"), "--device", cfg["training"]["device"],
                            "--split", split, "--horizon", str(cfg["training"]["horizon"]),
                            "--plot", str(run / f"velocity-{split}.png")],
                           stdout=stream, stderr=log, check=True)
        reports[split] = json.loads(report_path.read_text())
    return reports


def save_results(output, results):
    ordered = {name: results[name] for name in sorted(results)}
    (output / "comparison.json").write_text(json.dumps(ordered, indent=2, allow_nan=False))
    rows = []
    for name, reports in ordered.items():
        for split, report in reports.items():
            for lead in report["rollout"]["horizons"]:
                rows.append({"architecture": name, "split": split,
                             "checkpoint_epoch": report["checkpoint_epoch"], **lead})
    if rows:
        with (output / "comparison.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-config", default="baseline_vit/config-rollout.yaml")
    parser.add_argument("--yalamanchi-config", default="yalamanchi_vit/config-rollout.yaml")
    parser.add_argument("--output", required=True, type=Path, help="New comparison directory")
    parser.add_argument("--launcher", choices=("local", "slurm"), default="local",
                        help="local: serial; slurm: concurrent workers on two allocated nodes")
    parser.add_argument("--test", action="store_true", help="Also evaluate held-out test after validation selection")
    parser.add_argument("--matched-specs", action="store_true",
                        help="Enforce equal patches/depth/head dimension/MLP ratio and <=5% parameter gap")
    args = parser.parse_args(argv)
    architectures = ("baseline_vit", "yalamanchi_vit")
    configs = [components(name)[0](path) for name, path in
               zip(architectures, (args.baseline_config, args.yalamanchi_config))]
    validate_comparison(configs)
    specs = validate_matched_specs(configs) if args.matched_specs else None
    for cfg in configs:
        index = CacheIndex(cfg)
        for split in ("train", "val", "test") if args.test else ("train", "val"):
            SequenceDataset(index, split, cfg["training"]["horizon"])
    prefixes = slurm_prefixes() if args.launcher == "slurm" else [[], []]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if specs is not None:
        (output / "matched-specs.json").write_text(json.dumps(specs, indent=2))
    (output / "launch.json").write_text(json.dumps(
        {"launcher": args.launcher, "workers": dict(zip(architectures, prefixes))}, indent=2))
    results, failures = {}, {}
    splits = ("val", "test") if args.test else ("val",)
    with ThreadPoolExecutor(max_workers=2 if args.launcher == "slurm" else 1) as pool:
        tasks = {pool.submit(run_model, name, cfg, output, splits, prefix): name
                 for name, cfg, prefix in zip(architectures, configs, prefixes)}
        for task in as_completed(tasks):
            name = tasks[task]
            try:
                results[name] = task.result()
                save_results(output, results)
            except Exception as error:
                failures[name] = str(error)
                print(f"{name} failed; see {output / (name + '.log')}: {error}", file=sys.stderr, flush=True)
    if failures:
        (output / "failures.json").write_text(json.dumps(failures, indent=2))
        raise RuntimeError("Comparison incomplete; successful results retained. See failures.json")
    print(f"Comparison complete: {output / 'comparison.csv'}", flush=True)


if __name__ == "__main__":
    main()
