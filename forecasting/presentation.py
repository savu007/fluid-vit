"""Export a validation forecast as a PDF and self-contained HTML slideshow."""
import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from baseline_vit.dataset import CacheIndex
from baseline_vit.engine import normalization_tensors, select_device
from .checkpoint import check_cache, read_checkpoint, restore_model
from .dataset import CrossFileSequenceDataset
from .rollout import evaluate_rollout


@torch.no_grad()
def forecast_window(model, initial, targets, intervals, stats, device):
    """Forecast recursively; targets are used only to measure errors."""
    model.eval()
    mean, std = normalization_tensors(stats, device)
    state = (initial.unsqueeze(0).to(device) - mean) / std
    predictions = []
    for interval in intervals:
        state = model(state, interval.reshape(1).to(device))
        physical = state * std + mean
        if not torch.isfinite(physical).all():
            raise FloatingPointError(f"Nonfinite forecast at step {len(predictions) + 1}")
        predictions.append(physical[0].cpu().numpy())
    prediction = np.stack(predictions)
    truth = targets.numpy()
    metrics = []
    for step, (predicted, actual) in enumerate(zip(prediction, truth), 1):
        error = predicted.astype(np.float64) - actual
        reference = np.square(actual.astype(np.float64)).sum()
        persistence = np.square(initial.numpy().astype(np.float64) - actual).mean()
        mse = np.square(error).mean()
        metrics.append(dict(lead_steps=step, physical_mse=float(mse),
                            relative_l2=float(np.sqrt(np.square(error).sum() / reference)) if reference else None,
                            persistence_mse=float(persistence),
                            mse_skill_vs_persistence=float(1 - mse / persistence) if persistence else None,
                            predicted_kinetic_energy=float(0.5 * np.square(predicted.astype(np.float64)).sum(axis=0).mean()),
                            target_kinetic_energy=float(0.5 * np.square(actual.astype(np.float64)).sum(axis=0).mean())))
    return prediction, metrics


def render_deck(output, initial, truth, prediction, times, metadata, metrics, axis="z", slice_index=None):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.figure import Figure

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    spatial_axis = ("x", "y", "z").index(axis)
    side = initial.shape[spatial_axis + 1]
    plane = side // 2 if slice_index is None else slice_index
    if not 0 <= plane < side:
        raise ValueError("Slice index outside grid")
    actual = np.take(truth, plane, axis=spatial_axis + 2)
    predicted = np.take(prediction, plane, axis=spatial_axis + 2)
    actual_speed = np.sqrt(np.square(actual).sum(axis=1))
    predicted_speed = np.sqrt(np.square(predicted).sum(axis=1))
    speed_limit = max(float(actual_speed.max()), float(predicted_speed.max()), 1e-12)
    component_limit = max(float(np.abs(actual[:, 0]).max()), float(np.abs(predicted[:, 0]).max()), 1e-12)
    error_limits = [max(float(np.abs(predicted_speed - actual_speed).max()), 1e-12),
                    max(float(np.abs(predicted[:, 0] - actual[:, 0]).max()), 1e-12)]
    images = []
    remaining = [a for a in ("x", "y", "z") if a != axis]
    with PdfPages(output / "forecast-slides.pdf") as pdf:
        def save(figure, name):
            FigureCanvasAgg(figure)
            path = output / name
            figure.savefig(path, dpi=120)
            pdf.savefig(figure)
            images.append(path)
            figure.clear()

        intro = Figure(figsize=(16, 9), layout="constrained")
        intro.text(0.07, 0.85, "Turbulence forecasting · validation", fontsize=30, weight="bold")
        info = (f"Model: {metadata['architecture']} | checkpoint epoch: {metadata['epoch']}\n"
                f"Training: {metadata['objective']}, horizon {metadata['training_horizon']}\n"
                f"Forecast: {len(prediction)} recursive steps; no ground-truth resets\n"
                f"Start: file {metadata['start']['file_id']}, frame {metadata['start']['frame']}, t={times[0]:.6f}\n"
                f"End: t={times[-1]:.6f} | elapsed simulation time: {times[-1] - times[0]:.6f}\n"
                f"Filtered {side}³ velocity data | fixed {axis}={plane} slice\n"
                "Fixed color scales across all leads; errors are prediction minus actual.\n"
                "Field units are undocumented; metrics use original cached field values.\n"
                "This trajectory is the first validation window, selected before evaluation.\n"
                "Validation was used for checkpoint selection; these are not independent test results.")
        intro.text(0.07, 0.68, info, fontsize=19, va="top", linespacing=1.7)
        save(intro, "00-context.png")
        for i in range(len(prediction)):
            fig = Figure(figsize=(16, 9), layout="constrained")
            axes = fig.subplots(2, 3)
            for row, (target, pred, limit, label) in enumerate([
                    (actual_speed[i], predicted_speed[i], speed_limit, "Speed |velocity|"),
                    (actual[i, 0], predicted[i, 0], component_limit, "x-velocity u")]):
                for col, (field, heading) in enumerate([(target, "Actual"), (pred, "Predicted"), (pred - target, "Prediction − actual")]):
                    signed = row == 1 or col == 2
                    cap = error_limits[row] if col == 2 else limit
                    art = axes[row, col].imshow(field.T, origin="lower", interpolation="nearest",
                                               cmap="RdBu_r" if signed else "viridis",
                                               vmin=-cap if signed else 0, vmax=cap)
                    axes[row, col].set_title(f"{label}: {heading}", fontsize=15)
                    axes[row, col].set_xlabel(f"{remaining[0]} voxel")
                    axes[row, col].set_ylabel(f"{remaining[1]} voxel")
                    fig.colorbar(art, ax=axes[row, col], shrink=0.85)
            m = metrics[i]
            relative = 'undefined' if m['relative_l2'] is None else f"{m['relative_l2']:.3f}"
            fig.suptitle(f"Lead {i + 1}/{len(prediction)} · t={times[i + 1]:.6f} · Δt={times[i + 1] - times[0]:.6f}\n"
                         f"Full-volume relative L2={relative} | MSE={m['physical_mse']:.4g} | {axis}={plane}", fontsize=20)
            save(fig, f"{i + 1:02d}-forecast.png")
        summary = Figure(figsize=(16, 9), layout="constrained")
        ax = summary.subplots(1, 2)
        leads = [m['lead_steps'] for m in metrics]
        ax[0].plot(leads, [m['physical_mse'] for m in metrics], "o-", label="Model")
        ax[0].plot(leads, [m['persistence_mse'] for m in metrics], "o--", label="Persistence: keep initial field")
        ax[0].set_ylabel("Full-volume MSE")
        ax[1].plot(leads, [m['predicted_kinetic_energy'] for m in metrics], "o-", label="Predicted")
        ax[1].plot(leads, [m['target_kinetic_energy'] for m in metrics], "o--", label="Actual")
        ax[1].set_ylabel("Mean kinetic energy: ½〈u²+v²+w²〉")
        for panel in ax:
            panel.set_xlabel("Forecast lead (saved-snapshot intervals)")
            panel.legend()
            panel.grid(alpha=0.3)
        summary.suptitle("Selected validation trajectory · error and energy", fontsize=24)
        save(summary, f"{len(prediction) + 1:02d}-summary.png")
    tags = []
    for i, path in enumerate(images):
        data = base64.b64encode(path.read_bytes()).decode()
        tags.append(f'<img class="slide" alt="Slide {i + 1}" src="data:image/png;base64,{data}">')
    html = '''<!doctype html><html><head><meta charset="utf-8"><title>Validation forecast</title>
<style>body{margin:0;background:#151515;color:white;font-family:sans-serif}.slide{display:none;width:100vw;height:94vh;object-fit:contain}.active{display:block}nav{text-align:center;height:6vh}button{font-size:18px;margin:0 12px}</style></head><body>'''
    html += ''.join(tags)
    html += '''<nav><button onclick="show(n-1)">Previous</button><span id="count"></span><button onclick="show(n+1)">Next</button><button onclick="document.documentElement.requestFullscreen()">Fullscreen</button></nav>
<script>let n=0;const slides=Array.from(document.querySelectorAll('.slide'));function show(i){n=Math.max(0,Math.min(slides.length-1,i));slides.forEach((s,j)=>s.classList.toggle('active',j===n));document.getElementById('count').textContent=(n+1)+' / '+slides.length}document.addEventListener('keydown',e=>{if(e.key==='ArrowRight'||e.key===' '){e.preventDefault();show(n+1)}if(e.key==='ArrowLeft')show(n-1)});show(0);</script></body></html>'''
    (output / "forecast-slides.html").write_text(html)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--architecture", choices=("baseline_vit", "yalamanchi_vit"), default="yalamanchi_vit")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--slice-axis", choices=("x", "y", "z"), default="z")
    parser.add_argument("--slice-index", type=int)
    parser.add_argument("--skip-full-validation", action="store_true", help="Export the first validation window only")
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    device = select_device(args.device)
    saved = read_checkpoint(args.checkpoint, args.architecture)
    index = CacheIndex(saved['config'])
    check_cache(saved, index)
    data_cfg = saved['config']['data']
    dataset = CrossFileSequenceDataset(index, 'val', args.horizon,
                                      dt_rtol=data_cfg['dt_rtol'], dt_atol=data_cfg['dt_atol'])
    model = restore_model(saved, device, args.architecture)
    initial, targets, intervals = dataset[0]
    prediction, metrics = forecast_window(model, initial, targets, intervals, saved['normalization'], device)
    checkpoint = Path(args.checkpoint).resolve()
    metadata = dict(architecture=args.architecture, epoch=saved['epoch'],
                    objective=saved['config']['training'].get('objective', 'single_step'),
                    training_horizon=saved['config']['training'].get('horizon', 1),
                    checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    start=dataset.starts()[0], split='val', selection='first chronological window',
                    validation_windows=len(dataset), horizon=args.horizon,
                    cache_fingerprint=index.fingerprint, dt_reference=index.dt_reference,
                    slice_axis=args.slice_axis, slice_index=args.slice_index if args.slice_index is not None else initial.shape[1] // 2)
    times = dataset.times[:args.horizon + 1]
    render_deck(output, initial.numpy(), targets.numpy(), prediction, times, metadata, metrics,
                args.slice_axis, args.slice_index)
    np.savez_compressed(output / 'selected-forecast.npz', initial=initial.numpy(), actual=targets.numpy(),
                        predicted=prediction, times=times, normalized_intervals=intervals.numpy())
    report = dict(metadata=metadata, selected_trajectory=metrics)
    report_path = output / 'report.json'
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f"Slides and selected forecast saved to {output}", flush=True)
    if not args.skip_full_validation:
        print(f"Evaluating all {len(dataset)} validation windows", flush=True)
        report['full_validation'] = evaluate_rollout(model, DataLoader(dataset, batch_size=1, num_workers=0), saved['normalization'], device)
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False))
    print('Complete', flush=True)


if __name__ == '__main__':
    main()
