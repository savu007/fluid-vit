"""Render one forecast as physical-velocity slices without a GUI or display server."""

from pathlib import Path

import numpy as np
import torch

from .engine import normalization_tensors


@torch.no_grad()
def plot_prediction(model, dataset, stats, device, output, *, sample_index=0,
                    axis="z", slice_index=None):
    """Save a PNG: rows are channels, columns are target, prediction, and error.

    The selected plane fixes one spatial coordinate; axes use voxel indices
    because the cache does not record physical spatial coordinates.
    """
    if not 0 <= sample_index < len(dataset):
        raise ValueError(f"sample_index must be between 0 and {len(dataset) - 1}")
    if axis not in ("x", "y", "z"):
        raise ValueError("axis must be x, y, or z")
    output = Path(output)
    if output.suffix.lower() != ".png":
        raise ValueError("Plot output must have a .png extension")
    x, target, dt = dataset[sample_index]
    spatial_axis = ("x", "y", "z").index(axis)
    size = target.shape[spatial_axis + 1]
    if slice_index is None:
        slice_index = size // 2
    if not 0 <= slice_index < size:
        raise ValueError(f"slice_index must be between 0 and {size - 1}")

    mean, std = normalization_tensors(stats, device)
    model.eval()
    prediction = model((x.unsqueeze(0).to(device) - mean) / std,
                       dt.reshape(1).to(device)) * std + mean
    if not torch.isfinite(prediction).all():
        raise FloatingPointError("Nonfinite visualization prediction")
    predicted_plane = np.take(prediction[0].cpu().numpy(), slice_index,
                              axis=spatial_axis + 1)
    target_plane = np.take(target.cpu().numpy(), slice_index, axis=spatial_axis + 1)

    # Import only when requested, so ordinary training/evaluation needs no matplotlib.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    channels = dataset.index.manifest["channels"]
    file_id, time_index = dataset.pairs[sample_index]
    times = dataset.index.times[file_id]
    remaining_axes = [name for name in ("x", "y", "z") if name != axis]
    figure = Figure(figsize=(12, 3.4 * len(channels)), layout="constrained")
    FigureCanvasAgg(figure)
    panels = figure.subplots(len(channels), 3, squeeze=False)
    for channel_index, channel in enumerate(channels):
        truth = target_plane[channel_index]
        predicted = predicted_plane[channel_index]
        error = predicted - truth
        # Shared limits make target and forecast directly comparable per channel.
        velocity_limit = max(float(np.abs(truth).max()),
                             float(np.abs(predicted).max()), 1e-12)
        error_limit = max(float(np.abs(error).max()), 1e-12)
        for column, (values, title, limit) in enumerate((
            (truth, "Target", velocity_limit),
            (predicted, "ViT prediction", velocity_limit),
            (error, "Prediction - target", error_limit),
        )):
            panel = panels[channel_index, column]
            artist = panel.imshow(values.T, origin="lower", cmap="RdBu_r",
                                  vmin=-limit, vmax=limit, interpolation="nearest")
            panel.set_title(f"{channel}: {title}")
            panel.set_xlabel(f"{remaining_axes[0]} voxel")
            panel.set_ylabel(f"{remaining_axes[1]} voxel")
            figure.colorbar(artist, ax=panel, label="Velocity (physical values)")
    figure.suptitle(
        f"File {file_id}, t={times[time_index]:.6g} → {times[time_index + 1]:.6g}"
        f" | {axis}={slice_index} | sample {sample_index}"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, format="png")
    return output
