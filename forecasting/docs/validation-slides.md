# Validation forecast presentation

Submit from the project root:

```bash
sbatch jobs/validation-slides.sbatch
```

Default checkpoint: `runs/matched-comparison-59178955/yalamanchi_vit/best.pt`.
This model was trained with five-step rollout and MSE. Inference feeds predictions
back into the model for ten saved-snapshot intervals, without ground-truth resets.
No training occurs. The saved configuration, normalization and validation split
are used, and the checkpoint cache fingerprint must match.

The job requests one GPU and a one-hour walltime. It creates a presentation for
the first chronological validation window, then calculates metrics across all
contiguous ten-step validation windows. Cross-file repeated timestamps are
removed only after checking identical fields. Recorded intervals are preserved.

Outputs under `runs/validation-slides-JOBID/`:

- `forecast-slides.pdf`: 12 pages, ready to present.
- `forecast-slides.html`: self-contained browser slideshow, with arrow-key
  navigation and fullscreen. Download the file for local presentation.
- PNGs: one per slide, suitable for importing into PowerPoint.
- `selected-forecast.npz`: initial field, ten predictions, actual fields and times.
- `report.json`: checkpoint provenance, selected-trajectory metrics and full
  validation metrics once the job completes.

The slides compare velocity magnitude and u on the z-midplane. Actual and
predicted fields share color scales, fixed across steps. Error panels show
prediction minus actual. Metrics use all three components across the full volume.
The reference data is the filtered 64-cubed cache, not the unfiltered originals.
Units are undocumented. Pressure is not included. Validation data was used for
checkpoint selection, so these are not independent test results.

A verified CPU export of the selected window is already available under
`runs/validation-slides-preview/`. It does not include full-validation evaluation.
The Slurm job has not been submitted automatically.

For another selected-window export, use a new output directory:

```bash
python -B -m forecasting.presentation --checkpoint runs/matched-comparison-59178955/yalamanchi_vit/best.pt --output runs/my-slides --device cuda --skip-full-validation
```
