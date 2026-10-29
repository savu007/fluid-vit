# Ordinary MATEY and Yalamanchi: 200 epochs

Submit from the project directory, with no arguments:

```bash
bash jobs/submit-forecast-200.sh
```

This is a separate experiment from finest-stage looping. Both ordinary models train from scratch for 200 epochs. MATEY retains 6,612,393 registered parameters, width 192, 12 transformer blocks, 2³ patches, three hierarchy levels and four input frames. Yalamanchi retains 8,313,683 parameters, widths 96/192/384, ten blocks, 4³ patches and one input frame. Their sizes are retained individually; they do not have equal parameter counts.

Both use five-step autonomous rollout with full gradients and normalized MSE averaged across all five steps. There are no gates, shared transformer weights, auxiliary loop predictions, or energy-loss terms. Activation checkpointing preserves gradients and reduces memory. The optimizer remains AdamW with constant learning rate 1e-4, weight decay 0.01 and clipping 1.0; extending training does not add a learning-rate schedule.

Data retains 107 train files (131–237), validation 451–490 and test 492–531. Training stays within files, with 214 windows per epoch for MATEY and 535 for Yalamanchi, batch sizes 1 and 2 respectively. Totals are 42,800 and 53,600 optimizer updates. Thus equal epochs are not equal updates, history, parameters or compute. Seed is 42. Recorded intervals and the accepted 0.3% tolerance remain in use.

Two one-GPU profiling tasks run first, with a 45-minute limit each. Each runs three complete training updates, a validation batch and a cross-file validation forecast. The measured memory gate must pass before that model schedules training. The two model chains are independent and can run concurrently.

Training proceeds in dependent one-GPU segments with 12-hour allocation limits. A segment contains at most 50 complete epochs; the actual epoch count uses the profiling estimate, 35% timing headroom and an evaluation reserve based on measured forecast time (at least 30 minutes). Estimates are not runtime guarantees. A failed or timed-out segment does not schedule another segment automatically. The most recent completed epoch is retained for recovery; no mid-epoch resume is claimed.

Each completed epoch atomically saves last.pt; best.pt tracks lowest mean five-step final-prediction validation NMSE across all 200 epochs. Resume restores model parameters, AdamW moments and step counts, Python/NumPy/Torch/CUDA RNG. Epoch-specific seeded data shuffling makes it independent of segment boundaries. Checkpoint selection does not reset at each segment.

At epoch 200 each model reloads its best validation weights and evaluates all 353 common cross-file five-step test origins. Yalamanchi takes the final observed frame; MATEY takes all four. The model that finishes last writes comparison.json and comparison.csv after checking test origins agree. This experiment evaluates five-step forecasts, not 100-step forecasts.

Outputs:

- `runs/forecast-200-<UTC timestamp>/`: profile and segment job IDs, per-model configuration, metrics, learning curves, contours, energy evolution, spectra, completion/failure reports and final comparison.
- `$PSCRATCH/fluid-vit/forecast-200-<UTC timestamp>/<model>/`: best.pt and last.pt including optimizer state.
- `logs/slurm/forecast-200-*`: scheduler stdout/stderr.

Submitting the command starts the entire pipeline; the terminal can disconnect afterward. GPU-time requests are per segment, not a total 12-hour limit for the experiment. Total training cost is estimated during profiling and can exceed 24 hours per model.
