# Retired physical-loss experiment

Retired at the project owner’s request on October 3, 2026; treated as an
unsuccessful experiment and excluded from active training and test discovery.
This classification is a project decision, not evidence that every spectral loss
is ineffective. The run did not demonstrate a convincing prediction improvement;
a direct spectral comparison against MSE-only checkpoints was not performed.

This directory preserves the experiment source, original config-loader hooks,
job script, documentation, Slurm logs, and run 59184738 including weights.
Paths and imports inside this historical snapshot retain their original meanings.
It is NOT a standalone runnable package: do not submit its archived job script.
Restoring it requires reconstructing the original paths and config hooks.

The source layout mirrors the old repository. No archived module is imported by
active training. Existing weights remain loadable through forecasting.checkpoint
using the original architecture tags. Load checkpoints from their new location:
archive/physical_loss/runs/physical-loss-59184738/<model>/best.pt.
Embedded output paths in historical JSON remain unchanged for provenance.

Both jobs timed out after 12 hours. In this physical-loss run, baseline completed
63 epochs (best checkpoint 62) and Yalamanchi completed 100. Weights were retained;
optimizer state was not saved. See forecasting/docs/physical-loss-experiment.md
here for the historical loss definition, not current submission instructions.
