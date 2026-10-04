# Runtime-targeted comparison

Use jobs/runtime-comparison.sbatch with the two config-runtime.yaml files.
This supersedes matched-specs as the recommended experiment; historical configs
and model classes are unchanged, so old checkpoints remain compatible.

Baseline uses embedding width 240, ten attention heads, and ten transformer
blocks. Head dimension remains 24. Patch size is now 8³, giving an 8³ token grid
(512 tokens) from each 64³ velocity snapshot. The output remains a full 64³ field.
Yalamanchi is unchanged: widths 96/192/384/192/96 across five two-block stages,
heads [4,8,16], and 4³ initial patches.

Measured MSE-only median epoch times for the original 4³-patch baseline and
Yalamanchi on A100 were 676 and 197 seconds respectively. Increasing baseline
patch size from 4³ to 8³ reduces tokens eightfold and attention score pairs
64-fold. This is not a 64-fold wall-clock speedup: patch embedding/reconstruction,
MLPs, data loading, and other work still contribute. No timing for the new setup
has been measured; verify metrics.jsonl before claiming a runtime match.

Larger patches change spatial tokenization, and the embedding/reconstruction
layers have more parameters per patch. This is a compute-reduction experiment,
not a parameter-matched or purely attention-architecture comparison. Training
starts from scratch; previous checkpoints remain usable with their saved configs,
but their patch-layer weights do not fit the new patch size directly.

Both use identical data splits, seed, batch size 2, 100 epochs, five-step MSE,
AdamW settings, float32, and recorded timestep intervals with the approved
0.3% spacing tolerance. The full scratch cache remains unchanged.

The new job uses the existing comparison validator for shared training settings,
but omits --matched-specs because capacities now differ. The job retains a 24-hour limit because the reduced baseline is not benchmarked.

Physical-loss training is retired under archive/physical_loss/. Active loaders
reject a loss section rather than silently ignoring it. The archive includes
source and config-loader snapshots for provenance; it is not an executable package.
