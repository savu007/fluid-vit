# Causal MATEY training preparation

Prepared, not submitted. Command: `bash jobs/submit-matey-causal.sh`.

Approved: 200 epochs each, causal versus normalization-matched unmasked attention;
107 training files; four input frames; five-step rollout normalized MSE.

Both production models: 6,612,393 parameters, width 192, three hierarchy levels,
four blocks per level. Prefix input normalization and frame-local block norms in
both arms; temporal mask in causal arm only.

30 tests passed. Full-model tests perturb later physical input frames and check
all earlier token prefixes after every block at all levels. Checkpointed gradients
match direct execution. Tiny end-to-end tests cover profiling, train/resume/test,
contours, spectra and final comparison. Segmented training produces exactly the
same final weights as uninterrupted training for both arms in the tiny fixture.

Fake Slurm tests passed for the two-task profile array, separate afterok training
chains, and no successor after failure or completion. No real jobs submitted.
Production GPU memory/timing remain to be measured by the three-update gate.

See forecasting/docs/matey-causal-training.md for precise model changes, allocation
limits, outputs, scientific interpretation and recovery semantics.
