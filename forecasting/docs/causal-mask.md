# Temporal ordering diagnostic for MATEY

This is a small CPU developer test in `forecasting/test_causal.py`; it does not
train, submit Slurm jobs, or alter existing models/checkpoints. Verified results
are saved in `runs/causal-mask-diagnostic/tests.log`. There is no user-facing
launcher in `jobs/`.

## Question

Can changing a later observed frame affect a representation assigned to an earlier
frame? This is different from leakage of unavailable future targets into a forecast.
The existing history-to-one-future model can validly mix its observed history, but
does not enforce ordered, causal hidden representations.

## Mask

For time-major tokens, a query at time i can see every spatial key at times j <= i.
Within-frame attention is unrestricted. A plain token-triangular mask would also
impose an unwanted ordering on spatial patches. MATEY's local spatial grouping
retains time-major ordering within each group.

`forecasting/causal.py` supplies an isolated block adapter using explicit SDPA masks,
upstream QKV packing, BF16 attention, residuals, MLP and normalization parameters.
It rejects padding and distributed execution. This is not a RingX implementation.

## Tests

1. Mask geometry permits all same-frame patches and forbids later frames.
2. With masking disabled, adapter outputs, input gradients and parameter gradients
   match the original MATEY block on a small CPU case.
3. Three successive blocks receive a perturbation to each future suffix. Compare
   the unchanged prefix for ordinary attention, masked attention, and masked
   attention with frame-local InstanceNorm. Check gradients as well as finite
   perturbations. Residual scales are deliberately 1 so dependencies are visible.
4. Independently check the upstream input normalization: it also pools over time.
   Its statistics are computed under no_grad, so a gradient-only check could miss
   its dependence on later values.

## Interpretation and remaining design decision

Masking alone cannot guarantee temporal ordering: norm1, norm2 and mlp_norm pool
statistics over the space-time sequence. The frame-local diagnostic instead folds
frames into the batch and normalizes spatial tokens separately for each frame.
This preserves affine parameters but changes the model's normalization semantics.
It is a candidate for discussion, not an approved production architecture change.

Even frame-local block normalization does not establish whole-model causality:
per-history input normalization also needs a causal alternative, such as fixed
training statistics or prefix statistics. That choice affects scale information
and output denormalization and requires a separate design decision. No training
comparison is submitted or prepared as though mask-only were sufficient.

These tests establish dependency structure, not conservation, physically meaningful
hidden states, improved spectra, or forecast accuracy. Those require training and
held-out rollout evaluation. Yalamanchi currently takes one frame, so this temporal
mask experiment applies to MATEY only.

## Follow-up training experiment

The user has now approved prefix input normalization, frame-local block norms,
and a 200-epoch causal-versus-normalization-matched-unmasked comparison. The full
model and cluster pipeline are documented in [matey-causal-training.md](matey-causal-training.md).
The earlier limitations above describe the original block-only diagnostic.
