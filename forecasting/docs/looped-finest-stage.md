# Finest-stage looping diagnostic: MATEY and Yalamanchi

This is a forecasting adaptation inspired by [Looped Diffusion Transformer](https://arxiv.org/html/2609.40305v1), especially its intermediate-loop supervision and headwise sigmoid gates (§2). It is **not** a reproduction of the diffusion model. We retain deterministic next-field forecasting, the original inputs, spatial hierarchies, and five-step recursive training. No diffusion noise or sampling time is introduced.

## Approved experiment

Four fresh runs, each for 25 epochs, one GPU per run:

| Run | Finest processor | Supervision | Gate |
|---|---|---|---|
| MATEY control | Four independently weighted blocks | Final prediction | Original attention |
| MATEY looped | Two blocks, applied twice: A → B → A → B | Decode after B at both loop boundaries | Token-dependent sigmoid per head |
| Yalamanchi control | Two independently weighted Swin blocks | Final prediction | Original attention |
| Yalamanchi looped | Shared block parameters, applied unshifted then shifted | Decode after each application | Token-dependent sigmoid per head |

Only the final, finest-resolution processor changes. Other levels/stages, hierarchy assembly, patches, widths, time conditioning, and residual paths retain the existing implementations. No existing production factory, job or vendored source is edited.

- **MATEY:** history 4; physical 64³ velocity fields; factor-2 field hierarchy 16³ → 32³ → 64³; 2³ patches per level; width 192; three heads; four applications at each of three levels, 12 total. The treatment shares the final level's blocks 0/2 and 1/3, yielding ten distinct transformer blocks overall. Levels 0 and 1 remain independently weighted.
- **Yalamanchi:** history 1; physical 64³ fields normalized using train statistics; 4³ patches; token hierarchy 16³ → 8³ → 4³ → 8³ → 16³; widths 96/192/384; heads 4/8/16; 4³-token windows; two applications at each of five stages, ten total. The final decoder stage shares its two blocks' parameters, yielding nine distinct transformer blocks overall. The first application remains unshifted; the second remains shifted. This is geometry-dependent reuse of shared weights, rather than repeating an identical spatial operator.

The initial stages are reproduced using the same seed within each architecture. Shared blocks take their initial learned tensors from the first occurrence; they are not averaged from two pretrained blocks. These are fresh models, not fine-tuned checkpoints.

## Exact parameter-sharing semantics

Every learned parameter within each selected block is shared: attention projections, Q/K normalization where present, relative-bias MLP where present, attention gates, normalization affine parameters, MLP and layer-scale parameters. Modules at different positions remain distinct objects. Consequently deterministic masks/window shifts and stochastic-depth probabilities remain tied to the application position.

In MATEY the original four finest-level drop-path probabilities remain 0, approximately 0.01818, 0.03636, 0.05455. Sharing parameters does not repeat the first pair's probabilities. Lead-time conditioning remains in the first block application only, exactly as in the original stage; it is not injected again at loop 2. This is another departure from strict repeated application of an identical function.

Verified registered parameter counts are 6,612,393 → 5,719,471 for MATEY and 8,313,683 → 8,198,227 for Yalamanchi (including the upstream MATEY modules that are registered but unused in 3D).

The optimizer sees each shared parameter once and accumulates contributions from both occurrences. Checkpoints have duplicate named entries for shared tensors because positional modules remain distinct; PyTorch serializes their shared storage. The dedicated loader verifies that aliases agree before loading. These new checkpoints use architecture identifier `finest_stage_looped_forecast_v1`; existing evaluators must not load them as ordinary MATEY/Yalamanchi weights.

## Gate and its placement

For token i and head h, the added gate is

    g[i,h] = sigmoid(W[h] u[i] + b[h])
    attended[i,h,:] = g[i,h] * attended[i,h,:]

Gate weights and biases initialize to zero, so the initial gate is 0.5. Gates are learned from the loss; they are not a prescribed dissipation or physical constraint.

- **Yalamanchi:** `u` is an affine-free per-token LayerNorm of the window-attention input. Existing cosine attention, relative biases, masks and output projection are preserved. The gate multiplies each head output before head concatenation and output projection. The added normalization supplies a stable gate input without replacing the original Swin post-normalization.
- **MATEY:** `u` is the existing pre-attention InstanceNorm output, per token. The gate multiplies head outputs before the existing post-attention InstanceNorm and output projection. Preserve upstream QKV packing, Q/K normalization and BF16 Q/K/V attention casts. Because upstream normalizes the concatenated attention output, constant rescaling can largely be normalized away; token-dependent/head-dependent variation can still change the representation. This is an explicit architectural difference from the paper's vanilla attention setting.

The looped variant supports only the current full-volume, single-GPU, unpadded MATEY path. RingX spatial partitioning is outside this experiment.

## What each decoded output means

Both loop predictions target **the same next physical timestep**. Loops refine a representation inside one forecast call. They do not advance simulation time.

Yalamanchi decodes the first finest decoder block's output through the same final norm, unpatch layer and convolutional head as the final prediction, adding the same last input field as its residual.

MATEY captures the finest processor's tokens after its second application. It reverses the actual local-token grouping used upstream (including the upstream grouping-size clamp), decodes through the same patch decoder, and applies the same hierarchy formula:

    auxiliary = [candidate - U(F(candidate)) + base] * history_std + history_mean

`base` is the actual upsampled coarser prediction already computed for the final branch; `F` is the existing no-gradient filter; `U` is the existing learned upsampler. The auxiliary decode repeats the candidate-removal upsampling but reuses the coarse base. It does not treat an intermediate normalized candidate as a physical velocity field. Filtering retains the upstream no-gradient behavior; this experiment does not repair or redefine it.

Only the final loop output goes into the next recursive forecast context. Auxiliary outputs contribute to loss but never enter the history.

## Loss and gradients

For five physical forecast steps, with loop predictions p[k,1], p[k,2] and DNS targets y[k]:

    NMSE(p,y) = mean_elements[((p-y)/training_channel_std)^2]
    L_looped = (1/5) sum_k {0.5 NMSE(p[k,1],y[k]) + 0.5 NMSE(p[k,2],y[k])}
    L_control = (1/5) sum_k NMSE(p[k],y[k])

For Yalamanchi this is implemented equivalently in already-normalized coordinates. There is no energy, spectral, KL or physics term. Means cancel in the error. Equal loop weights are used; the paper's flow-matching weighting is not copied.

All five forecast calls retain autograd. Later losses backpropagate through earlier final predictions. This is rollout training, not pushforward or teacher forcing. Activation checkpointing recomputes transformer blocks while preserving RNG and the MATEY attention backend. Intermediate captures are disabled during backward recomputation to avoid retaining extra graphs.

## Data and controls

Both architectures use the existing 107 training files 131–237, validation files 451–490 and test files 492–531 from scratch cache `hit-coarse-64-131-531`. Preserve the accepted 0.3% interval tolerance and recorded intervals.

Training and validation retain each architecture's existing within-file window construction and batch size:

| | MATEY | Yalamanchi |
|---|---:|---:|
| History frames | 4 | 1 |
| Forecast targets per sample | 5 | 5 |
| Train windows | 214 | 535 |
| Validation windows | 80 | 200 |
| Batch size | 1 | 2 |
| Updates per epoch | 214 | 268 |
| Updates over 25 epochs | 5,350 | 6,700 |

Within each architecture, treatment/control share data, seed 42, batch size, AdamW learning rate 1e-4, weight decay 0.01, clipping 1.0, epochs and validation selection (mean final-output NMSE across five leads). No global mixed-precision mode is added; MATEY retains its existing attention-only BF16 casts. One seed makes this a diagnostic, not a statistical claim of superiority.

Test forecasts use the same 353 deduplicated cross-file origins for all four models; Yalamanchi takes the last of the four observed frames. No validation/test target enters a forecast input. Test is evaluated only after selecting the best validation checkpoint.

The comparison is **matched in block applications**, not exact FLOPs or training wall time. Gates, auxiliary decoding and its backward pass add work. Parameter counts decrease through sharing and increase slightly through gates; measured totals are recorded. The three components change together, so a gain would not establish that parameter sharing alone caused it. The initial question is whether this combined treatment is useful at approximately the existing inference depth.

## Submission and allocation

Run the entire pipeline with no arguments:

```bash
bash jobs/submit-looped-pipeline.sh
```

The four profiling tasks run first. A CPU coordination job depends on successful completion of the whole array, validates all four profiles, then submits four independent GPU training/evaluation jobs with measured allocations. A final CPU job writes the comparison after all training jobs terminate; it reports failure if any result is incomplete. No terminal session needs to remain connected. Job IDs are recorded in the experiment directory. CPU coordination uses account m5401, shared QoS; GPU runs use m5401_g. No jobs are submitted while preparing the code.

The individual commands below remain available for manual staging; do not run them additionally for an already submitted automatic pipeline.

First submit the profiling array (four separate one-GPU tasks, each capped at 45 minutes):

```bash
bash jobs/submit-looped-profile.sh
```

The command prints a report directory. No directory arguments are needed: training and summary automatically select the newest submitted looped experiment. They never silently fall back to an older completed profile. Wait for all four profile tasks to complete. Each performs **three actual full-size five-step training updates**, including backward and AdamW, one native-validation batch and one cross-file validation forecast. These weights are discarded. The timed cross-file forecast uses validation data; test predictions are reserved for evaluation after training. A peak-allocation gate requires less than 90% of device capacity. CPU tests do not establish full-size GPU acceptance.

Then:

```bash
bash jobs/submit-looped-training.sh
```

The launcher reads all four gates and verifies configuration and model/training source fingerprints before submitting anything. It estimates training + validation for 25 epochs and full 353-origin test evaluation, adds 35% timing headroom and 30 minutes fixed reserve. Within each architecture, both runs receive the larger of their two estimates. Requests above 24 hours are rejected for a later explicit segmentation decision. This is a runtime estimate, not a guarantee; a three-update gate cannot prove that a run will never OOM.

Four independent one-GPU training jobs can run concurrently subject to scheduling. Existing weights are not reused. Each run saves atomic best/last checkpoints with optimizer state to:

    $PSCRATCH/fluid-vit/<experiment-directory-name>/<case>/

Configurations, profiles, metrics, learning curves, speed contours, energy-versus-lead plots, spectra, small slice arrays, and completion/failure reports go under:

    runs/looped-finest-<UTC timestamp>/<case>/

No full test fields or model weights are copied to home. A missing `complete.json` means completion has not been certified. Partial epoch metrics/checkpoints remain if interrupted; automatic resume is not implemented.

After all four finish, run `bash jobs/summarize-looped.sh` to write `comparison.csv` and `comparison.json`. The summarizer refuses missing results or different test origins. Compare physical MSE, normalized MSE, relative L2, kinetic energy and spectra, with contours as qualitative context. This initial diagnostic tests five-step forecasts; it does not yet evaluate 100-step rollouts.

## Local verification

```bash
module load pytorch/2.8.0
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
  .venv/matey-memory-torch28/bin/python -B -m unittest \
  forecasting.test_looped forecasting.test_training_strategies ornl_vit.memory.test_memory -v
```

Tests cover shared-gradient accumulation, original-control identity, preserved shifts/drop-path, independent intermediate-decode equality, consistent checkpoint aliases, checkpointed output/gradient parity including gates, the five-step/dual-output loss and recursion, and tiny end-to-end profiling/training/reloading/plotting for both architectures.
