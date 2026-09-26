# Source audit and reuse decision

Checked September 25, 2026. This is a bounded public-source search, not proof
that unpublished or unindexed code does not exist.

## Result

**No verified public repository for the Yalamanchi fluid-forecasting model was
located.** No author code, pretrained weights, or third-party files have been
downloaded into this project.

A runnable original local implementation now accompanies the plan. See
[current implementation](../README.md) for provenance and limitations.

## Sources inspected

| Source | What was established |
|---|---|
| [Paper abstract](https://arxiv.org/abs/2604.02483) and [v1 full text](https://arxiv.org/html/2604.02483v1) | Architecture and forecasting methodology; no model-code repository identified in the inspected text |
| [Kiran Yalamanchi's public GitHub profile](https://github.com/kiranyalamanchi) | Visible repositories did not identify this model |
| [Shivam Barwey's public repository list](https://github.com/sbarwey?tab=repositories) | Listed projects did not identify this model |
| [Official Microsoft Swin repository](https://github.com/microsoft/Swin-Transformer) | A verified upstream source for Swin/SwinV2 building blocks, not this fluid model |
| [Official SwinV2 source file](https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer_v2.py) | Concrete attention and block semantics to use as a semantic reference |
| [Microsoft repository license](https://github.com/microsoft/Swin-Transformer/blob/main/LICENSE) | Repository carries the MIT license |

Searches included the exact paper title, arXiv identifier, author names combined
with GitHub/SwinV2, and GitHub-domain searches for the fluid model.
Search results were discovery aids; the plan's technical references point to
the paper or official source.

The arXiv HTML page has GitHub links for reporting rendering errors. Those
links are not an implementation repository for the scientific model.

A video Swin repository was also considered as a possible 3D indexing reference,
but its page fetch failed during this audit; no claims about its implementation
or suitability were used.

## What can be reused

The official SwinV2 file is a candidate reference for a future 3D port.
It is a 2D backbone implementation, not a complete velocity forecasting
encoder–decoder. It cannot be inserted unchanged into the current 3D pipeline.

Concrete adaptation work would include:

- 3D window indexing, partitioning, reversal, and relative coordinates.
- Explicit periodic boundary treatment.
- Eight-child merging and expansion instead of four-child image operations.
- A decoder with matching feature skips.
- Three velocity input/output channels and residual output convention.
- A timestep branch, training objectives, and dataset integration.

Existing local cache validation, normalization, metrics, and plotting helpers
are stronger immediate reuse candidates than a third-party training script.

If upstream code is copied later, pin a commit, retain its license and attribution,
and record local modifications in a provenance file. No upstream revision has
been vendored or pinned by this implementation.

Image-pretrained weights are not part of the initial plan: their dimensions
and task differ, and using them would introduce another experimental variable.

## What is not established

Do not silently fill these gaps with assumptions labeled as paper facts:

- Exact author implementation, configuration files, and released checkpoints.
- Precise block counts, width/window choices, and stage ordering in code.
- Optimizer settings, normalization conventions, and initialization details.
- Boundary masks and padding details.
- Whether any implementation-specific modifications depart from canonical SwinV2.
- Exact data-split and execution details needed to reproduce reported results.

The architecture plan gives explicit local defaults so development can proceed
without presenting these uncertainties as solved.

## Recommended path

Implement the documented 3D adaptation using verified SwinV2 semantics and the
existing local forecasting pipeline. Keep the plain ViT as an independent
reference. If an author repository becomes available, compare it against this
plan before changing provenance or claiming reproduction.

This recommendation is an engineering proposal, not a claim that the adapted
architecture will improve accuracy. That conclusion requires the planned
controlled experiments.
