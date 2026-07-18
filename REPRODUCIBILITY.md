# Reproducibility record

## Experimental design

- Main dataset: GSM8K `main`, train split.
- External dataset: CommonsenseQA `default`, train split.
- Statistical unit: original problem ID.
- Prefixes: one sampled prefix per problem.
- Candidate inputs: 320 shared tokens, including 64 calibration tokens and
  256 analysis tokens.
- Candidate evaluation: four disjoint 192-fit/64-held-out folds.
- Controls: target-context PCA, matched-common subspace, and five wrong-context
  donors.
- Inference: four folds are averaged within problem before problem-level
  bootstrap confidence intervals are calculated.
- External replication: decoder block 0 and rank 64 are fixed before evaluation;
  ranks 1, 2, 4, 8, 16, 32, and 64 form the sensitivity curve.

## Model record

The main and CommonsenseQA runs used `Qwen/Qwen2.5-Math-1.5B`.
The resolved checkpoint revision recorded by the executed run was:

```text
4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2
```

Configurations retain the requested Hub revision and every extraction manifest
records the resolved revision returned by Transformers.
When exact artifact reproduction matters, compare the manifest revision rather
than relying on a mutable branch name.

## Shared-direction control

Each candidate fold fits pooled PCA, CPC, and a non-orthogonal dictionary using
only the 192 training candidates.
Held-out EV uses the disjoint 64-candidate evaluation subset.

The full control grid contains:

- dictionary widths 64, 96, 128, 160, 192, and 256;
- evaluation ranks 1, 2, 4, 8, 16, 32, and 64;
- dictionary coherence penalties 0, 0.0001, and 0.001;
- five deterministic optimization restarts per fit;
- 2,000 optimization steps per restart.

Restarts are selected by training loss, never by held-out EV.
The reported primary dictionary condition fixes width 256, penalty 0, rank 64,
and `leave_one_context_out=False`.

## Artifact flow

1. `scripts/prepare_reasoning_trajectories.py` generates deterministic greedy
   trajectories when the configured JSONL is absent.
2. `build_prefix_pool` creates deterministic problem-level groups and prefix
   positions.
3. `build_candidate_tokens` fixes the shared candidate set and four analysis
   folds.
4. `extract_successor_states` writes hidden-state arrays.
5. `compute_contrast_residuals` applies split-local double centering.
6. Geometry and control modules consume the saved arrays and write summaries.
7. `summarize_geometry_controls` applies the same problem bootstrap to GSM8K
   and CommonsenseQA rank curves.

The resolved configuration, candidate split, prefix snapshot, wrong-context
assignments, and stage manifests are the authoritative execution record.
Do not combine artifacts from different result roots unless their model
revision, configuration hash, prefix axis, and candidate axis agree.

## Publication hygiene

Generated manifests may contain an explicit local checkpoint path when
`--model-path` is used.
Do not commit generated `data/`, `results/`, `logs/`, or manifest archives to an
anonymous repository without inspecting and sanitizing them first.

The checked-in source and example figures contain no local paths, usernames,
hostnames, author names, email addresses, or repository-history metadata.
