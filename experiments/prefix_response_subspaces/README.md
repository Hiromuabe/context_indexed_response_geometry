# Prefix-specific one-step response subspaces

For the final BlackboxNLP experiment set (two main experiments plus one cheap
checkpoint replication), use [PAPER_README.md](PAPER_README.md) and
`run_paper_pipeline`. The older `run_pipeline` remains only for auditing the
superseded Matched-control development runs.

This package implements the two BlackboxNLP experiments in the attached specification. It uses normal forwards (never generation) for forced successor tokens, single-process `torch.nn.DataParallel` when multiple GPUs are visible, problem-level splits, disjoint calibration/training/evaluation tokens, split-local double centering, and row-level artifacts.

If the configured trajectory file has fewer unique GSM8K problems than the
requested prefix pool, `build_prefix_pool` invokes the repository's existing
deterministic greedy trajectory generator. The response-subspace smoke profile
uses its own 48-problem trajectory file so it can populate the required
32-problem prefix pool; it does not silently shrink the experiment to the older
8-problem successor-subspace smoke artifact.

Run the full smoke pipeline from the repository root:

```bash
python3 -m experiments.prefix_response_subspaces.run_pipeline \
  --config experiments/prefix_response_subspaces/configs/smoke.yaml
```

On the GPU server:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m experiments.prefix_response_subspaces.run_pipeline \
  --config experiments/prefix_response_subspaces/configs/pilot.yaml
```

Every stage also has its own module entry point. Completed manifests with matching config/input hashes are reused; mismatched artifacts fail instead of being overwritten. Experiment 2 is gated on Experiment 1 and a successful Oracle reinjection check.

Representative stage commands are:

```bash
python3 -m experiments.prefix_response_subspaces.build_prefix_pool --config experiments/prefix_response_subspaces/configs/smoke.yaml
python3 -m experiments.prefix_response_subspaces.build_candidate_tokens --config experiments/prefix_response_subspaces/configs/smoke.yaml
python3 -m experiments.prefix_response_subspaces.match_prefixes --config experiments/prefix_response_subspaces/configs/smoke.yaml
python3 -m experiments.prefix_response_subspaces.extract_successor_states --config experiments/prefix_response_subspaces/configs/smoke.yaml
python3 -m experiments.prefix_response_subspaces.compute_contrast_residuals --config experiments/prefix_response_subspaces/configs/smoke.yaml
python3 -m experiments.prefix_response_subspaces.analyze_geometry --config experiments/prefix_response_subspaces/configs/smoke.yaml
python3 -m experiments.prefix_response_subspaces.analyze_functional_recovery --config experiments/prefix_response_subspaces/configs/smoke.yaml
```

The full run is:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m experiments.prefix_response_subspaces.run_pipeline \
  --config experiments/prefix_response_subspaces/configs/full.yaml
```

The package intentionally does not implement state-conditioned projectors, subspace prediction from the current state, steering, final-answer interventions, or attention/MLP decomposition.

## Audit-sensitive diagnostics

- Matched-prefix quality reports raw and log(2)-normalized JS, random-pair improvement, an absolute preconfigured ceiling, and the number of valid evaluation matches. Gate 5 cannot pass on a development quantile alone.
- High-probability sensitivity uses full-tokenizer-vocabulary ranks at top-128 and top-256, not a probability quantile over the selected candidate set.
- Rank-0 stability uses a fixed stratified nested calibration order and includes an exact full-M development anchor on the same cells. Test and development rank-0 means are labeled separately.
- Permutation p-values are marked invalid when the configured length/progress strata do not contain enough exchangeable prefixes.
- Functional Gate 7a (Local > 0 and Local > Global) is independent of matching quality. Gate 7b (Local > Matched) additionally requires a valid prediction match.

For pilot, run the large-prefix-pool pre-branch audit before any successor-state
branching:

```bash
python3 -m experiments.prefix_response_subspaces.run_pipeline \
  --config experiments/prefix_response_subspaces/configs/pilot.yaml \
  --through-stage match_prefixes
cat results/prefix_response_subspaces/pilot_v4/matches/prebranch_matching_gate.json
```

Matching requires the same top-1 prediction, maximizes top-5 then top-20
overlap under the fixed length/progress and different-problem constraints, and
only then minimizes full-tokenizer-vocabulary NJS. If the pre-branch gate fails,
the pipeline can still run Local/Global/Content and functional Gate 7a; Matched,
Gate 5, Gate 7b, and any “beyond current prediction” claim remain disabled.
