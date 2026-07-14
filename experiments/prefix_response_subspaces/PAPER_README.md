# BlackboxNLP final paper experiment suite

This is the paper-facing entry point. It contains:

1. Experiment 1: interaction energy, held-out-token rank–EV, Local versus
   length/progress-conditioned Global and five Wrong-prefix spaces, plus
   full-vocabulary Top-256 sensitivity. It also reports normalized projection
   distance, split-half within-prefix reliability versus between-prefix
   rotation, and uses an equal-prefix trace-normalized covariance Global.
2. Experiment 2: Oracle, Rank-0, Local, conditioned Global, and the mean of
   Wrong-prefix reconstructions at one development-selected layer/rank.
3. Cheap checkpoint replication: Experiment 1 only on `Qwen/Qwen2.5-1.5B`,
   using the same GSM8K prefixes, candidate token IDs, folds, fixed rank, and
   corresponding normalized layer depth.
4. First-layer mechanism localization: the candidate endpoint immediately
   before self-attention, after the attention residual addition, and after the
   MLP residual addition; plus the rank-64 first-layer value/output space
   `span{W_O E_h W_V^(h) h_i,t}` and its geometric and Functional comparison
   with the fitted Local response space.
5. Rank saturation: eight deterministic 128-fit/128-heldout re-splits of the
   saved 256 analysis-token responses at ranks 1, 2, 4, 8, 16, 32, 64, 96,
   and 127. `r90` is defined relative to rank 127. No model forward is used.
6. Shared-backbone decomposition: eigendecomposition of the development-only
   mean projector `mean_i U_i U_i^T`, followed by a held-out test of the Local
   components orthogonal to shared directions. No model forward is used.

Prediction-matched controls, current-prediction claims, steering, learned
projectors, multistep transitions, per-head causal interventions, and
final-answer interventions are not called by this pipeline.

## Commands

Smoke without the checkpoint replication:

```bash
python3 -m experiments.prefix_response_subspaces.run_paper_pipeline \
  --config experiments/prefix_response_subspaces/configs/paper_smoke.yaml \
  --skip-replication
```

Pilot:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m experiments.prefix_response_subspaces.run_paper_pipeline \
  --config experiments/prefix_response_subspaces/configs/paper_pilot.yaml
```

Full main experiments plus the cheap checkpoint replication:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m experiments.prefix_response_subspaces.run_paper_pipeline \
  --config experiments/prefix_response_subspaces/configs/paper_full.yaml
```

Faster claim-equivalent Full profile:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m experiments.prefix_response_subspaces.run_paper_pipeline \
  --config experiments/prefix_response_subspaces/configs/paper_full_fast.yaml
```

The additional experiments above are enabled by the pipeline without changing
the resolved paper config. This is intentional: editing a completed run's YAML
would invalidate its stored config hash. To retain the earlier two-experiment
suite only, pass `--skip-additional-experiments`.

For an already completed `full_fast_v1` main run, resume directly at the first
new post-hoc stage. Existing geometry and Functional manifests are reused:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m experiments.prefix_response_subspaces.run_paper_pipeline \
  --config experiments/prefix_response_subspaces/configs/paper_full_fast.yaml \
  --from-stage analyze_rank_saturation \
  --skip-replication
```

Before the new model-forward stage, validate the Qwen first-layer module
contract on one prefix. The completed prefix is checkpointed and the full run
resumes after it:

```bash
python3 -m experiments.prefix_response_subspaces.extract_first_layer_mechanism \
  --config experiments/prefix_response_subspaces/configs/paper_full_fast.yaml \
  --preflight-only
```

The stages run in this order:

1. `analyze_rank_saturation` (saved responses only)
2. `analyze_shared_backbone` (saved residuals only)
3. `extract_first_layer_mechanism` (new model forward; resumable by stage)
4. `analyze_first_layer_mechanism` (saved mechanism states only)
5. `analyze_paper_functional` (skips when the completed manifest matches)
6. `analyze_value_space_functional` (one new value-space condition per cell,
   with JSONL batch checkpoints)

Run `run_paper_replication` separately, or omit `--skip-replication`, for the
general Qwen checkpoint. Replication uses decoder layer 0 and rank 64 fixed in
advance and records `confirmatory_no_reselection=true`; layer search is not
used for its confirmatory estimate.

This keeps the 256 evaluation prefixes, 128 development prefixes, 128 Global
training prefixes, 320 candidates, five Wrong-prefix controls, four folds, and
500 permutations. It reduces only the otherwise-unused donor reservoir
(5000 to 1536 total prefixes), computes full-vocabulary logits only for the
512 candidate/dev/test prefixes, shares each clean functional forward across
all controls for the same prefix/token cell, and uses a larger functional
batch. When trajectories are missing, it uses deterministic greedy decoding
with a KV cache (batch 8 on the primary GPU) and prints processed problems,
throughput, and ETA after every batch. Results go to
`results/prefix_response_paper/full_fast_v1` so an
existing `full_v1` run remains resumable.

For local checkpoints, `--model-path` addresses the Math checkpoint and
`--replication-model-path` independently addresses the general checkpoint.

Before a long Functional run, the online Oracle path can be checked on one
batch. The batch is checkpointed and the full run resumes after it:

```bash
python3 -m experiments.prefix_response_subspaces.analyze_paper_functional \
  --config experiments/prefix_response_subspaces/configs/paper_full_fast.yaml \
  --preflight-only
```

Functional rows are appended under `functional/checkpoints/` after every
batch. A restart with the same config/model resumes from the saved row count;
`--force` intentionally discards those checkpoints. Every batch is rejected
immediately if its exact online Oracle exceeds `functional.oracle_tolerance`.

To add the fair-Global and rotation analyses to an already completed run under
the **same paper config and `results_root`**, reuse its hidden states and
residuals with:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m experiments.prefix_response_subspaces.run_paper_pipeline \
  --config experiments/prefix_response_subspaces/configs/paper_full.yaml \
  --recompute-analysis
```

This starts at geometry and overwrites only derived geometry, functional,
figure, table, and summary artifacts. It does not regenerate trajectories,
candidate tokens, or successor branches. Functional reconstruction still
executes the downstream continuation from the stored branch states.
It is not a first-run command and does not automatically mix older Matched-run
artifacts from `results/prefix_response_subspaces/*` with the paper artifacts.
For a first run, use the Full command above without `--recompute-analysis`.

The final artifacts are `paper_summary.md`, `paper_gate_results.json`,
`tables/paper_results.md`, and the four figure-data groups under `figures/`.
Every intermediate stage is resumable and can be selected with `--from-stage`
or `--through-stage`.

Additional outputs are:

- `metrics/rank_saturation_summary.json`
- `metrics/shared_backbone_summary.json`
- `metrics/first_layer_mechanism_summary.json`
- `functional/value_space_summary.json`

The compactness decision uses two predeclared diagnostics: median `r90 <= 64`
relative to rank 127, and mean relative EV gain from rank 64 to 127 no larger
than 5%. The shared skeleton is fit on `analysis_dev` only; a shared direction
has mean-projector eigenvalue at least 0.5. The orthogonal Local-vs-Wrong test is
then evaluated only on `analysis_test`, avoiding test-set estimation of the
shared basis.

For a geometry run completed by an older version of the code, refresh the
Top-128/256 audit and conditioned-Global coverage without rerunning rank curves
or permutations:

```bash
python3 -m experiments.prefix_response_subspaces.refresh_paper_geometry \
  --config experiments/prefix_response_subspaces/configs/paper_full_fast.yaml
```

This reads the saved selected-layer residuals. It pools projection numerator
and denominator across the four held-out folds for each prefix and only then
applies `high_probability_min_tokens`. Each candidate remains evaluated only
in its held-out fold. The command also records the one permitted sensitivity
fallback: nearest length bin within the same reasoning-progress bin. Exact-bin
results remain primary and require at least 99% prefix coverage.

If the predeclared minimum is not met for every prefix, the Top-k result is
reported only for the eligible subset, with its numerator and denominator made
explicit. It does not block the full-candidate geometry or functional
experiments and must not be described as generalizing to every evaluation
prefix. The summary also records minimum-token sensitivities at 1, 4, 8, and
the configured primary threshold; these are diagnostics, not replacements for
the predeclared threshold.

## Primary inferential unit and selection

All splits and bootstrap confidence intervals use original GSM8K problem IDs.
The layer is selected on development prefixes by the predeclared maximum
minimum-control-delta rule at rank 16. The primary rank is then the smallest
configured rank reaching 90% of development rank-64 EV. The test split never
changes either choice. The replication run reuses this rank and normalized
depth without reselection.

The conditioned Global basis is not pooled PCA. For each training prefix its
response covariance is divided by its trace, these normalized covariances are
averaged within the fixed length/progress bin, and the top eigenspace is used.
At the selected layer/rank, `metrics/paper_rotation_rows.csv` reports
`1 - ||U_i^T U_k||_F^2/r`, plus deterministic train-token split-half
`R_within`, `R_between`, and their paired difference. No additional model
forward is needed for these diagnostics.

## Final rank-64 checks

These stages reuse the completed main branching run. They extract the full
per-prefix value/output span, fit the training-EV-optimal rank-64 basis inside
it, evaluate its Functional recovery, and deterministically recompute the
eight largest Rank-0 duplicate-anchor outliers in FP32:

```bash
CONFIG=experiments/prefix_response_subspaces/configs/paper_full_fast.yaml
python3 -m experiments.prefix_response_subspaces.run_paper_pipeline \
  --config "$CONFIG" \
  --from-stage extract_value_output_spans \
  --through-stage verify_rank0_outliers \
  --skip-replication
```

A value span equal to the full hidden space is recorded explicitly and makes
the optimal value control identical to Local by construction; such cells are
not counted as evidence that Local beats the control.

The fixed-condition cross-model run uses decoder block 0 and rank 64 without
reselection. Prefixes are retokenized, and candidate tokens, length bins, and
Wrong controls are rebuilt independently for each tokenizer:

```bash
python3 -m experiments.prefix_response_subspaces.run_paper_replication \
  --config "$CONFIG"
```

Llama 3.2 may require prior Hugging Face checkpoint access. Local checkpoint
directories can be supplied with repeated `--model-path NAME=PATH`, using the
names `qwen25_15b` and `llama32_3b`.

Qwen3 is an optional fixed-condition replication.  The large-model checkpoint
is the pretraining-only 8B model.  The 4B Base checkpoint remains available as
a closer scale match to Llama-3.2-3B, and the 1.7B Base checkpoint as a size
match to Qwen2.5-1.5B.  All reuse fixed decoder block 0 and rank 64 while
rebuilding candidates and controls under their own Qwen3 tokenizer:

```bash
python3 -m experiments.prefix_response_subspaces.run_paper_replication \
  --config experiments/prefix_response_subspaces/configs/paper_full_fast.yaml \
  --only qwen3_8b_base
```

The Qwen3 model implementation requires `transformers>=4.51.0`.  Verify the
server version before starting.  With two visible A100s, the existing
single-process `torch.nn.DataParallel` path uses both devices; do not launch
the same replication separately on each GPU.

The new summaries are `metrics/optimal_value_control_summary.json`,
`functional/optimal_value_summary.json`,
`functional/rank0_outlier_fp32_summary.json`, and
`fixed_replications/summary.json`.

## Reviewer-scale refresh without new model forwards

After a completed run, refresh the absolute Functional scale directly from
the saved cell summary.  This adds `D_rank0`, all reconstructed distances, and
problem-bootstrap confidence intervals for the ratio-of-totals recovery
fractions.  Do not use `--force` for this step: `--summary-only` deliberately
does not load the model or rewrite Functional checkpoints.

```bash
python3 -m experiments.prefix_response_subspaces.analyze_paper_functional \
  --config experiments/prefix_response_subspaces/configs/paper_full_fast.yaml \
  --summary-only
```

The complete value-span retention diagnostic is a CPU-only replay over saved
interaction states and value-span bases.  It reports, at each first-block
site, both the fraction of held-out interaction energy inside the complete
value span and the complementary fraction outside it.

```bash
python3 -m experiments.prefix_response_subspaces.analyze_optimal_value_control \
  --config experiments/prefix_response_subspaces/configs/paper_full_fast.yaml \
  --force
```

Finally regenerate the consolidated JSON and Markdown table:

```bash
python3 -m experiments.prefix_response_subspaces.make_paper_tables \
  --config experiments/prefix_response_subspaces/configs/paper_full_fast.yaml
```

## Post-review rank, centering, and maximum-rank controls

These analyses are separate from the frozen confirmatory pipeline and do not
overwrite `paper_geometry_summary.json` or `paper_summary.json`.  Use the
dedicated config so the completed main run keeps its original config hash.

First run the CPU-only shared batch.  It (i) selects one global matched-common
rank and one global wrong-context rank on exact-bin development problems only by
searching every integer rank from 64 through 191, (ii) evaluates the same-rank
target-versus-control curves at ranks 8--127, and (iii) recomputes rank-64 EV
with evaluation-fold-independent target-context centering from the saved
successor-state memmap.  It also records the fraction of this sensitivity
residual's energy carried by its candidate-constant row mean.  No model
forward is performed.

The stage saves fold-local rank-191 sample-space SVD caches and rank-191 common
bases.  The Functional stage reconstructs only the selected hidden-space
bases from this cache instead of repeating the decompositions.

The upper rank is 191 because each primary fold has 192 fit candidates and
centering reduces the algebraic rank ceiling by one.

```bash
python3 -m experiments.prefix_response_subspaces.analyze_control_rank_sensitivity \
  --config experiments/prefix_response_subspaces/configs/paper_additional_experiments.yaml
```

After a code or reporting update, refresh exact-bin summaries, paired
centering differences, relative gaps, and their confidence intervals without
recomputing the SVDs:

```bash
python3 -m experiments.prefix_response_subspaces.analyze_control_rank_sensitivity \
  --config experiments/prefix_response_subspaces/configs/paper_additional_experiments.yaml \
  --summary-only
```

The fixed-replication configs written by `run_paper_replication` can be passed
directly.  Run every root that has completed residual and successor-state
artifacts:

```bash
for name in qwen25_15b llama32_3b qwen3_8b_base; do
  python3 -m experiments.prefix_response_subspaces.analyze_control_rank_sensitivity \
    --config "results/prefix_response_paper/full_fast_v1/fixed_replications/$name/replication_config.json"
done
```

The current four-model results selected the ceiling rank 191 for both controls
and still left substantial evaluation EV gaps.  Therefore these controls are
not EV-matched, and `analyze_ev_matched_functional` must not be used for the
current manuscript.  It is retained only for a future run in which a
pre-specified matching tolerance is actually achieved.

If that condition is met in a future experiment, the stage reuses the saved
Oracle, Rank-0, and target-context rank-64 distances and runs only the selected
matched-common and five wrong-context conditions:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
python3 -m experiments.prefix_response_subspaces.analyze_ev_matched_functional \
  --config experiments/prefix_response_subspaces/configs/paper_additional_experiments.yaml \
  --preflight-only

CUDA_VISIBLE_DEVICES=0,1 \
python3 -m experiments.prefix_response_subspaces.analyze_ev_matched_functional \
  --config experiments/prefix_response_subspaces/configs/paper_additional_experiments.yaml
```

The principal outputs are:

- `metrics/ev_matched_rank_selection.json`: exact-bin development-only ranks,
  the EV match actually achieved on exact-bin evaluation problems, absolute
  and target-rank-64-normalized gaps, and paired problem-bootstrap intervals.
- `metrics/control_rank_sensitivity_summary.json`: same-rank delta curves.
- `metrics/inductive_centering_summary.json`: rank-64
  evaluation-fold-independent target-context centering sensitivity, plus the
  mean-shift energy fraction `rho` and paired differences from the primary
  centering estimator.
- `functional/ev_matched_summary.json`: JSD, additive gains, recovery
  fractions, and paired problem-bootstrap target advantages.  Advantages are
  defined as `D_control - D_target`, so positive values favor the target
  context.  Five wrong contexts are averaged within each cell before the
  problem-level bootstrap.  This difference is algebraically identical to
  `G_target - G_control` under the shared additive baseline and is treated as
  one inferential contrast, not two independent pieces of evidence.

The rank figure deliberately retains the existing random 128/128 saturation
analysis in panel A.  Panel B uses the primary 192-fit/64-heldout four-fold
protocol; the generated figure footer states this protocol difference.
