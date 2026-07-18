# Context-indexed next-token response geometry

This repository contains the anonymous reproducibility package for the paper's
main GSM8K experiments, shared-direction controls, and CommonsenseQA external
dataset replication.

No model weights, dataset copies, generated trajectories, hidden states, or
result tensors are included.
All generated artifacts are written below `data/` and `results/`, which are
ignored by Git.

## Included experiments

- Context-specific PCA, matched-common, and wrong-context held-out EV.
- Within-context and between-context subspace distance across ranks.
- Functional activation reinjection and mechanism analyses.
- Fixed-condition cross-model replication.
- Pooled PCA, CPC, and non-orthogonal shared-dictionary controls.
- CommonsenseQA replication using the GSM8K split and bootstrap protocol.
- Candidate-transfer, structural-context, Jacobian, and stability diagnostics.

## Installation

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Install a CUDA build of PyTorch appropriate for the target server when the
default wheel is unsuitable.

An explicit local checkpoint can be supplied with `--model-path` or through
`RESPONSE_GEOMETRY_MODEL_PATH`.
Without either override, Transformers resolves the public checkpoint ID in the
selected configuration.

## Main GSM8K experiment

```bash
CUDA_VISIBLE_DEVICES=0,1 \
python3 -m experiments.prefix_response_subspaces.run_paper_pipeline \
  --config experiments/prefix_response_subspaces/configs/paper_full_fast.yaml \
  --skip-replication
```

For a local checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
python3 -m experiments.prefix_response_subspaces.run_paper_pipeline \
  --config experiments/prefix_response_subspaces/configs/paper_full_fast.yaml \
  --model-path /path/to/Qwen2.5-Math-1.5B \
  --skip-replication
```

## Shared-direction controls

The shared models reuse the saved main-run hidden states and residuals.
The command evaluates all configured ranks, dictionary widths, and coherence
penalties with five restarts per fit.

```bash
CUDA_VISIBLE_DEVICES=0,1 \
python3 -m experiments.prefix_response_subspaces.analyze_shared_dictionary_controls \
  --config experiments/prefix_response_subspaces/configs/paper_full_fast.yaml \
  --data-parallel-device-ids 0 1 \
  --dtype float32
```

Audit and package the three reviewer-facing artifacts without retraining:

```bash
RUN=$(ls -dt results/prefix_response_paper/full_fast_v1/shared_dictionary_controls/run_* | head -1)

python3 -m experiments.prefix_response_subspaces.prepare_shared_dictionary_artifacts \
  --run-dir "$RUN"
```

The audit checks the full setting grid, restart failures, maximum-step
termination, CPC orthogonality, dictionary coherence, missing folds, duplicate
rows, and non-finite held-out EV values.

## CommonsenseQA replication

Generate deterministic greedy trajectories:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
python3 scripts/prepare_reasoning_trajectories.py \
  --config experiments/prefix_response_subspaces/configs/trajectory_generation_commonsenseqa_confirmatory.yaml
```

Run only the geometry stages required for the external replication:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
python3 -m experiments.prefix_response_subspaces.run_paper_pipeline \
  --config experiments/prefix_response_subspaces/configs/commonsenseqa_geometry_confirmatory.yaml \
  --through-stage analyze_paper_geometry \
  --skip-replication

python3 -m experiments.prefix_response_subspaces.summarize_geometry_controls \
  --config experiments/prefix_response_subspaces/configs/commonsenseqa_geometry_confirmatory.yaml
```

The summarizer writes the rank-wise Target, Matched-common, Wrong-context,
Within, and Between estimates and the combined GSM8K/CommonsenseQA figures.

## Two-GPU wrapper

The complete public workflow is encoded in:

```bash
bash scripts/run_two_gpu_suite.sh
```

Optional local checkpoint paths are accepted through `MAIN_MODEL_PATH`,
`BASE_MODEL_PATH`, and `LLAMA_MODEL_PATH`.

## Tests and release validation

```bash
python3 -m unittest discover -s tests -v
python3 -m unittest discover -s experiments/prefix_response_subspaces/tests -v
python3 scripts/validate_release.py
```

The validator rejects generated tensors, Python bytecode, credentials, email
addresses, private path markers, identifying project strings, and symlinks
that escape the release directory.

The source package contains no author or affiliation metadata.
Choose a software license before publishing the repository.
