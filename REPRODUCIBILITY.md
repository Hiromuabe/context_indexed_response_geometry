# Reproducibility record

## Experimental design encoded by the pipeline

- Dataset: GSM8K `main`, train split; one sampled prefix per original problem.
- Candidate inputs: 320 shared tokens, with disjoint calibration and analysis
  partitions; analysis uses four 192-fit/64-held-out folds.
- Statistical unit: original GSM8K problem ID.
- Controls: a development-conditioned common space and five wrong-context
  donors. Wrong-donor metrics are averaged within a problem before bootstrap.
- Selection: layer and rank are selected on development data only. Fixed-model
  replications use decoder block 0 and rank 64 without reselection.
- Functional analysis: reconstructed activations are reintroduced and compared
  by Jensen--Shannon distance to the clean output distribution.

The resolved configuration and manifests written by every stage are the
authoritative record. A run is not reproducible from a table of aggregate
numbers alone; retain the manifests, candidate-token file, prefix split, and
wrong-context assignments with the published artifact archive.

## Standard public reproduction environment

- Qwen2.5-Math-1.5B, Qwen2.5-1.5B, and Llama-3.2-3B run on two NVIDIA A100
  80GB GPUs. These three checkpoints form the standard public suite.

Qwen3-8B-Base is an optional large-model replication, not a requirement of the
standard entry point. The reported Qwen3 run used eight NVIDIA A100 GPUs with
`torch.nn.DataParallel` and bfloat16; hidden size 4096 and 36 decoder blocks.

The code records device IDs, precision, parallelism, hidden size, decoder
depth, platform, and resolved model metadata in its manifests. A100 memory
capacity and the exact CUDA/driver/library versions must be taken from the
server-generated `environment.txt`; they are not inferable from source code.
`environment.txt` records the standard two-A100 server. An eight-GPU
environment record is needed only when distributing the optional Qwen3 run.

## Checkpoint revisions currently verified

- `Qwen/Qwen2.5-Math-1.5B`:
  `4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2`
- `Qwen/Qwen3-8B-Base`:
  `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`

The exact resolved revisions for Qwen2.5-1.5B and Llama-3.2-3B must be copied
from their `manifests/hidden_states.json` files before the release tag. The
checked-in configurations retain `main` because they mirror the executed
working configuration; for a frozen replication release, make pinned copies.

## Artifact flow

1. `scripts/prepare_gsm8k_trajectories.py` generates deterministic greedy
   reasoning trajectories when the configured JSONL is absent.
2. `build_prefix_pool` fixes problem-level partitions and prefix positions.
3. `build_candidate_tokens` fixes shared candidate inputs and analysis folds.
4. `extract_successor_states` records the branched hidden states.
5. `compute_contrast_residuals` performs double centering.
6. Geometry, mechanism, rank, and functional modules consume these immutable
   artifacts and write summaries under the configured `results_root`.

Do not combine outputs from different roots unless their manifest hashes,
prefix axes, token axes, model revision, and configuration hashes agree.
