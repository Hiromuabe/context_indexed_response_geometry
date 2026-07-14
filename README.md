# Context-indexed next-token response geometry

This directory is the standalone reproducibility package for the paper's
geometry, mechanism, functional-reinjection, cross-model replication, rank
sensitivity, and evaluation-fold-independent centering experiments.

Large generated artifacts, model weights, GSM8K copies, and result tensors are
not included. The code creates them under `data/` and `results/`, both ignored
by Git.

## Installation

Python 3.10 or newer is required. From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Install a CUDA build of PyTorch appropriate for the target server if the
default wheel is not suitable. Record the final server environment with:

```bash
python scripts/capture_environment.py > environment.txt
```

The checked-in `environment.txt` covers the standard two-GPU suite. A separate
eight-GPU record is needed only for the optional Qwen3 replication.

## Standard two-GPU reproduction

The default public run covers:

- `Qwen/Qwen2.5-Math-1.5B` (main geometry and functional experiments)
- `Qwen/Qwen2.5-1.5B` (fixed block 0/rank 64 replication)
- `meta-llama/Llama-3.2-3B` (fixed block 0/rank 64 replication)

With public Hugging Face checkpoints:

```bash
bash scripts/run_two_gpu_suite.sh
```

Authenticate with Hugging Face first if checkpoint access requires it, or pass
the local paths shown below.

With locally downloaded checkpoints:

```bash
QWEN_MATH_PATH=/path/to/models/Qwen2.5-Math-1.5B \
QWEN_BASE_PATH=/path/to/models/Qwen2.5-1.5B \
LLAMA_PATH=/path/to/models/Llama-3.2-3B \
bash scripts/run_two_gpu_suite.sh
```

It defaults to GPUs `0,1`. The wrapper runs the main experiment, both
fixed-condition replications, and the saved-state rank/centering analyses.

## Individual commands

The paper's main profile is `paper_full_fast.yaml`:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
python -m experiments.prefix_response_subspaces.run_paper_pipeline \
  --config experiments/prefix_response_subspaces/configs/paper_full_fast.yaml \
  --model-path /path/to/Qwen2.5-Math-1.5B \
  --skip-replication
```

Run fixed-condition replications separately so each checkpoint path and GPU
allocation is explicit:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
python -m experiments.prefix_response_subspaces.run_paper_replication \
  --config experiments/prefix_response_subspaces/configs/paper_full_fast.yaml \
  --source-model-path /path/to/Qwen2.5-Math-1.5B \
  --model-path qwen25_15b=/path/to/Qwen2.5-1.5B \
  --model-path llama32_3b=/path/to/Llama-3.2-3B
```

Qwen3-8B is optional and is not run by `run_two_gpu_suite.sh`. Its reported
large-model replication can be regenerated independently on eight GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m experiments.prefix_response_subspaces.run_paper_replication \
  --config experiments/prefix_response_subspaces/configs/paper_full_fast.yaml \
  --source-model-path /path/to/Qwen2.5-Math-1.5B \
  --only qwen3_8b_base \
  --model-path qwen3_8b_base=/path/to/Qwen3-8B-Base
```

Post-review rank and centering analyses reuse saved response tensors and do
not require new model forwards:

```bash
python -m experiments.prefix_response_subspaces.analyze_control_rank_sensitivity \
  --config experiments/prefix_response_subspaces/configs/paper_additional_experiments.yaml
```

See `REPRODUCIBILITY.md` for the artifact flow and
`experiments/prefix_response_subspaces/PAPER_README.md` for stage-level details.

## Tests

```bash
python -m unittest discover -s experiments/prefix_response_subspaces/tests -v
python scripts/validate_release.py
```

## Before making a public repository

Choose and add a software license, replace mutable `revision: main` values with
the exact commits from the run manifests, and archive `environment.txt` with
the tagged release. These items should not be guessed by this code package.
