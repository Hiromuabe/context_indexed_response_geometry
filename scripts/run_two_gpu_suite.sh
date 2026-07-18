#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

MAIN_CONFIG="${MAIN_CONFIG:-experiments/prefix_response_subspaces/configs/paper_full_fast.yaml}"
ADDITIONAL_CONFIG="${ADDITIONAL_CONFIG:-experiments/prefix_response_subspaces/configs/paper_additional_experiments.yaml}"
EXTERNAL_CONFIG="${EXTERNAL_CONFIG:-experiments/prefix_response_subspaces/configs/commonsenseqa_geometry_confirmatory.yaml}"
EXTERNAL_GENERATION_CONFIG="${EXTERNAL_GENERATION_CONFIG:-experiments/prefix_response_subspaces/configs/trajectory_generation_commonsenseqa_confirmatory.yaml}"

main_model_args=()
source_model_args=()
replication_model_args=()

if [[ -n "${MAIN_MODEL_PATH:-}" ]]; then
  main_model_args+=(--model-path "$MAIN_MODEL_PATH")
  source_model_args+=(--source-model-path "$MAIN_MODEL_PATH")
fi
if [[ -n "${BASE_MODEL_PATH:-}" ]]; then
  replication_model_args+=(--model-path "qwen25_15b=$BASE_MODEL_PATH")
fi
if [[ -n "${LLAMA_MODEL_PATH:-}" ]]; then
  replication_model_args+=(--model-path "llama32_3b=$LLAMA_MODEL_PATH")
fi

python3 -m experiments.prefix_response_subspaces.run_paper_pipeline \
  --config "$MAIN_CONFIG" \
  "${main_model_args[@]}" \
  --skip-replication

python3 -m experiments.prefix_response_subspaces.run_paper_replication \
  --config "$MAIN_CONFIG" \
  "${source_model_args[@]}" \
  "${replication_model_args[@]}" \
  --only qwen25_15b \
  --only llama32_3b

python3 -m experiments.prefix_response_subspaces.analyze_control_rank_sensitivity \
  --config "$ADDITIONAL_CONFIG"

python3 -m experiments.prefix_response_subspaces.analyze_shared_dictionary_controls \
  --config "$MAIN_CONFIG" \
  --data-parallel-device-ids 0 1 \
  --dtype float32

shared_run=$(ls -dt results/prefix_response_paper/full_fast_v1/shared_dictionary_controls/run_* | head -1)
python3 -m experiments.prefix_response_subspaces.prepare_shared_dictionary_artifacts \
  --run-dir "$shared_run"

external_trajectory="data/commonsenseqa_prefix_response_trajectories_confirmatory.jsonl"
if [[ ! -f "$external_trajectory" ]]; then
  python3 scripts/prepare_reasoning_trajectories.py \
    --config "$EXTERNAL_GENERATION_CONFIG" \
    "${main_model_args[@]}"
fi

python3 -m experiments.prefix_response_subspaces.run_paper_pipeline \
  --config "$EXTERNAL_CONFIG" \
  "${main_model_args[@]}" \
  --through-stage analyze_paper_geometry \
  --skip-replication

python3 -m experiments.prefix_response_subspaces.summarize_geometry_controls \
  --config "$EXTERNAL_CONFIG"

python3 scripts/validate_release.py
