#!/usr/bin/env bash
set -euo pipefail

# Standard public reproduction: main Qwen2.5-Math plus the Qwen2.5 and Llama
# fixed-condition replications. Qwen3 is intentionally not included.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

CONFIG="${CONFIG:-experiments/prefix_response_subspaces/configs/paper_full_fast.yaml}"
ADDITIONAL_CONFIG="${ADDITIONAL_CONFIG:-experiments/prefix_response_subspaces/configs/paper_additional_experiments.yaml}"

main_model_args=()
source_model_args=()
replication_model_args=()

# Local checkpoint paths are optional. With no variables set, Transformers
# resolves the public Hugging Face checkpoints named in the configuration.
if [[ -n "${QWEN_MATH_PATH:-}" ]]; then
  main_model_args+=(--model-path "$QWEN_MATH_PATH")
  source_model_args+=(--source-model-path "$QWEN_MATH_PATH")
fi
if [[ -n "${QWEN_BASE_PATH:-}" ]]; then
  replication_model_args+=(--model-path "qwen25_15b=$QWEN_BASE_PATH")
fi
if [[ -n "${LLAMA_PATH:-}" ]]; then
  replication_model_args+=(--model-path "llama32_3b=$LLAMA_PATH")
fi

python3 -m experiments.prefix_response_subspaces.run_paper_pipeline \
  --config "$CONFIG" \
  "${main_model_args[@]}" \
  --skip-replication

python3 -m experiments.prefix_response_subspaces.run_paper_replication \
  --config "$CONFIG" \
  "${source_model_args[@]}" \
  "${replication_model_args[@]}" \
  --only qwen25_15b \
  --only llama32_3b

# These analyses consume saved states and add no model forward.
python3 -m experiments.prefix_response_subspaces.analyze_control_rank_sensitivity \
  --config "$ADDITIONAL_CONFIG"

for name in qwen25_15b llama32_3b; do
  replication_config="results/prefix_response_paper/full_fast_v1/fixed_replications/$name/replication_config.json"
  python3 -m experiments.prefix_response_subspaces.analyze_control_rank_sensitivity \
    --config "$replication_config"
done
