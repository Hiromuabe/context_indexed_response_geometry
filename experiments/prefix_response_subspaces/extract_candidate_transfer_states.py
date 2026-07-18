from __future__ import annotations

import argparse

from .src.review_extraction import extract_endpoint_grid
from .src.review_experiments import review_roots
from .src.utils import atomic_json, ensure_layout, file_sha256, load_config, read_json, read_jsonl, stable_hash, stage_is_complete


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    source_root, root = review_roots(config)
    candidate_path = root / "candidate_tokens/review_candidate_sets.json"
    hidden_manifest_path = source_root / "manifests/hidden_states.json"
    source_candidate_path = source_root / "candidate_tokens/candidate_tokens.json"
    geometry_path = source_root / "metrics/paper_geometry_summary.json"
    inputs = {
        "implementation_version": "review_grid_v2_precomputed_early_exit",
        "candidate_sets_sha256": file_sha256(candidate_path),
        "hidden_states_sha256": file_sha256(hidden_manifest_path),
        "source_candidate_tokens_sha256": file_sha256(source_candidate_path),
        "paper_geometry_sha256": file_sha256(geometry_path),
    }
    manifest_path = root / "manifests/candidate_transfer_states.json"
    if not args.force and stage_is_complete(manifest_path, config, inputs):
        print(manifest_path)
        return
    hidden = read_json(hidden_manifest_path)
    geometry = read_json(geometry_path)
    selected_layer = int(config.get("review_experiments", {}).get("candidate_transfer_layer", geometry["selected_layer"]))
    prefixes = read_jsonl(hidden["prefix_snapshot"])
    review = config.get("review_experiments", {})
    active_groups = set(review.get("candidate_transfer_groups", ["auxiliary", "analysis_dev", "analysis_test"]))
    per_group_limit = review.get("candidate_transfer_max_contexts_per_group")
    group_counts = {group: 0 for group in active_groups}
    contexts = []
    for row in sorted(prefixes, key=lambda item: str(item["prefix_id"])):
        if row["problem_group"] in active_groups:
            group = str(row["problem_group"])
            if per_group_limit is not None and group_counts[group] >= int(per_group_limit):
                continue
            group_counts[group] += 1
            contexts.append({
                "context_id": str(row["prefix_id"]), "problem_id": str(row["problem_id"]),
                "problem_group": str(row["problem_group"]), "prefix_token_ids": list(map(int, row["prefix_token_ids"])),
            })
    candidate_sets = read_json(candidate_path)
    destination_token_ids = list(map(int, candidate_sets["candidate_token_ids"]))
    source_candidates = read_json(source_candidate_path)
    source_token_axis = {int(token_id): index for index, token_id in enumerate(source_candidates["candidate_token_ids"])}
    source_context_axis = {str(row["prefix_id"]): index for index, row in enumerate(prefixes)}
    precomputed = {
        "fingerprint": stable_hash(inputs),
        "context_indices": [source_context_axis.get(row["context_id"], -1) for row in contexts],
        "token_indices": [source_token_axis.get(token_id, -1) for token_id in destination_token_ids],
        "layers": {int(row["layer"]): row["successor_path"] for row in hidden["layers"] if int(row["layer"]) == selected_layer},
    }
    entries, model, reuse = extract_endpoint_grid(
        config, contexts=contexts, token_ids=destination_token_ids,
        output_root=root / "hidden_states/candidate_transfer",
        checkpoint_path=root / "hidden_states/candidate_transfer/checkpoint.json",
        model_path=args.model_path, force=args.force, precomputed=precomputed, target_layers=[selected_layer],
    )
    context_path = root / "hidden_states/candidate_transfer/contexts.json"
    atomic_json(context_path, contexts)
    atomic_json(manifest_path, {
        "complete": True, "config_hash": stable_hash(config), **inputs,
        "contexts": str(context_path), "contexts_sha256": file_sha256(context_path),
        "candidate_token_ids": list(map(int, candidate_sets["candidate_token_ids"])),
        "candidate_set_hash": stable_hash(candidate_sets["candidate_token_ids"]),
        "model": model, "layers": entries, "selected_layer": selected_layer, "reuse_diagnostics": reuse,
    })
    print(manifest_path)


if __name__ == "__main__":
    main()
