from __future__ import annotations

import argparse

from .src.review_extraction import extract_endpoint_grid
from .src.review_experiments import review_roots
from .src.utils import atomic_json, ensure_layout, file_sha256, load_config, read_json, read_jsonl, stable_hash, stage_is_complete


def balanced_fold_candidate_indices(candidates: dict, limit: int | None) -> list[int]:
    """Keep a small, deterministic evaluation slice from every analysis fold."""
    if limit is None:
        return list(range(len(candidates["candidate_token_ids"])))
    folds = candidates.get("folds", [])
    if not folds:
        return list(range(min(int(limit), len(candidates["candidate_token_ids"]))))
    target = max(len(folds), int(limit))
    quotient, remainder = divmod(target, len(folds))
    selected: list[int] = []
    for fold_number, fold in enumerate(folds):
        count = quotient + (1 if fold_number < remainder else 0)
        selected.extend(sorted(map(int, fold["evaluation_indices"]))[:count])
    return sorted(set(selected))[:target]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    source_root, root = review_roots(config)
    controls_path = root / "controls/review_context_controls.jsonl"
    candidate_path = source_root / "candidate_tokens/candidate_tokens.json"
    hidden_manifest_path = source_root / "manifests/hidden_states.json"
    geometry_path = source_root / "metrics/paper_geometry_summary.json"
    inputs = {
        "implementation_version": "review_grid_v2_precomputed_early_exit",
        "controls_sha256": file_sha256(controls_path),
        "candidate_tokens_sha256": file_sha256(candidate_path),
        "hidden_states_sha256": file_sha256(hidden_manifest_path),
        "paper_geometry_sha256": file_sha256(geometry_path),
    }
    manifest_path = root / "manifests/context_control_states.json"
    if not args.force and stage_is_complete(manifest_path, config, inputs):
        print(manifest_path)
        return
    contexts = read_jsonl(controls_path)
    candidates = read_json(candidate_path)
    candidate_limit = config.get("review_experiments", {}).get("context_control_candidate_limit")
    candidate_indices = balanced_fold_candidate_indices(candidates, candidate_limit)
    token_ids = [int(candidates["candidate_token_ids"][index]) for index in candidate_indices]
    hidden = read_json(hidden_manifest_path)
    geometry = read_json(geometry_path)
    selected_layer = int(config.get("review_experiments", {}).get("context_control_layer", geometry["selected_layer"]))
    source_prefixes = read_jsonl(hidden["prefix_snapshot"])
    source_context_axis = {str(row["prefix_id"]): index for index, row in enumerate(source_prefixes)}
    source_context_rows = {str(row["prefix_id"]): row for row in source_prefixes}
    context_indices = []
    for row in contexts:
        source_id = str(row["source_prefix_id"])
        source_row = source_context_rows.get(source_id)
        exact_copy = source_row is not None and list(map(int, source_row["prefix_token_ids"])) == list(map(int, row["prefix_token_ids"]))
        context_indices.append(source_context_axis[source_id] if exact_copy else -1)
    precomputed = {
        "fingerprint": stable_hash(inputs),
        "context_indices": context_indices,
        "token_indices": candidate_indices,
        "layers": {int(row["layer"]): row["successor_path"] for row in hidden["layers"] if int(row["layer"]) == selected_layer},
    }
    entries, model, reuse = extract_endpoint_grid(
        config, contexts=contexts, token_ids=token_ids,
        output_root=root / "hidden_states/context_controls",
        checkpoint_path=root / "hidden_states/context_controls/checkpoint.json",
        model_path=args.model_path, force=args.force, precomputed=precomputed, target_layers=[selected_layer],
    )
    atomic_json(manifest_path, {
        "complete": True, "config_hash": stable_hash(config), **inputs,
        "contexts": str(controls_path), "candidate_indices": candidate_indices,
        "candidate_token_ids": token_ids,
        "candidate_set_hash": stable_hash(token_ids), "model": model, "layers": entries, "selected_layer": selected_layer,
        "reuse_diagnostics": reuse,
    })
    print(manifest_path)


if __name__ == "__main__":
    main()
