from __future__ import annotations

import argparse

from .src.review_experiments import build_context_control_records, load_review_tokenizer, review_roots
from .src.utils import atomic_json, atomic_jsonl, file_sha256, load_config, read_jsonl, stable_hash, stage_is_complete


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    source_root, root = review_roots(config)
    prefix_path = source_root / "prefix_pool/prefixes.jsonl"
    inputs = {"prefixes_sha256": file_sha256(prefix_path)}
    manifest_path = root / "manifests/review_context_controls.json"
    if not args.force and stage_is_complete(manifest_path, config, inputs):
        print(manifest_path)
        return
    tokenizer, tokenizer_info = load_review_tokenizer(config, args.model_path, source_root=source_root)
    review = config.get("review_experiments", {})
    records, diagnostics = build_context_control_records(
        read_jsonl(prefix_path),
        lambda token_ids: tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False),
        seed=int(config["seed"]) + 24001,
        target_groups=tuple(map(str, review.get("context_control_target_groups", ["analysis_test"]))),
        minimum_timepoint_gap=int(review.get("same_problem_timepoint_gap", 4)),
        max_targets=review.get("context_control_max_targets"),
        max_auxiliary=review.get("context_control_auxiliary_contexts"),
    )
    output = root / "controls/review_context_controls.jsonl"
    atomic_jsonl(output, records)
    diagnostics_path = root / "controls/review_context_control_diagnostics.json"
    atomic_json(diagnostics_path, {
        **diagnostics,
        "tokenizer_source": tokenizer_info["source"],
        "tokenizer_resolution": tokenizer_info,
    })
    atomic_json(manifest_path, {
        "complete": True, "config_hash": stable_hash(config), **inputs,
        "controls": str(output), "controls_sha256": file_sha256(output),
        "diagnostics": str(diagnostics_path), "diagnostics_sha256": file_sha256(diagnostics_path),
    })
    print(manifest_path)


if __name__ == "__main__":
    main()
