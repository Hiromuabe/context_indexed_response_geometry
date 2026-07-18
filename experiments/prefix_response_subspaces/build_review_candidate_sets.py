from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .src.review_experiments import (
    deterministic_context_split,
    distribution_groups,
    load_review_tokenizer,
    review_token_category,
    review_roots,
    select_candidates_from_logits,
)
from .src.utils import atomic_json, file_sha256, load_config, read_json, read_jsonl, stable_hash, stage_is_complete


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    source_root, root = review_roots(config)
    review = config.get("review_experiments", {})
    logits_path = source_root / "prefix_pool/next_token_logits.npy"
    candidate_manifest_path = source_root / "manifests/candidate_tokens.json"
    candidate_path = source_root / "candidate_tokens/candidate_tokens.json"
    prefix_path = source_root / "prefix_pool/prefixes.jsonl"
    inputs = {
        "candidate_manifest_sha256": file_sha256(candidate_manifest_path),
        "candidate_tokens_sha256": file_sha256(candidate_path),
        "prefixes_sha256": file_sha256(prefix_path),
        "logits_sha256": file_sha256(logits_path),
    }
    manifest_path = root / "manifests/review_candidate_sets.json"
    if not args.force and stage_is_complete(manifest_path, config, inputs):
        print(manifest_path)
        return
    tokenizer, tokenizer_info = load_review_tokenizer(config, args.model_path, source_root=source_root)
    manifest = read_json(candidate_manifest_path)
    prefixes = read_jsonl(prefix_path)
    prefix_by_id = {str(row["prefix_id"]): row for row in prefixes}
    axis_ids = list(map(str, manifest.get("prefix_axis_ids", [row["prefix_id"] for row in prefixes])))
    selection_ids = [prefix_id for prefix_id in axis_ids if prefix_by_id[prefix_id]["problem_group"] == "candidate_selection"]
    split_a, split_b = deterministic_context_split(selection_ids, int(config["seed"]) + 17011)
    context_limit = review.get("candidate_selection_contexts_per_half")
    if context_limit is not None:
        split_a = split_a[: int(context_limit)]
        split_b = split_b[: int(context_limit)]
    axis = {prefix_id: index for index, prefix_id in enumerate(axis_ids)}
    logits = np.load(logits_path, mmap_mode="r")
    set_size = int(review.get("independent_candidate_set_size", 192))
    proposal_top_k = int(config["candidates"]["proposal_top_k"])
    selected: dict[str, list[dict]] = {}
    for name, ids in (("selection_A", split_a), ("selection_B", split_b)):
        started = time.monotonic()
        print(
            f"[review_candidates] START set={name} contexts={len(ids)} "
            f"top_k={proposal_top_k} requested={set_size}",
            flush=True,
        )
        load_started = time.monotonic()
        logit_block = np.asarray(logits[[axis[prefix_id] for prefix_id in ids]], dtype=np.float32)
        print(
            f"[review_candidates] PHASE set={name} phase=load_logits shape={list(logit_block.shape)} "
            f"elapsed={time.monotonic()-load_started:.2f}s",
            flush=True,
        )
        selected[name] = select_candidates_from_logits(
            logit_block,
            tokenizer=tokenizer, total=set_size, proposal_top_k=proposal_top_k, progress_label=name,
        )
        del logit_block
        print(
            f"[review_candidates] DONE set={name} selected={len(selected[name])} "
            f"elapsed={time.monotonic()-started:.2f}s",
            flush=True,
        )
    original = read_json(candidate_path)
    original_rows = []
    for row in original["candidate_tokens"]:
        clean = dict(row)
        clean["review_category"] = review_token_category(str(row["text"]))
        original_rows.append(clean)
    union: dict[int, dict] = {int(row["token_id"]): dict(row) for row in original_rows}
    membership: dict[int, set[str]] = {token_id: {"original"} for token_id in union}
    for name, rows in selected.items():
        for row in rows:
            token_id = int(row["token_id"])
            union.setdefault(token_id, dict(row))
            membership.setdefault(token_id, set()).add(name)
    union_rows = []
    for token_id in sorted(union):
        row = dict(union[token_id])
        row["membership"] = sorted(membership[token_id])
        union_rows.append(row)
    a_ids = [int(row["token_id"]) for row in selected["selection_A"]]
    b_ids = [int(row["token_id"]) for row in selected["selection_B"]]
    groups = distribution_groups(original_rows)
    groups.update({
        "independent_A": a_ids,
        "independent_B": b_ids,
        "independent_A_exclusive": sorted(set(a_ids) - set(b_ids)),
        "independent_B_exclusive": sorted(set(b_ids) - set(a_ids)),
    })
    payload = {
        "candidate_rows": union_rows,
        "candidate_token_ids": [int(row["token_id"]) for row in union_rows],
        "groups": groups,
        "selection_contexts": {"selection_A": split_a, "selection_B": split_b},
        "diagnostics": {
            "set_size": set_size, "union_size": len(union_rows), "overlap": len(set(a_ids) & set(b_ids)),
            "A_exclusive": len(set(a_ids) - set(b_ids)), "B_exclusive": len(set(b_ids) - set(a_ids)),
            "tokenizer_source": tokenizer_info["source"],
            "tokenizer_resolution": tokenizer_info,
            "candidate_selection_contexts_are_disjoint": not bool(set(split_a) & set(split_b)),
        },
    }
    output = root / "candidate_tokens/review_candidate_sets.json"
    atomic_json(output, payload)
    atomic_json(manifest_path, {
        "complete": True, "config_hash": stable_hash(config), **inputs,
        "output": str(output), "output_sha256": file_sha256(output),
        "candidate_set_hash": stable_hash(payload["candidate_token_ids"]),
    })
    print(manifest_path)


if __name__ == "__main__":
    main()
