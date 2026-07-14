from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

from .src.data import assign_groups, choose_stratified_position, eligible_positions, trajectory_tokens
from .src.utils import atomic_json, atomic_jsonl, ensure_layout, file_sha256, git_commit, load_config, read_jsonl, stable_hash, stage_is_complete


def _unique_problem_count(path: Path) -> int:
    if not path.is_file():
        return 0
    return len({str(row["problem_id"]) for row in read_jsonl(path)})


def _ensure_trajectory_pool(config: dict, model_path: str | None) -> Path:
    data = config["data"]
    source = Path(data["trajectories_jsonl"])
    target = int(data["prefix_pool_size"])
    available = _unique_problem_count(source)
    if available >= target:
        return source
    generation_config = data.get("trajectory_generation_config")
    if not generation_config:
        raise ValueError(
            f"need {target} unique problems, found {available}; "
            "data.trajectory_generation_config is not configured"
        )
    generation = load_config(generation_config)
    generated_path = Path(generation["output"]["trajectories_jsonl"])
    generated_available = _unique_problem_count(generated_path)
    if generated_available >= target:
        return generated_path
    from scripts import prepare_gsm8k_trajectories
    previous_argv = sys.argv
    sys.argv = [
        "prepare_gsm8k_trajectories.py", "--config", str(generation_config),
        *(["--model-path", model_path] if model_path else []),
    ]
    try:
        prepare_gsm8k_trajectories.main()
    finally:
        sys.argv = previous_argv
    source = generated_path
    available = _unique_problem_count(source)
    if available < target:
        raise ValueError(
            f"trajectory generation completed but only produced {available} unique "
            f"problems; {target} are required"
        )
    return source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path")
    args = parser.parse_args()
    config = load_config(args.config)
    root = ensure_layout(config)
    manifest_path = root / "manifests/prefix_pool.json"
    source = _ensure_trajectory_pool(config, args.model_path)
    source_hash = file_sha256(source)
    if stage_is_complete(manifest_path, config, {"trajectory_sha256": source_hash}):
        print(manifest_path); return
    trajectories = read_jsonl(source)
    by_problem: dict[str, dict] = {}
    for row in trajectories:
        by_problem.setdefault(str(row["problem_id"]), row)
    target = int(config["data"]["prefix_pool_size"])
    if len(by_problem) < target:
        raise ValueError(f"need {target} eligible unique problems, found {len(by_problem)}")
    eligible, exclusions = [], []
    for problem_id, row in by_problem.items():
        if eligible_positions(row, int(config["data"]["min_prefix_tokens"]), int(config["data"]["min_remaining_tokens"])):
            eligible.append(problem_id)
        else:
            exclusions.append({"problem_id": problem_id, "reason": "no eligible prefix position"})
    random.Random(int(config["seed"])).shuffle(eligible)
    selected = sorted(eligible[:target])
    if len(selected) < target:
        raise ValueError(f"need {target} eligible unique problems, found {len(selected)}")
    groups = assign_groups(selected, config, int(config["seed"]))
    records = []
    for rank, problem_id in enumerate(selected):
        row = by_problem[problem_id]
        try:
            position, progress_bin, progress = choose_stratified_position(row, rank, strata=int(config["data"]["position_strata"]), min_prefix=int(config["data"]["min_prefix_tokens"]), min_remaining=int(config["data"]["min_remaining_tokens"]), seed=int(config["seed"]))
        except ValueError as exc:
            exclusions.append({"problem_id": problem_id, "reason": str(exc)}); continue
        tokens = trajectory_tokens(row)
        prefix = tokens[:position + 1]
        records.append({
            "problem_id": problem_id, "prefix_id": f"{problem_id}/p0", "prefix_token_ids": prefix,
            "prefix_length": len(prefix), "prefix_position_fraction": progress,
            "reasoning_progress_bin": progress_bin, "last_token_id": prefix[-1],
            "correct_or_incorrect": bool(row.get("correctness", False)), "problem_group": groups[problem_id],
            "trajectory_id": str(row.get("trajectory_id", problem_id)),
            "evaluation_suffix_token_ids": tokens[position + 1:],
            "positive_token_id": row.get("positive_token_id"), "negative_token_id": row.get("negative_token_id"),
        })
    if len(records) != target:
        raise ValueError(f"only {len(records)} of {target} selected trajectories are eligible; exclusions saved in error context: {exclusions[:5]}")
    lengths = sorted(row["prefix_length"] for row in records)
    for row in records:
        row["prefix_length_bin"] = min(int(config["data"]["length_bins"]) - 1, sum(value <= row["prefix_length"] for value in lengths) * int(config["data"]["length_bins"]) // (len(lengths) + 1))
    output = root / "prefix_pool/prefixes.jsonl"
    atomic_jsonl(output, records)
    counts = defaultdict(int)
    for row in records: counts[row["problem_group"]] += 1
    atomic_json(manifest_path, {"complete": True, "config_hash": stable_hash(config), "trajectory_sha256": source_hash, "prefixes": str(output), "prefixes_sha256": file_sha256(output), "problem_split_unit": "problem_id", "counts": dict(counts), "git_commit": git_commit()})
    print(manifest_path)


if __name__ == "__main__": main()
