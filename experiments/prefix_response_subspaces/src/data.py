from __future__ import annotations

import math
import random
from typing import Any


def trajectory_tokens(row: dict[str, Any]) -> list[int]:
    values = row.get("input_ids")
    if not isinstance(values, list) or not values or not all(isinstance(x, int) for x in values):
        raise ValueError("trajectory must contain non-empty integer input_ids")
    return values


def eligible_positions(row: dict[str, Any], min_prefix: int, min_remaining: int) -> list[int]:
    tokens = trajectory_tokens(row)
    positions = row.get("transition_positions")
    if not isinstance(positions, list):
        start = int(row.get("generated_start_position", min_prefix - 1))
        positions = list(range(start, len(tokens) - 1))
    result = sorted({int(position) for position in positions if int(position) + 1 >= min_prefix and len(tokens) - int(position) - 1 >= min_remaining})
    return result


def choose_stratified_position(row: dict[str, Any], problem_rank: int, *, strata: int, min_prefix: int, min_remaining: int, seed: int) -> tuple[int, int, float]:
    positions = eligible_positions(row, min_prefix, min_remaining)
    if not positions:
        raise ValueError("trajectory has no eligible prefix position")
    stratum = problem_rank % strata
    chunks = [positions[math.floor(len(positions) * s / strata):math.floor(len(positions) * (s + 1) / strata)] for s in range(strata)]
    choices = chunks[stratum] or positions
    position = random.Random(seed + problem_rank * 1009).choice(choices)
    progress = positions.index(position) / max(1, len(positions) - 1)
    return position, stratum, progress


def assign_groups(problem_ids: list[str], config: dict[str, Any], seed: int) -> dict[str, str]:
    data = config["data"]
    requested = {
        "candidate_selection": int(data["candidate_selection_prefixes"]),
        "auxiliary": int(data["auxiliary_prefixes"]),
        "analysis_dev": int(data["analysis_dev_prefixes"]),
        "analysis_test": int(data["evaluation_prefixes"]),
        "analysis_train": int(data["analysis_train_prefixes"]),
    }
    requested["matching_pool"] = len(problem_ids) - sum(requested.values())
    if requested["matching_pool"] < 0:
        raise ValueError("configured prefix groups exceed prefix_pool_size")
    shuffled = sorted(problem_ids)
    random.Random(seed).shuffle(shuffled)
    result, offset = {}, 0
    for group in ("candidate_selection", "auxiliary", "analysis_dev", "analysis_test", "analysis_train", "matching_pool"):
        count = requested[group]
        for problem_id in shuffled[offset:offset + count]:
            result[problem_id] = group
        offset += count
    return result


def pad_token_rows(rows: list[list[int]], pad_id: int):
    import torch
    maximum = max(map(len, rows))
    input_ids = torch.full((len(rows), maximum), int(pad_id), dtype=torch.long)
    mask = torch.zeros_like(input_ids)
    positions = torch.empty(len(rows), dtype=torch.long)
    for index, values in enumerate(rows):
        input_ids[index, :len(values)] = torch.tensor(values)
        mask[index, :len(values)] = 1
        positions[index] = len(values) - 1
    return input_ids, mask, positions
