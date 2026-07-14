from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROBLEM_GROUPS = (
    "auxiliary_prefixes",
    "analysis_train_prefixes",
    "analysis_dev_prefixes",
    "analysis_test_prefixes",
)

_GROUP_ALIASES = {
    "auxiliary": "auxiliary_prefixes",
    "aux": "auxiliary_prefixes",
    "train": "analysis_train_prefixes",
    "analysis_train": "analysis_train_prefixes",
    "dev": "analysis_dev_prefixes",
    "analysis_dev": "analysis_dev_prefixes",
    "test": "analysis_test_prefixes",
    "analysis_test": "analysis_test_prefixes",
}


class DataIntegrityError(ValueError):
    """Raised for malformed trajectories or problem-level leakage."""


_OPTIONAL_TRAJECTORY_METADATA_FIELDS = (
    "positive_token_id",
    "negative_token_id",
    "answer_positive_token_id",
    "answer_negative_token_id",
    "margin_positive_token_id",
    "margin_negative_token_id",
    "positive_token",
    "negative_token",
    "evaluation_position",
    "answer_shared_prefix_ids",
    "generated_answer",
    "operation_structure",
)

_UNKNOWN_TRAJECTORY_METADATA_FIELDS = (
    "reference_answer",
    "reference_answer_text",
    "negative_answer_candidate",
    "generation_strategy",
    "margin_definition",
)


def _trajectory_metadata_for_prefix(row: Mapping[str, Any]) -> dict[str, Any]:
    """Preserve trajectory-level answer-margin provenance on every prefix.

    These fields are not derived from prefix position.  Missing values remain
    explicit rather than being guessed: IDs/lists use ``None`` and descriptive
    strings use ``UNKNOWN``.
    """

    metadata = {
        name: row.get(name, None) for name in _OPTIONAL_TRAJECTORY_METADATA_FIELDS
    }
    metadata.update(
        {
            name: row[name]
            if name in row and row[name] is not None
            else "UNKNOWN"
            for name in _UNKNOWN_TRAJECTORY_METADATA_FIELDS
        }
    )
    return metadata


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataIntegrityError(
                    f"Invalid JSON at {path}:{line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise DataIntegrityError(f"Expected object at {path}:{line_number}")
            yield row


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def _normalize_ratios(ratios: Mapping[str, Any]) -> dict[str, float]:
    normalized = {name: 0.0 for name in PROBLEM_GROUPS}
    for raw_name, raw_value in ratios.items():
        name = _GROUP_ALIASES.get(str(raw_name), str(raw_name))
        if name not in normalized:
            raise DataIntegrityError(f"Unknown problem split group: {raw_name}")
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise DataIntegrityError(f"Split ratio for {raw_name} must be numeric")
        normalized[name] += float(raw_value)
    if any(value < 0 for value in normalized.values()):
        raise DataIntegrityError("Problem split ratios cannot be negative")
    total = sum(normalized.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise DataIntegrityError(f"Problem split ratios must sum to 1, got {total}")
    return normalized


def _group_counts(n_items: int, ratios: Mapping[str, float]) -> dict[str, int]:
    positive = [name for name in PROBLEM_GROUPS if ratios[name] > 0]
    if n_items < len(positive):
        raise DataIntegrityError(
            f"n_problems={n_items} cannot populate all {len(positive)} positive-ratio groups"
        )
    # Apply largest remainder to the configured ratio over the *full* sample.
    # Preallocating one row before multiplying would silently change pilot/full
    # ratios (for example 0.5 of 500 would become 249 rather than 250).
    raw = {name: ratios[name] * n_items for name in PROBLEM_GROUPS}
    floors = {name: math.floor(raw[name]) for name in PROBLEM_GROUPS}
    counts = dict(floors)
    unassigned = n_items - sum(counts.values())
    order = sorted(
        PROBLEM_GROUPS,
        key=lambda name: (-(raw[name] - floors[name]), PROBLEM_GROUPS.index(name)),
    )
    for name in order[:unassigned]:
        counts[name] += 1
    # Extremely small positive ratios can round to zero.  Because all four
    # groups have distinct scientific roles, deterministically transfer one
    # item from the least ratio-distorting eligible donor.
    for empty in (name for name in positive if counts[name] == 0):
        donors = [name for name in positive if counts[name] > 1]
        if not donors:
            raise DataIntegrityError("Unable to populate every positive-ratio split")
        donor = max(
            donors,
            key=lambda name: (
                counts[name] - raw[name],
                counts[name],
                -PROBLEM_GROUPS.index(name),
            ),
        )
        counts[donor] -= 1
        counts[empty] += 1
    if sum(counts.values()) != n_items:
        raise AssertionError("Internal split count error")
    return counts


def split_problem_ids(
    problem_ids: Sequence[str], ratios: Mapping[str, Any], seed: int
) -> dict[str, Any]:
    unique = sorted(set(map(str, problem_ids)))
    if len(unique) != len(problem_ids):
        raise DataIntegrityError("problem_ids must be unique before splitting")
    normalized = _normalize_ratios(ratios)
    shuffled = list(unique)
    random.Random(seed).shuffle(shuffled)
    counts = _group_counts(len(shuffled), normalized)
    groups: dict[str, list[str]] = {}
    offset = 0
    for name in PROBLEM_GROUPS:
        groups[name] = sorted(shuffled[offset : offset + counts[name]])
        offset += counts[name]
    registry = {
        "unit": "original_gsm8k_problem_id",
        "seed": int(seed),
        "ratios": normalized,
        "groups": groups,
        "problem_to_group": {
            problem_id: name
            for name in PROBLEM_GROUPS
            for problem_id in groups[name]
        },
    }
    assert_no_problem_leakage(registry)
    return registry


def assert_no_problem_leakage(
    registry_or_rows: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> None:
    seen: dict[str, str] = {}
    if isinstance(registry_or_rows, Mapping):
        groups = registry_or_rows.get("groups")
        if not isinstance(groups, Mapping):
            raise DataIntegrityError("Split registry is missing groups")
        pairs = (
            (str(problem_id), str(group))
            for group, problem_ids in groups.items()
            for problem_id in problem_ids
        )
    else:
        pairs = (
            (str(row.get("problem_id", "")), str(row.get("problem_group", "")))
            for row in registry_or_rows
        )
    for problem_id, group in pairs:
        if not problem_id or not group:
            raise DataIntegrityError("Rows require problem_id and problem_group")
        prior = seen.setdefault(problem_id, group)
        if prior != group:
            raise DataIntegrityError(
                f"Problem leakage: {problem_id} occurs in both {prior} and {group}"
            )


def select_problem_ids(
    rows: Sequence[Mapping[str, Any]], n_problems: int, seed: int
) -> list[str]:
    available = sorted({str(row.get("problem_id", "")) for row in rows})
    if "" in available:
        raise DataIntegrityError("Every trajectory must have problem_id")
    if n_problems > len(available):
        raise DataIntegrityError(
            f"Requested {n_problems} problems but trajectory artifact has {len(available)}"
        )
    random.Random(seed).shuffle(available)
    return sorted(available[:n_problems])


def _validated_trajectory(row: Mapping[str, Any]) -> tuple[list[int], list[int]]:
    input_ids = row.get("input_ids")
    positions = row.get("transition_positions")
    if not isinstance(input_ids, list) or not input_ids:
        raise DataIntegrityError("Trajectory input_ids must be a non-empty list")
    if not isinstance(positions, list) or not positions:
        raise DataIntegrityError("Trajectory transition_positions must be non-empty")
    ids = [int(value) for value in input_ids]
    pos = [int(value) for value in positions]
    if pos != sorted(set(pos)):
        raise DataIntegrityError("transition_positions must be sorted and unique")
    if pos[0] < 0 or pos[-1] + 1 >= len(ids):
        raise DataIntegrityError("transition_positions contain an invalid adjacent pair")
    return ids, pos


def choose_prefix_positions(
    row: Mapping[str, Any],
    *,
    count: int,
    min_prefix_tokens: int,
    min_remaining_tokens: int,
) -> list[int]:
    input_ids, positions = _validated_trajectory(row)
    last_transition = positions[-1]
    eligible = [
        position
        for position in positions
        if position + 1 >= min_prefix_tokens
        and last_transition - position + 1 >= min_remaining_tokens
    ]
    if len(eligible) < count:
        raise DataIntegrityError(
            f"Trajectory {row.get('trajectory_id', 'UNKNOWN')} has {len(eligible)} eligible "
            f"prefixes, fewer than configured prefixes_per_problem={count}"
        )
    # Interior quantiles avoid both the first possible state and the final state.
    chosen_indices = []
    for slot in range(count):
        fraction = (slot + 1) / (count + 1)
        index = int(round(fraction * (len(eligible) - 1)))
        index = min(max(index, 0), len(eligible) - 1)
        if chosen_indices and index <= chosen_indices[-1]:
            index = chosen_indices[-1] + 1
        chosen_indices.append(index)
    if chosen_indices[-1] >= len(eligible):
        # This is only reachable for very small eligible sets; use evenly spaced
        # integer positions without changing the requested sample count.
        chosen_indices = [
            (slot * (len(eligible) - 1)) // max(count - 1, 1)
            for slot in range(count)
        ]
    return [eligible[index] for index in chosen_indices]


def build_prefix_records(
    trajectory_rows: Sequence[Mapping[str, Any]],
    registry: Mapping[str, Any],
    *,
    prefixes_per_problem: int,
    min_prefix_tokens: int,
    min_remaining_tokens: int,
) -> list[dict[str, Any]]:
    by_problem: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in trajectory_rows:
        by_problem[str(row.get("problem_id", ""))].append(row)
    mapping = registry["problem_to_group"]
    records: list[dict[str, Any]] = []
    for problem_id in sorted(mapping):
        trajectories = sorted(
            by_problem.get(problem_id, []), key=lambda row: str(row.get("trajectory_id", ""))
        )
        if not trajectories:
            raise DataIntegrityError(f"No trajectory for selected problem {problem_id}")
        last_error: Exception | None = None
        for row in trajectories:
            try:
                positions = choose_prefix_positions(
                    row,
                    count=prefixes_per_problem,
                    min_prefix_tokens=min_prefix_tokens,
                    min_remaining_tokens=min_remaining_tokens,
                )
            except DataIntegrityError as exc:
                last_error = exc
                continue
            input_ids, transition_positions = _validated_trajectory(row)
            first = transition_positions[0]
            last = transition_positions[-1]
            trajectory_id = str(row.get("trajectory_id", ""))
            if not trajectory_id:
                raise DataIntegrityError(f"Trajectory for {problem_id} lacks trajectory_id")
            for within_problem_index, endpoint in enumerate(positions):
                denominator = max(last - first, 1)
                evaluation_position = row.get("evaluation_position")
                if (
                    isinstance(evaluation_position, int)
                    and not isinstance(evaluation_position, bool)
                    and endpoint + 1 <= evaluation_position < len(input_ids)
                ):
                    # The forced candidate replaces the naturally observed next
                    # token.  Tokens after that natural token through the saved
                    # answer-evaluation position form an auditable teacher-forced
                    # counterfactual suffix for final-margin gradients.
                    evaluation_suffix_token_ids: list[int] | None = input_ids[
                        endpoint + 2 : evaluation_position + 1
                    ]
                    margin_context = (
                        "forced_candidate_plus_original_teacher_forced_suffix_to_"
                        "saved_answer_evaluation_position"
                    )
                else:
                    evaluation_suffix_token_ids = None
                    margin_context = "UNKNOWN"
                records.append(
                    {
                        "problem_id": problem_id,
                        "problem_group": mapping[problem_id],
                        "trajectory_id": trajectory_id,
                        "prefix_id": f"{trajectory_id}:prefix:{endpoint}",
                        "prefix_index_within_problem": within_problem_index,
                        "prefix_token_ids": input_ids[: endpoint + 1],
                        "prefix_length": endpoint + 1,
                        "prefix_position_fraction": (endpoint - first) / denominator,
                        "absolute_endpoint_position": endpoint,
                        "last_token_id": input_ids[endpoint],
                        "observed_next_token_id": input_ids[endpoint + 1],
                        "evaluation_suffix_token_ids": evaluation_suffix_token_ids,
                        "counterfactual_margin_context": margin_context,
                        "correctness": row.get("correctness"),
                        **_trajectory_metadata_for_prefix(row),
                    }
                )
            break
        else:
            raise DataIntegrityError(
                f"No trajectory for {problem_id} can satisfy prefix selection: {last_error}"
            )
    assert_no_problem_leakage(records)
    expected = len(mapping) * prefixes_per_problem
    if len(records) != expected:
        raise DataIntegrityError(f"Expected {expected} prefixes, built {len(records)}")
    for index, record in enumerate(records):
        record["prefix_index"] = index
    return records


def collate_prefix_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    pad_token_id: int,
    candidate_token_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    from prefix_displacement.schema import require_torch

    torch = require_torch()
    if not rows:
        raise DataIntegrityError("Cannot collate an empty prefix batch")
    max_length = max(len(row["prefix_token_ids"]) for row in rows)
    input_ids = torch.full((len(rows), max_length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    endpoints = torch.empty(len(rows), dtype=torch.long)
    for row_index, row in enumerate(rows):
        ids = torch.tensor(row["prefix_token_ids"], dtype=torch.long)
        input_ids[row_index, : len(ids)] = ids
        attention_mask[row_index, : len(ids)] = 1
        endpoints[row_index] = len(ids) - 1
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "endpoint_positions": endpoints,
        "sample_index": torch.tensor(
            [int(row["prefix_index"]) for row in rows], dtype=torch.long
        ),
        "metadata": list(rows),
    }
    if candidate_token_ids is not None:
        candidates = torch.tensor(candidate_token_ids, dtype=torch.long)
        batch["candidate_token_ids"] = candidates[None, :].expand(len(rows), -1).clone()
    return batch


def collate_branch_cells(
    prefix_rows: Sequence[Mapping[str, Any]],
    candidate_token_ids: Sequence[int],
    cell_indices: Sequence[int],
    *,
    pad_token_id: int,
) -> dict[str, Any]:
    from prefix_displacement.schema import require_torch

    torch = require_torch()
    n_candidates = len(candidate_token_ids)
    if not cell_indices or not n_candidates:
        raise DataIntegrityError("Branch batch and common candidate set must be non-empty")
    decoded: list[tuple[Mapping[str, Any], int]] = []
    for flat_index in cell_indices:
        prefix_offset, candidate_offset = divmod(int(flat_index), n_candidates)
        if not 0 <= prefix_offset < len(prefix_rows):
            raise DataIntegrityError(f"Branch cell index out of range: {flat_index}")
        decoded.append((prefix_rows[prefix_offset], int(candidate_token_ids[candidate_offset])))
    max_length = max(len(row["prefix_token_ids"]) + 1 for row, _ in decoded)
    input_ids = torch.full((len(decoded), max_length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    endpoints = torch.empty(len(decoded), dtype=torch.long)
    for batch_index, (row, candidate_id) in enumerate(decoded):
        ids = list(map(int, row["prefix_token_ids"])) + [candidate_id]
        input_ids[batch_index, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[batch_index, : len(ids)] = 1
        endpoints[batch_index] = len(ids) - 1
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "endpoint_positions": endpoints,
        "sample_index": torch.tensor(list(map(int, cell_indices)), dtype=torch.long),
    }


def probability_strata(log_probabilities: Any) -> Any:
    """Return int8 low/medium/high codes (0/1/2) within each prefix.

    Rank-based thirds are deterministic under ties because candidate index is the
    secondary ordering key.  The function intentionally imports NumPy lazily.
    """

    import numpy as np

    values = np.asarray(log_probabilities)
    if values.ndim != 2 or values.shape[1] < 3:
        raise DataIntegrityError("Probability strata require a [prefix, candidate>=3] matrix")
    if not np.isfinite(values).all():
        raise DataIntegrityError("Candidate log probabilities contain NaN or infinity")
    output = np.empty(values.shape, dtype=np.int8)
    n_candidates = values.shape[1]
    low_end = n_candidates // 3
    medium_end = (2 * n_candidates) // 3
    for row_index, row in enumerate(values):
        # lexsort uses the final key as primary: descending log probability then id.
        order = np.lexsort((np.arange(n_candidates), -row))
        output[row_index, order[:low_end]] = 2
        output[row_index, order[low_end:medium_end]] = 1
        output[row_index, order[medium_end:]] = 0
    return output
