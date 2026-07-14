from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Mapping, Sequence


class MatchingError(ValueError):
    pass


def _match_key(row: Mapping[str, Any], mode: str, position_bin_size: int, surprisal_bin_size: float):
    if mode == "unconditional":
        return ()
    if mode == "same_next_token":
        return (row["next_token_id"],)
    if mode == "same_current_token":
        return (row["current_token_id"],)
    if mode == "same_token_pair":
        return (row["current_token_id"], row["next_token_id"])
    if mode == "position_matched":
        return (int(row["absolute_position"]) // position_bin_size,)
    if mode == "boundary_matched":
        return (row["boundary_class"],)
    if mode == "surprisal_matched":
        return (int(float(row["surprisal"]) // surprisal_bin_size),)
    if mode == "conditional":
        return (
            row["current_token_id"],
            row["next_token_id"],
            int(row["absolute_position"]) // position_bin_size,
            row["boundary_class"],
            int(float(row["surprisal"]) // surprisal_bin_size),
        )
    raise ValueError(f"Unknown matching mode: {mode}")


def build_wrong_basepoint_map(
    rows: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    seed: int,
    position_bin_size: int = 16,
    surprisal_bin_size: float = 1.0,
) -> tuple[list[int | None], dict[str, Any]]:
    """Greedy deterministic matching that always requires a different problem."""
    rng = random.Random(seed)
    groups: dict[Any, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[_match_key(row, mode, position_bin_size, surprisal_bin_size)].append(index)
    mapping: list[int | None] = [None] * len(rows)
    reused = 0
    for index, row in enumerate(rows):
        candidates = [
            candidate
            for candidate in groups[_match_key(row, mode, position_bin_size, surprisal_bin_size)]
            if rows[candidate]["problem_id"] != row["problem_id"]
        ]
        if not candidates:
            continue
        rng.shuffle(candidates)
        mapping[index] = candidates[0]
        reused += int(candidates[0] in {value for value in mapping[:index] if value is not None})
    matched = sum(value is not None for value in mapping)
    diagnostics = {
        "mode": mode,
        "seed": seed,
        "total": len(rows),
        "matched": matched,
        "unmatched": len(rows) - matched,
        "exclusion_rate": (len(rows) - matched) / max(len(rows), 1),
        "reused_donors": reused,
        "requires_different_problem": True,
    }
    return mapping, diagnostics


def validate_wrong_basepoint_map(
    rows: Sequence[Mapping[str, Any]], mapping: Sequence[int | None]
) -> None:
    if len(rows) != len(mapping):
        raise MatchingError("Mapping length mismatch")
    for index, donor in enumerate(mapping):
        if donor is None:
            continue
        if donor == index:
            raise MatchingError("A row received itself as a wrong basepoint")
        if rows[index]["problem_id"] == rows[donor]["problem_id"]:
            raise MatchingError("Wrong basepoint came from the same GSM8K problem")
