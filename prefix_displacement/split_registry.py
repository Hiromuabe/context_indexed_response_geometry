from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


SPLIT_NAMES = ("train", "dev", "test")


class SplitLeakageError(ValueError):
    """Raised immediately when problem-level split integrity is violated."""


@dataclass(frozen=True)
class SplitRatios:
    train: float
    dev: float
    test: float

    def validate(self) -> None:
        values = (self.train, self.dev, self.test)
        if any(not 0.0 < value < 1.0 for value in values):
            raise ValueError("All split ratios must be strictly between 0 and 1")
        if abs(sum(values) - 1.0) > 1e-12:
            raise ValueError(f"Split ratios must sum to 1.0, got {sum(values):.17g}")

    def as_dict(self) -> dict[str, float]:
        return {"train": self.train, "dev": self.dev, "test": self.test}


def _normalize_problem_ids(problem_ids: Iterable[str]) -> list[str]:
    normalized = sorted({str(problem_id) for problem_id in problem_ids})
    if not normalized:
        raise ValueError("At least one problem_id is required")
    if any(not problem_id for problem_id in normalized):
        raise ValueError("problem_id must be non-empty")
    return normalized


def _allocate_counts(total: int, ratios: SplitRatios) -> dict[str, int]:
    exact = {
        "train": total * ratios.train,
        "dev": total * ratios.dev,
        "test": total * ratios.test,
    }
    counts = {name: math.floor(value) for name, value in exact.items()}
    remainder = total - sum(counts.values())
    priority = sorted(
        SPLIT_NAMES,
        key=lambda name: (exact[name] - counts[name], -SPLIT_NAMES.index(name)),
        reverse=True,
    )
    for name in priority[:remainder]:
        counts[name] += 1
    return counts


def create_split_registry(
    problem_ids: Iterable[str],
    *,
    ratios: SplitRatios,
    seed: int,
) -> dict[str, Any]:
    """Create one deterministic assignment per original GSM8K problem ID."""
    ratios.validate()
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    ids = _normalize_problem_ids(problem_ids)
    shuffled = ids.copy()
    random.Random(seed).shuffle(shuffled)
    counts = _allocate_counts(len(shuffled), ratios)

    train_end = counts["train"]
    dev_end = train_end + counts["dev"]
    splits = {
        "train": sorted(shuffled[:train_end]),
        "dev": sorted(shuffled[train_end:dev_end]),
        "test": sorted(shuffled[dev_end:]),
    }
    registry = {
        "schema_version": 1,
        "unit": "gsm8k_problem_id",
        "seed": seed,
        "ratios": ratios.as_dict(),
        "num_problems": len(ids),
        "counts": {name: len(splits[name]) for name in SPLIT_NAMES},
        "splits": splits,
        "problem_to_split": {
            problem_id: split
            for split in SPLIT_NAMES
            for problem_id in splits[split]
        },
    }
    validate_split_registry(registry, expected_problem_ids=ids)
    return registry


def validate_split_registry(
    registry: Mapping[str, Any],
    *,
    expected_problem_ids: Iterable[str] | None = None,
) -> None:
    if registry.get("unit") != "gsm8k_problem_id":
        raise SplitLeakageError("Registry unit must be gsm8k_problem_id")
    splits = registry.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != set(SPLIT_NAMES):
        raise SplitLeakageError("Registry must contain exactly train/dev/test splits")

    split_sets: dict[str, set[str]] = {}
    for split in SPLIT_NAMES:
        values = splits[split]
        if not isinstance(values, list):
            raise SplitLeakageError(f"splits.{split} must be a list")
        normalized = [str(value) for value in values]
        if len(normalized) != len(set(normalized)):
            raise SplitLeakageError(f"Duplicate problem_id within {split}")
        split_sets[split] = set(normalized)

    for left_index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[left_index + 1 :]:
            overlap = split_sets[left] & split_sets[right]
            if overlap:
                sample = sorted(overlap)[:5]
                raise SplitLeakageError(
                    f"Problem-level split leakage between {left} and {right}: {sample}"
                )

    all_ids = set().union(*split_sets.values())
    if expected_problem_ids is not None:
        expected = set(_normalize_problem_ids(expected_problem_ids))
        missing = expected - all_ids
        extra = all_ids - expected
        if missing or extra:
            raise SplitLeakageError(
                f"Registry coverage mismatch: missing={sorted(missing)[:5]}, "
                f"extra={sorted(extra)[:5]}"
            )

    problem_to_split = registry.get("problem_to_split")
    if not isinstance(problem_to_split, Mapping):
        raise SplitLeakageError("problem_to_split mapping is required")
    expected_mapping = {
        problem_id: split
        for split, problem_set in split_sets.items()
        for problem_id in problem_set
    }
    normalized_mapping = {str(key): value for key, value in problem_to_split.items()}
    if normalized_mapping != expected_mapping:
        raise SplitLeakageError("problem_to_split does not match the split lists")


def assert_rows_respect_registry(
    rows: Iterable[Mapping[str, Any]], registry: Mapping[str, Any]
) -> None:
    """Fail on cross-split problems, row/registry mismatch, or ID collisions."""
    validate_split_registry(registry)
    assignment = registry["problem_to_split"]
    observed_problem_splits: dict[str, set[str]] = {}
    transition_keys: set[tuple[str, str, str]] = set()

    for row_index, row in enumerate(rows):
        problem_id = str(row.get("problem_id", ""))
        trajectory_id = str(row.get("trajectory_id", ""))
        transition_id = str(row.get("transition_id", ""))
        split = str(row.get("split", ""))
        if not problem_id or not trajectory_id or not transition_id:
            raise SplitLeakageError(f"Missing ID at row {row_index}")
        expected_split = assignment.get(problem_id)
        if expected_split is None:
            raise SplitLeakageError(f"Unknown problem_id at row {row_index}: {problem_id}")
        if split != expected_split:
            raise SplitLeakageError(
                f"Row {row_index} has split={split!r}, expected {expected_split!r} "
                f"for problem_id={problem_id!r}"
            )

        observed_problem_splits.setdefault(problem_id, set()).add(split)
        if len(observed_problem_splits[problem_id]) != 1:
            raise SplitLeakageError(f"problem_id {problem_id!r} spans multiple splits")

        key = (problem_id, trajectory_id, transition_id)
        if key in transition_keys:
            raise SplitLeakageError(f"Duplicate transition key at row {row_index}: {key}")
        transition_keys.add(key)


def registry_sha256(registry: Mapping[str, Any]) -> str:
    payload = json.dumps(registry, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_split_registry(registry: Mapping[str, Any], path: str | Path) -> Path:
    validate_split_registry(registry)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(registry, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output


def load_split_registry(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        registry = json.load(handle)
    validate_split_registry(registry)
    return registry
