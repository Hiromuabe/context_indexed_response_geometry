"""Deterministic, problem-disjoint matching of wrong prefixes.

The matchers in this module deliberately operate at the prefix level while
enforcing exclusion at the original GSM8K problem level.  They do not use any
held-out response tensors; matching variables are restricted to pre-response
metadata and the cached prefix representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class Match:
    """One query-prefix to wrong-prefix match."""

    prefix_id: str
    matched_prefix_id: str
    distance: float
    metadata_distance: float
    hidden_cosine_distance: float
    same_last_token: bool
    same_problem: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "prefix_id": self.prefix_id,
            "matched_prefix_id": self.matched_prefix_id,
            "distance": self.distance,
            "metadata_distance": self.metadata_distance,
            "hidden_cosine_distance": self.hidden_cosine_distance,
            "same_last_token": self.same_last_token,
            "same_problem": self.same_problem,
        }


def _as_1d(values: Sequence[Any], name: str, n: int | None = None) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {array.shape}")
    if n is not None and len(array) != n:
        raise ValueError(f"{name} has {len(array)} rows, expected {n}")
    return array


def _standardize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("metadata matrix must be two-dimensional")
    center = np.nanmedian(matrix, axis=0)
    # A robust scale keeps one outlying very long trajectory from dominating.
    q75 = np.nanpercentile(matrix, 75.0, axis=0)
    q25 = np.nanpercentile(matrix, 25.0, axis=0)
    scale = q75 - q25
    standard_deviation = np.nanstd(matrix, axis=0)
    scale = np.where(scale > 1e-12, scale, standard_deviation)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (matrix - center) / scale
    if not np.isfinite(standardized).all():
        raise ValueError("matching metadata contain NaN or infinite values")
    return standardized


def _normalize_rows(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("prefix hidden states must have shape [prefix, hidden]")
    norms = np.linalg.norm(matrix, axis=1)
    valid = norms > 1e-12
    normalized = np.zeros_like(matrix)
    normalized[valid] = matrix[valid] / norms[valid, None]
    return normalized, valid


def match_wrong_prefixes(
    *,
    prefix_ids: Sequence[Any],
    problem_ids: Sequence[Any],
    prefix_lengths: Sequence[float],
    last_token_ids: Sequence[Any],
    position_fractions: Sequence[float],
    entropies: Sequence[float],
    hidden_states: np.ndarray,
    n_matches: int = 10,
    require_same_last_token: bool = False,
    query_mask: Sequence[bool] | None = None,
    candidate_mask: Sequence[bool] | None = None,
    metadata_weight: float = 1.0,
    hidden_weight: float = 1.0,
) -> list[Match]:
    """Match every selected query to deterministic nearest wrong prefixes.

    Exact last-token matches are preferred whenever available.  If
    ``require_same_last_token`` is false and too few exact matches exist, the
    remaining slots are filled using the joint standardized-metadata and
    hidden-cosine distance.  A candidate from the query's GSM8K problem is
    never eligible.
    """

    if n_matches <= 0:
        raise ValueError("n_matches must be positive")
    if metadata_weight < 0 or hidden_weight < 0:
        raise ValueError("matching weights must be non-negative")
    prefix_ids_array = _as_1d(prefix_ids, "prefix_ids")
    n = len(prefix_ids_array)
    if len(set(map(str, prefix_ids_array.tolist()))) != n:
        raise ValueError("prefix_ids must be unique")
    problem_ids_array = _as_1d(problem_ids, "problem_ids", n)
    prefix_lengths_array = _as_1d(prefix_lengths, "prefix_lengths", n).astype(float)
    last_token_array = _as_1d(last_token_ids, "last_token_ids", n)
    fractions_array = _as_1d(position_fractions, "position_fractions", n).astype(float)
    entropies_array = _as_1d(entropies, "entropies", n).astype(float)
    hidden_states = np.asarray(hidden_states)
    if hidden_states.shape[0] != n:
        raise ValueError("hidden_states row count does not match prefix_ids")

    metadata = _standardize(
        np.column_stack([prefix_lengths_array, fractions_array, entropies_array])
    )
    hidden_normalized, valid_hidden = _normalize_rows(hidden_states)
    if query_mask is None:
        query_mask_array = np.ones(n, dtype=bool)
    else:
        query_mask_array = _as_1d(query_mask, "query_mask", n).astype(bool)
    if candidate_mask is None:
        candidate_mask_array = np.ones(n, dtype=bool)
    else:
        candidate_mask_array = _as_1d(candidate_mask, "candidate_mask", n).astype(bool)

    matches: list[Match] = []
    for query_index in np.flatnonzero(query_mask_array):
        eligible = candidate_mask_array.copy()
        eligible &= problem_ids_array != problem_ids_array[query_index]
        eligible[query_index] = False
        exact_last = last_token_array == last_token_array[query_index]
        if require_same_last_token:
            eligible &= exact_last
        candidate_indices = np.flatnonzero(eligible)
        if not len(candidate_indices):
            raise ValueError(
                "No problem-disjoint wrong-prefix candidate for "
                f"{prefix_ids_array[query_index]!r}"
            )

        metadata_distance = np.linalg.norm(
            metadata[candidate_indices] - metadata[query_index], axis=1
        )
        if valid_hidden[query_index]:
            cosine = hidden_normalized[candidate_indices] @ hidden_normalized[query_index]
            hidden_distance = np.where(valid_hidden[candidate_indices], 1.0 - cosine, 1.0)
        else:
            hidden_distance = np.ones(len(candidate_indices), dtype=np.float64)
        total = metadata_weight * metadata_distance + hidden_weight * hidden_distance

        # Prefer exact last-token candidates without turning it into an
        # arbitrary large numerical penalty. lexsort's final key is primary.
        non_exact = (~exact_last[candidate_indices]).astype(np.int8)
        stable_id = np.asarray(
            [str(prefix_ids_array[index]) for index in candidate_indices], dtype=object
        )
        ordering = np.lexsort((stable_id, total, non_exact))
        selected = ordering[: min(n_matches, len(ordering))]
        for local_index in selected:
            candidate_index = int(candidate_indices[local_index])
            matches.append(
                Match(
                    prefix_id=str(prefix_ids_array[query_index]),
                    matched_prefix_id=str(prefix_ids_array[candidate_index]),
                    distance=float(total[local_index]),
                    metadata_distance=float(metadata_distance[local_index]),
                    hidden_cosine_distance=float(hidden_distance[local_index]),
                    same_last_token=bool(exact_last[candidate_index]),
                )
            )

    if any(match.same_problem for match in matches):  # Defensive invariant.
        raise AssertionError("same-problem prefix escaped matching exclusion")
    return matches


def matches_by_prefix(matches: Iterable[Match | Mapping[str, Any]]) -> dict[str, list[str]]:
    """Convert match rows to the lookup used by geometry analyses."""

    result: dict[str, list[str]] = {}
    for match in matches:
        if isinstance(match, Match):
            query = match.prefix_id
            candidate = match.matched_prefix_id
            same_problem = match.same_problem
        else:
            query = str(match["prefix_id"])
            candidate = str(match["matched_prefix_id"])
            same_problem = bool(match.get("same_problem", False))
        if same_problem:
            raise ValueError(f"same-problem wrong-prefix match for {query}")
        result.setdefault(query, []).append(candidate)
    return result


def validate_problem_disjoint_matches(
    matches: Iterable[Match | Mapping[str, Any]],
    prefix_to_problem: Mapping[str, Any],
) -> None:
    """Fail fast if any wrong-prefix pair belongs to the same problem."""

    for match in matches:
        if isinstance(match, Match):
            left, right = match.prefix_id, match.matched_prefix_id
        else:
            left = str(match["prefix_id"])
            right = str(match["matched_prefix_id"])
        if left not in prefix_to_problem or right not in prefix_to_problem:
            raise KeyError(f"unknown prefix in match: {left!r}, {right!r}")
        if str(prefix_to_problem[left]) == str(prefix_to_problem[right]):
            raise ValueError(
                f"wrong-prefix match crosses within problem {prefix_to_problem[left]!r}: "
                f"{left!r} -> {right!r}"
            )


def build_surface_control_pairs(
    prefix_rows: Sequence[Mapping[str, Any]],
    *,
    suffix_tokens: int = 3,
    maximum_pairs: int | None = None,
) -> list[dict[str, Any]]:
    """Build conservative natural-GSM8K surface-controlled prefix pairs.

    Rows may provide ``prefix_token_ids`` or ``token_ids``.  Pairs are emitted
    only across problems.  This helper does not invent semantic-operation
    labels; structural controls requiring such labels must be supplied by the
    dataset and are marked via ``control_type``.
    """

    if suffix_tokens <= 0:
        raise ValueError("suffix_tokens must be positive")
    pairs: list[dict[str, Any]] = []
    for left_index, left in enumerate(prefix_rows):
        left_tokens = tuple(left.get("prefix_token_ids", left.get("token_ids", ())))
        if not left_tokens:
            continue
        for right in prefix_rows[left_index + 1 :]:
            if str(left["problem_id"]) == str(right["problem_id"]):
                continue
            right_tokens = tuple(right.get("prefix_token_ids", right.get("token_ids", ())))
            if not right_tokens:
                continue
            same_suffix = (
                left_tokens[-suffix_tokens:] == right_tokens[-suffix_tokens:]
                and left_tokens[:-suffix_tokens] != right_tokens[:-suffix_tokens]
            )
            same_length_last = (
                len(left_tokens) == len(right_tokens)
                and left_tokens[-1] == right_tokens[-1]
                and left_tokens[:-1] != right_tokens[:-1]
            )
            same_structure = (
                left.get("operation_structure") is not None
                and left.get("operation_structure") == right.get("operation_structure")
                and left_tokens != right_tokens
            )
            if not (same_suffix or same_length_last or same_structure):
                continue
            control_types = []
            if same_suffix:
                control_types.append("same_suffix")
            if same_length_last:
                control_types.append("same_length_and_last_token")
            if same_structure:
                control_types.append("same_operation_structure")
            pairs.append(
                {
                    "prefix_id": str(left["prefix_id"]),
                    "matched_prefix_id": str(right["prefix_id"]),
                    "problem_id": str(left["problem_id"]),
                    "matched_problem_id": str(right["problem_id"]),
                    "control_type": "+".join(control_types),
                }
            )
            if maximum_pairs is not None and len(pairs) >= maximum_pairs:
                return pairs
    return pairs
