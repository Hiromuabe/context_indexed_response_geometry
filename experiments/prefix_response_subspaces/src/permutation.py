from __future__ import annotations

import hashlib
import math

import numpy as np

from .residualization import assert_row_centered


def permute_prefix_labels_by_token(residuals: np.ndarray, strata: np.ndarray, rng: np.random.Generator, *, return_diagnostics: bool = False):
    """Independently permute prefix labels per token within fixed strata, then recenter."""
    r = np.asarray(residuals, dtype=np.float32)
    labels = np.asarray(strata)
    if r.ndim != 3 or labels.ndim != 1 or len(labels) != len(r):
        raise ValueError("residuals/strata shapes are incompatible")
    permuted = np.empty_like(r)
    identity = np.arange(len(r), dtype=np.int64)
    moved_prefix = np.zeros(len(r), dtype=bool)
    moved_assignments = 0
    plan_hasher = hashlib.sha256()
    for token in range(r.shape[1]):
        source_plan = identity.copy()
        for label in np.unique(labels):
            indices = np.flatnonzero(labels == label)
            sources = rng.permutation(indices)
            source_plan[indices] = sources
            permuted[indices, token] = r[sources, token]
        moved = source_plan != identity
        moved_prefix |= moved
        moved_assignments += int(moved.sum())
        plan_hasher.update(source_plan.tobytes())
    permuted -= permuted.mean(axis=1, keepdims=True)
    assert_row_centered(permuted)
    if not return_diagnostics:
        return permuted
    diagnostics = {
        "actual_moved_prefix_count": int(moved_prefix.sum()),
        "actual_moved_assignment_count": moved_assignments,
        "total_prefix_token_assignments": int(r.shape[0] * r.shape[1]),
        "plan_sha256": plan_hasher.hexdigest(),
    }
    return permuted, diagnostics


def stratification_labels(length_bins: np.ndarray, progress_bins: np.ndarray) -> np.ndarray:
    left, right = np.asarray(length_bins), np.asarray(progress_bins)
    if left.shape != right.shape:
        raise ValueError("stratification arrays must have identical shapes")
    return np.asarray([f"{a}:{b}" for a, b in zip(left, right)])


def exchangeability_diagnostics(labels: np.ndarray, minimum_stratum_size: int = 2) -> dict:
    values = np.asarray(labels)
    if values.ndim != 1 or not len(values):
        raise ValueError("exchangeability labels must be a non-empty vector")
    if minimum_stratum_size < 2:
        raise ValueError("minimum_stratum_size must be at least two")
    unique, counts = np.unique(values, return_counts=True)
    exchangeable = int(sum(int(count) for count in counts if int(count) >= minimum_stratum_size))
    return {
        "n_prefixes": int(len(values)),
        "n_strata": int(len(unique)),
        "stratum_sizes": list(map(int, counts)),
        "singleton_strata": int(np.sum(counts == 1)),
        "exchangeable_prefix_fraction": exchangeable / len(values),
    }


def permutation_space_size(labels: np.ndarray, n_token_permutations: int) -> dict:
    values = np.asarray(labels)
    if values.ndim != 1 or not len(values) or n_token_permutations <= 0:
        raise ValueError("labels and n_token_permutations must be non-empty/positive")
    _, counts = np.unique(values, return_counts=True)
    log10_per_token = float(sum(math.lgamma(int(count) + 1) for count in counts) / math.log(10.0))
    total_log10 = log10_per_token * int(n_token_permutations)
    exact = None
    if total_log10 <= 300:
        per_token = math.prod(math.factorial(int(count)) for count in counts)
        exact = str(per_token ** int(n_token_permutations))
    return {
        "tokenwise_permutation_count": int(n_token_permutations),
        "distinct_label_permutations_log10": total_log10,
        "distinct_label_permutations_scientific": f"10^{total_log10:.6f}",
        "distinct_label_permutations_exact": exact,
    }
