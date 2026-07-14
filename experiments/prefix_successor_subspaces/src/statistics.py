"""Problem-level inference, two-way bootstrap, and prefix/token nulls.

Candidate-token cells are never treated as independent observations here.  The
default hierarchy first aggregates all prefixes belonging to one GSM8K problem,
then resamples problems.  The optional two-way bootstrap additionally resamples
the shared candidate-token axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import numpy as np


class StatisticsError(ValueError):
    """Raised for invalid statistical units, missing cells, or nonfinite data."""


def _values(value: Any, *, name: str = "values", nan_policy: str = "raise") -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise StatisticsError(f"{name} must be numeric")
    if not np.issubdtype(array.dtype, np.floating) or array.dtype.itemsize < 4:
        array = array.astype(np.float64)
    if array.size == 0:
        raise StatisticsError(f"{name} must not be empty")
    if nan_policy not in {"raise", "omit"}:
        raise StatisticsError("nan_policy must be 'raise' or 'omit'")
    if nan_policy == "raise" and not np.isfinite(array).all():
        raise StatisticsError(f"{name} contains NaN or Inf")
    return array


def _normalized_ids(ids: Sequence[Any], *, length: int, name: str) -> np.ndarray:
    if len(ids) != length:
        raise StatisticsError(f"{name} length {len(ids)} != observations {length}")
    result = np.asarray([str(value) for value in ids], dtype=object)
    if any(not value for value in result.tolist()):
        raise StatisticsError(f"{name} contains an empty ID")
    return result


def _ordered_unique(values: Iterable[Any]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


def _mean(array: np.ndarray, *, axis: int | tuple[int, ...], nan_policy: str) -> np.ndarray:
    if nan_policy == "omit":
        with np.errstate(invalid="ignore"):
            result = np.nanmean(array, axis=axis)
        if not np.isfinite(result).all():
            raise StatisticsError("A resampled unit contains no finite observations")
        return result
    return np.mean(array, axis=axis)


def aggregate_by_problem(
    values: Any,
    problem_ids: Sequence[Any],
    *,
    nan_policy: str = "raise",
) -> tuple[np.ndarray, np.ndarray]:
    """Average all prefix/cell observations within each original problem."""

    array = _values(values, nan_policy=nan_policy)
    if array.ndim == 0:
        raise StatisticsError("values must have an observation axis")
    ids = _normalized_ids(problem_ids, length=array.shape[0], name="problem_ids")
    unique = _ordered_unique(ids)
    aggregated = np.stack(
        [_mean(array[ids == problem_id], axis=0, nan_policy=nan_policy) for problem_id in unique],
        axis=0,
    )
    return np.asarray(unique, dtype=object), aggregated


@dataclass(frozen=True)
class BootstrapResult:
    estimate: Any
    lower: Any
    upper: Any
    samples: np.ndarray
    confidence: float
    n_bootstrap: int
    seed: int
    n_units: int
    unit: str

    def as_dict(self, *, include_samples: bool = False) -> dict[str, Any]:
        def serializable(value: Any) -> Any:
            array = np.asarray(value)
            return float(array) if array.ndim == 0 else array.tolist()

        result = {
            "estimate": serializable(self.estimate),
            "ci_lower": serializable(self.lower),
            "ci_upper": serializable(self.upper),
            "confidence": float(self.confidence),
            "n_bootstrap": int(self.n_bootstrap),
            "seed": int(self.seed),
            "n_units": int(self.n_units),
            "unit": self.unit,
        }
        if include_samples:
            result["samples"] = self.samples.tolist()
        return result


def _validate_bootstrap_args(
    *, n_bootstrap: int, seed: int, confidence: float
) -> None:
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, int) or n_bootstrap <= 0:
        raise StatisticsError("n_bootstrap must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise StatisticsError("seed must be an integer")
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise StatisticsError("confidence must be in (0, 1)")


def _ci(samples: np.ndarray, confidence: float) -> tuple[np.ndarray, np.ndarray]:
    alpha = (1.0 - confidence) / 2.0
    return (
        np.quantile(samples, alpha, axis=0),
        np.quantile(samples, 1.0 - alpha, axis=0),
    )


def hierarchical_bootstrap_ci(
    values: Any,
    problem_ids: Sequence[Any],
    n_bootstrap: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
    *,
    statistic: Callable[[np.ndarray], Any] | None = None,
    nan_policy: str = "raise",
) -> BootstrapResult:
    """Aggregate within problem and bootstrap independent problem units."""

    _validate_bootstrap_args(
        n_bootstrap=n_bootstrap, seed=seed, confidence=confidence
    )
    _, problem_values = aggregate_by_problem(
        values, problem_ids, nan_policy=nan_policy
    )
    if problem_values.shape[0] < 2:
        raise StatisticsError("At least two independent problems are required")
    estimator = statistic or (lambda x: _mean(x, axis=0, nan_policy=nan_policy))
    estimate = np.asarray(estimator(problem_values))
    if not np.isfinite(estimate).all():
        raise StatisticsError("Statistic returned NaN or Inf")
    generator = np.random.default_rng(seed)
    samples: list[np.ndarray] = []
    for _ in range(n_bootstrap):
        indices = generator.integers(
            0, problem_values.shape[0], size=problem_values.shape[0]
        )
        value = np.asarray(estimator(problem_values[indices]))
        if not np.isfinite(value).all():
            raise StatisticsError("Bootstrap statistic returned NaN or Inf")
        samples.append(value)
    sample_array = np.stack(samples, axis=0)
    lower, upper = _ci(sample_array, confidence)
    return BootstrapResult(
        estimate=estimate.item() if estimate.ndim == 0 else estimate,
        lower=lower.item() if lower.ndim == 0 else lower,
        upper=upper.item() if upper.ndim == 0 else upper,
        samples=sample_array,
        confidence=confidence,
        n_bootstrap=n_bootstrap,
        seed=seed,
        n_units=int(problem_values.shape[0]),
        unit="problem",
    )


def paired_hierarchical_bootstrap_ci(
    left: Any,
    right: Any,
    problem_ids: Sequence[Any],
    n_bootstrap: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
    *,
    nan_policy: str = "raise",
) -> BootstrapResult:
    """Bootstrap the problem-level paired difference ``left - right``."""

    left_array = _values(left, name="left", nan_policy=nan_policy)
    right_array = _values(right, name="right", nan_policy=nan_policy)
    if left_array.shape != right_array.shape:
        raise StatisticsError("left and right must have the same shape")
    return hierarchical_bootstrap_ci(
        left_array - right_array,
        problem_ids,
        n_bootstrap=n_bootstrap,
        seed=seed,
        confidence=confidence,
        nan_policy=nan_policy,
    )


def _dense_problem_token_array(
    values: np.ndarray,
    problem_ids: Sequence[Any],
    token_ids: Sequence[Any],
    *,
    nan_policy: str,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Normalize dense [prefix, token, ...] or long [cell, ...] input."""

    # Dense layout: the first axis is prefixes, second is shared candidate tokens.
    if values.ndim >= 2 and len(problem_ids) == values.shape[0] and len(token_ids) == values.shape[1]:
        prefix_problem_ids = _normalized_ids(
            problem_ids, length=values.shape[0], name="problem_ids"
        )
        normalized_token_ids = [str(value) for value in token_ids]
        if len(set(normalized_token_ids)) != len(normalized_token_ids):
            raise StatisticsError("Dense token_ids must be unique")
        unique_problems = _ordered_unique(prefix_problem_ids)
        problem_token = np.stack(
            [
                _mean(
                    values[prefix_problem_ids == problem_id],
                    axis=0,
                    nan_policy=nan_policy,
                )
                for problem_id in unique_problems
            ],
            axis=0,
        )
        return problem_token, unique_problems, normalized_token_ids

    # Long layout: each first-axis row carries both a problem and token ID.
    if len(problem_ids) != values.shape[0] or len(token_ids) != values.shape[0]:
        raise StatisticsError(
            "For dense input IDs must match [prefix, token]; for long input both "
            "ID arrays must match the first axis"
        )
    row_problems = _normalized_ids(
        problem_ids, length=values.shape[0], name="problem_ids"
    )
    row_tokens = _normalized_ids(token_ids, length=values.shape[0], name="token_ids")
    unique_problems = _ordered_unique(row_problems)
    unique_tokens = _ordered_unique(row_tokens)
    output_shape = (len(unique_problems), len(unique_tokens), *values.shape[1:])
    output = np.empty(output_shape, dtype=values.dtype)
    for problem_index, problem_id in enumerate(unique_problems):
        for token_index, token_id in enumerate(unique_tokens):
            mask = (row_problems == problem_id) & (row_tokens == token_id)
            if not bool(mask.any()):
                raise StatisticsError(
                    "Two-way bootstrap requires a complete problem x token grid; "
                    f"missing ({problem_id!r}, {token_id!r})"
                )
            output[problem_index, token_index] = _mean(
                values[mask], axis=0, nan_policy=nan_policy
            )
    return output, unique_problems, unique_tokens


def two_way_bootstrap_ci(
    values: Any,
    problem_ids: Sequence[Any],
    token_ids: Sequence[Any],
    n_bootstrap: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
    *,
    statistic: Callable[[np.ndarray], Any] | None = None,
    nan_policy: str = "raise",
) -> BootstrapResult:
    """Resample original problems and shared candidate tokens independently."""

    _validate_bootstrap_args(
        n_bootstrap=n_bootstrap, seed=seed, confidence=confidence
    )
    array = _values(values, nan_policy=nan_policy)
    problem_token, unique_problems, unique_tokens = _dense_problem_token_array(
        array, problem_ids, token_ids, nan_policy=nan_policy
    )
    if len(unique_problems) < 2 or len(unique_tokens) < 2:
        raise StatisticsError(
            "Two-way bootstrap requires at least two problems and two candidate tokens"
        )
    estimator = statistic or (
        lambda x: _mean(x, axis=(0, 1), nan_policy=nan_policy)
    )
    estimate = np.asarray(estimator(problem_token))
    if not np.isfinite(estimate).all():
        raise StatisticsError("Statistic returned NaN or Inf")
    generator = np.random.default_rng(seed)
    samples: list[np.ndarray] = []
    for _ in range(n_bootstrap):
        problem_sample = generator.integers(
            0, len(unique_problems), size=len(unique_problems)
        )
        token_sample = generator.integers(
            0, len(unique_tokens), size=len(unique_tokens)
        )
        resampled = problem_token[problem_sample][:, token_sample]
        value = np.asarray(estimator(resampled))
        if not np.isfinite(value).all():
            raise StatisticsError("Bootstrap statistic returned NaN or Inf")
        samples.append(value)
    sample_array = np.stack(samples, axis=0)
    lower, upper = _ci(sample_array, confidence)
    return BootstrapResult(
        estimate=estimate.item() if estimate.ndim == 0 else estimate,
        lower=lower.item() if lower.ndim == 0 else lower,
        upper=upper.item() if upper.ndim == 0 else upper,
        samples=sample_array,
        confidence=confidence,
        n_bootstrap=n_bootstrap,
        seed=seed,
        n_units=len(unique_problems),
        unit="problem_x_candidate_token",
    )


@dataclass(frozen=True)
class TokenwisePermutationPlan:
    """Compact null plan: one independent prefix permutation per token."""

    permutation_indices: np.ndarray  # [permutation, token, output_prefix]
    prefix_ids: np.ndarray
    token_ids: np.ndarray
    seed: int

    @property
    def n_permutations(self) -> int:
        return int(self.permutation_indices.shape[0])

    def apply(self, residuals: Any, permutation_index: int) -> np.ndarray:
        array = _values(residuals, name="residuals")
        if array.ndim < 2:
            raise StatisticsError("residuals must have shape [prefix, token, ...]")
        expected = (len(self.prefix_ids), len(self.token_ids))
        if array.shape[:2] != expected:
            raise StatisticsError(
                f"residual shape {array.shape[:2]} != plan shape {expected}"
            )
        if not 0 <= permutation_index < self.n_permutations:
            raise StatisticsError("permutation_index out of bounds")
        output = np.empty_like(array)
        indices = self.permutation_indices[permutation_index]
        for token_index in range(array.shape[1]):
            output[:, token_index, ...] = array[indices[token_index], token_index, ...]
        return output

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_permutations": self.n_permutations,
            "n_prefixes": len(self.prefix_ids),
            "n_tokens": len(self.token_ids),
            "seed": int(self.seed),
            "prefix_ids": self.prefix_ids.tolist(),
            "token_ids": self.token_ids.tolist(),
        }


def make_tokenwise_prefix_permutation_plan(
    n_prefixes: int,
    n_tokens: int,
    n_permutations: int = 100,
    seed: int = 0,
    *,
    prefix_ids: Sequence[Any] | None = None,
    token_ids: Sequence[Any] | None = None,
) -> TokenwisePermutationPlan:
    """Generate ``pi_j(i)`` while retaining every token's residual multiset."""

    _validate_bootstrap_args(
        n_bootstrap=n_permutations, seed=seed, confidence=0.95
    )
    if n_prefixes < 2 or n_tokens < 1:
        raise StatisticsError("Permutation null needs >=2 prefixes and >=1 token")
    normalized_prefix_ids = np.asarray(
        list(prefix_ids) if prefix_ids is not None else list(range(n_prefixes)),
        dtype=object,
    )
    normalized_token_ids = np.asarray(
        list(token_ids) if token_ids is not None else list(range(n_tokens)),
        dtype=object,
    )
    if len(normalized_prefix_ids) != n_prefixes:
        raise StatisticsError("prefix_ids length mismatch")
    if len(normalized_token_ids) != n_tokens:
        raise StatisticsError("token_ids length mismatch")
    if len(set(normalized_prefix_ids.tolist())) != n_prefixes:
        raise StatisticsError("prefix_ids must be unique")
    if len(set(normalized_token_ids.tolist())) != n_tokens:
        raise StatisticsError("token_ids must be unique")
    generator = np.random.default_rng(seed)
    indices = np.empty((n_permutations, n_tokens, n_prefixes), dtype=np.int32)
    for permutation_index in range(n_permutations):
        for token_index in range(n_tokens):
            indices[permutation_index, token_index] = generator.permutation(n_prefixes)
    return TokenwisePermutationPlan(
        permutation_indices=indices,
        prefix_ids=normalized_prefix_ids,
        token_ids=normalized_token_ids,
        seed=seed,
    )


def tokenwise_prefix_permutation_null(
    residuals: Any,
    prefix_ids: Sequence[Any] | None = None,
    token_ids: Sequence[Any] | None = None,
    n_permutations: int = 100,
    seed: int = 0,
    *,
    statistic: Callable[[np.ndarray], Any] | None = None,
) -> TokenwisePermutationPlan | np.ndarray:
    """Build a compact null plan, or evaluate a statistic under that plan.

    Returning permutation indices by default avoids materializing a potentially
    enormous ``[permutation, prefix, token, hidden]`` tensor.  If ``statistic``
    is supplied, only its null values are retained.
    """

    array = _values(residuals, name="residuals")
    if array.ndim < 2:
        raise StatisticsError("residuals must have shape [prefix, token, ...]")
    plan = make_tokenwise_prefix_permutation_plan(
        array.shape[0],
        array.shape[1],
        n_permutations=n_permutations,
        seed=seed,
        prefix_ids=prefix_ids,
        token_ids=token_ids,
    )
    if statistic is None:
        return plan
    null_values: list[np.ndarray] = []
    for permutation_index in range(n_permutations):
        value = np.asarray(statistic(plan.apply(array, permutation_index)))
        if not np.isfinite(value).all():
            raise StatisticsError("Permutation statistic returned NaN or Inf")
        null_values.append(value)
    return np.stack(null_values, axis=0)


def permutation_p_value(
    observed: float,
    null_values: Any,
    *,
    alternative: str = "greater",
) -> float:
    """Finite-sample corrected permutation p-value."""

    null = _values(null_values, name="null_values").reshape(-1)
    if not np.isfinite(observed):
        raise StatisticsError("observed must be finite")
    if alternative == "greater":
        extreme = np.count_nonzero(null >= observed)
    elif alternative == "less":
        extreme = np.count_nonzero(null <= observed)
    elif alternative == "two-sided":
        extreme = np.count_nonzero(np.abs(null) >= abs(observed))
    else:
        raise StatisticsError("alternative must be greater, less, or two-sided")
    return float((extreme + 1) / (len(null) + 1))


__all__ = [
    "BootstrapResult",
    "StatisticsError",
    "TokenwisePermutationPlan",
    "aggregate_by_problem",
    "hierarchical_bootstrap_ci",
    "make_tokenwise_prefix_permutation_plan",
    "paired_hierarchical_bootstrap_ci",
    "permutation_p_value",
    "tokenwise_prefix_permutation_null",
    "two_way_bootstrap_ci",
]
