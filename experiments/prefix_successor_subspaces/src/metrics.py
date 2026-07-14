"""Rotation-invariant geometry and functional evaluation metrics."""

from __future__ import annotations

from typing import Any

import numpy as np


class MetricError(ValueError):
    """Raised when a metric is undefined or receives invalid inputs."""


def _numeric(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise MetricError(f"{name} must be numeric")
    if not np.issubdtype(array.dtype, np.floating) or array.dtype.itemsize < 4:
        array = array.astype(np.float64)
    if array.size == 0:
        raise MetricError(f"{name} must not be empty")
    if not np.isfinite(array).all():
        raise MetricError(f"{name} contains NaN or Inf")
    return array


def _paired_flat(target: Any, prediction: Any) -> tuple[np.ndarray, np.ndarray]:
    left = _numeric(target, name="target")
    right = _numeric(prediction, name="prediction")
    if left.shape != right.shape:
        raise MetricError(f"target shape {left.shape} != prediction shape {right.shape}")
    return left.reshape(-1).astype(np.float64), right.reshape(-1).astype(np.float64)


def pearson_correlation(target: Any, prediction: Any, *, eps: float = 1e-12) -> float:
    target_array, prediction_array = _paired_flat(target, prediction)
    target_centered = target_array - target_array.mean()
    prediction_centered = prediction_array - prediction_array.mean()
    denominator = np.linalg.norm(target_centered) * np.linalg.norm(prediction_centered)
    if denominator <= eps:
        raise MetricError("Pearson correlation is undefined for a constant input")
    return float(np.clip(np.dot(target_centered, prediction_centered) / denominator, -1, 1))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """One-based average ranks with deterministic handling of exact ties."""

    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def spearman_correlation(target: Any, prediction: Any, *, eps: float = 1e-12) -> float:
    target_array, prediction_array = _paired_flat(target, prediction)
    return pearson_correlation(
        _average_ranks(target_array), _average_ranks(prediction_array), eps=eps
    )


def coefficient_of_determination(
    target: Any, prediction: Any, *, eps: float = 1e-12
) -> float:
    """Return predictive R^2 = 1 - SSE/SST (not squared correlation)."""

    target_array, prediction_array = _paired_flat(target, prediction)
    residual_sum = float(np.sum(np.square(target_array - prediction_array)))
    total_sum = float(np.sum(np.square(target_array - target_array.mean())))
    if total_sum <= eps:
        raise MetricError("R^2 is undefined for a constant target")
    return float(1.0 - residual_sum / total_sum)


def mean_absolute_error(target: Any, prediction: Any) -> float:
    target_array, prediction_array = _paired_flat(target, prediction)
    return float(np.mean(np.abs(target_array - prediction_array)))


def sign_agreement(target: Any, prediction: Any, *, zero_tolerance: float = 0.0) -> float:
    target_array, prediction_array = _paired_flat(target, prediction)
    if zero_tolerance < 0 or not np.isfinite(zero_tolerance):
        raise MetricError("zero_tolerance must be finite and non-negative")
    target_sign = np.where(
        np.abs(target_array) <= zero_tolerance, 0, np.sign(target_array)
    )
    prediction_sign = np.where(
        np.abs(prediction_array) <= zero_tolerance, 0, np.sign(prediction_array)
    )
    return float(np.mean(target_sign == prediction_sign))


def linearized_effect_metrics(target: Any, prediction: Any) -> dict[str, float]:
    """Metrics for ``g^T gamma`` versus ``g^T P_U gamma``."""

    return {
        "r2": coefficient_of_determination(target, prediction),
        "pearson": pearson_correlation(target, prediction),
        "spearman": spearman_correlation(target, prediction),
        "mae": mean_absolute_error(target, prediction),
        "sign_agreement": sign_agreement(target, prediction),
    }


def cosine_similarity(x: Any, y: Any, *, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    left = _numeric(x, name="x")
    right = _numeric(y, name="y")
    if left.shape != right.shape:
        raise MetricError("x and y must have the same shape")
    numerator = np.sum(left * right, axis=axis)
    denominator = np.linalg.norm(left, axis=axis) * np.linalg.norm(right, axis=axis)
    if np.any(denominator <= eps):
        raise MetricError("Cosine similarity is undefined for zero-norm vectors")
    return np.clip(numerator / denominator, -1.0, 1.0)


def logits_to_probabilities(logits: Any, *, axis: int = -1) -> np.ndarray:
    values = _numeric(logits, name="logits").astype(np.float64)
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=axis, keepdims=True)


def _probabilities(value: Any, *, name: str, axis: int) -> np.ndarray:
    array = _numeric(value, name=name).astype(np.float64)
    if np.any(array < 0):
        raise MetricError(f"{name} contains negative probability")
    normalizer = array.sum(axis=axis, keepdims=True)
    if np.any(normalizer <= 0):
        raise MetricError(f"{name} has a zero-mass distribution")
    return array / normalizer


def _reduce(value: np.ndarray, reduction: str) -> np.ndarray | float:
    if reduction == "none":
        return value
    if reduction == "mean":
        return float(np.mean(value))
    if reduction == "sum":
        return float(np.sum(value))
    raise MetricError("reduction must be 'none', 'mean', or 'sum'")


def kl_divergence(
    p: Any,
    q: Any,
    *,
    axis: int = -1,
    eps: float = 1e-12,
    reduction: str = "none",
) -> np.ndarray | float:
    """Compute KL(P || Q) in nats after validating/renormalizing rows."""

    left = _probabilities(p, name="p", axis=axis)
    right = _probabilities(q, name="q", axis=axis)
    if left.shape != right.shape:
        raise MetricError("p and q must have the same shape")
    terms = np.where(
        left > 0,
        left * (np.log(np.clip(left, eps, None)) - np.log(np.clip(right, eps, None))),
        0.0,
    )
    values = np.maximum(np.sum(terms, axis=axis), 0.0)
    return _reduce(values, reduction)


def jensen_shannon_divergence(
    p: Any,
    q: Any,
    *,
    axis: int = -1,
    eps: float = 1e-12,
    reduction: str = "none",
) -> np.ndarray | float:
    left = _probabilities(p, name="p", axis=axis)
    right = _probabilities(q, name="q", axis=axis)
    if left.shape != right.shape:
        raise MetricError("p and q must have the same shape")
    midpoint = 0.5 * (left + right)
    left_kl = kl_divergence(left, midpoint, axis=axis, eps=eps, reduction="none")
    right_kl = kl_divergence(right, midpoint, axis=axis, eps=eps, reduction="none")
    values = 0.5 * (np.asarray(left_kl) + np.asarray(right_kl))
    return _reduce(values, reduction)


def top1_agreement(
    original_scores: Any,
    intervened_scores: Any,
    *,
    axis: int = -1,
    reduction: str = "none",
) -> np.ndarray | float:
    original = _numeric(original_scores, name="original_scores")
    intervened = _numeric(intervened_scores, name="intervened_scores")
    if original.shape != intervened.shape:
        raise MetricError("Score arrays must have the same shape")
    agreement = np.argmax(original, axis=axis) == np.argmax(intervened, axis=axis)
    return _reduce(agreement.astype(np.float64), reduction)


def topk_overlap(
    original_scores: Any,
    intervened_scores: Any,
    k: int,
    *,
    axis: int = -1,
    reduction: str = "none",
) -> np.ndarray | float:
    """Return set intersection size divided by k for each distribution."""

    original = _numeric(original_scores, name="original_scores")
    intervened = _numeric(intervened_scores, name="intervened_scores")
    if original.shape != intervened.shape:
        raise MetricError("Score arrays must have the same shape")
    axis = axis % original.ndim
    vocabulary_size = original.shape[axis]
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= vocabulary_size:
        raise MetricError(f"k must be in [1, {vocabulary_size}]")
    left = np.argpartition(original, vocabulary_size - k, axis=axis)
    right = np.argpartition(intervened, vocabulary_size - k, axis=axis)
    left = np.take(left, np.arange(vocabulary_size - k, vocabulary_size), axis=axis)
    right = np.take(right, np.arange(vocabulary_size - k, vocabulary_size), axis=axis)
    # Move vocabulary/top-k to the last axis for a broadcasted membership test.
    left = np.moveaxis(left, axis, -1)
    right = np.moveaxis(right, axis, -1)
    overlap = (left[..., :, None] == right[..., None, :]).any(axis=-1).sum(axis=-1) / k
    return _reduce(overlap.astype(np.float64), reduction)


def answer_logit_margin(
    logits: Any, positive_token_ids: Any, negative_token_ids: Any
) -> np.ndarray:
    values = _numeric(logits, name="logits")
    if values.ndim < 1:
        raise MetricError("logits needs a vocabulary axis")
    positive = np.asarray(positive_token_ids)
    negative = np.asarray(negative_token_ids)
    try:
        positive = np.broadcast_to(positive, values.shape[:-1])
        negative = np.broadcast_to(negative, values.shape[:-1])
    except ValueError as exc:
        raise MetricError("Token ID shapes do not broadcast to logits batch shape") from exc
    if not np.issubdtype(positive.dtype, np.integer) or not np.issubdtype(
        negative.dtype, np.integer
    ):
        raise MetricError("Token IDs must be integers")
    vocabulary_size = values.shape[-1]
    if (
        np.any(positive < 0)
        or np.any(negative < 0)
        or np.any(positive >= vocabulary_size)
        or np.any(negative >= vocabulary_size)
    ):
        raise MetricError("Token ID out of vocabulary bounds")
    positive_logits = np.take_along_axis(values, positive[..., None], axis=-1)[..., 0]
    negative_logits = np.take_along_axis(values, negative[..., None], axis=-1)[..., 0]
    return positive_logits - negative_logits


def mahalanobis_distance(x: Any, mean: Any, precision: Any) -> np.ndarray:
    values = _numeric(x, name="x")
    center = _numeric(mean, name="mean")
    inverse_covariance = _numeric(precision, name="precision")
    if center.ndim != 1 or values.shape[-1] != center.shape[0]:
        raise MetricError("mean must have shape [hidden]")
    if inverse_covariance.shape != (center.shape[0], center.shape[0]):
        raise MetricError("precision must have shape [hidden, hidden]")
    if not np.allclose(inverse_covariance, inverse_covariance.T, atol=1e-6, rtol=1e-6):
        raise MetricError("precision must be symmetric")
    centered = values - center
    squared = np.einsum("...d,de,...e->...", centered, inverse_covariance, centered)
    if np.any(squared < -1e-7):
        raise MetricError("precision produced a negative squared distance")
    return np.sqrt(np.maximum(squared, 0.0))


def nearest_neighbor_distance(
    x: Any,
    natural_activations: Any,
    *,
    chunk_size: int = 1024,
) -> np.ndarray:
    """Exact Euclidean nearest-neighbor distance with bounded temporary memory."""

    queries = _numeric(x, name="x")
    reference = _numeric(natural_activations, name="natural_activations")
    if queries.ndim < 2 or reference.ndim != 2:
        raise MetricError("x must be [..., hidden], reference must be [sample, hidden]")
    if queries.shape[-1] != reference.shape[-1]:
        raise MetricError("Query/reference hidden dimensions differ")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise MetricError("chunk_size must be a positive integer")
    flat = queries.reshape(-1, queries.shape[-1])
    distances = np.full(flat.shape[0], np.inf, dtype=np.float64)
    for start in range(0, reference.shape[0], chunk_size):
        chunk = reference[start : start + chunk_size]
        squared = np.sum(np.square(flat[:, None, :] - chunk[None, :, :]), axis=-1)
        distances = np.minimum(distances, np.sqrt(np.min(squared, axis=1)))
    return distances.reshape(queries.shape[:-1])


def intervention_distribution_metrics(
    original_logits: Any,
    intervened_logits: Any,
    *,
    top_k: int = 10,
) -> dict[str, Any]:
    original_probabilities = logits_to_probabilities(original_logits)
    intervened_probabilities = logits_to_probabilities(intervened_logits)
    return {
        "kl": kl_divergence(
            original_probabilities, intervened_probabilities, reduction="none"
        ),
        "js": jensen_shannon_divergence(
            original_probabilities, intervened_probabilities, reduction="none"
        ),
        "top1_agreement": top1_agreement(
            original_logits, intervened_logits, reduction="none"
        ),
        "topk_overlap": topk_overlap(
            original_logits, intervened_logits, top_k, reduction="none"
        ),
    }


# Common concise aliases.
pearson = pearson_correlation
spearman = spearman_correlation
r2_score = coefficient_of_determination
mae = mean_absolute_error
js_divergence = jensen_shannon_divergence


__all__ = [
    "MetricError",
    "answer_logit_margin",
    "coefficient_of_determination",
    "cosine_similarity",
    "intervention_distribution_metrics",
    "jensen_shannon_divergence",
    "js_divergence",
    "kl_divergence",
    "linearized_effect_metrics",
    "logits_to_probabilities",
    "mae",
    "mahalanobis_distance",
    "mean_absolute_error",
    "nearest_neighbor_distance",
    "pearson",
    "pearson_correlation",
    "r2_score",
    "sign_agreement",
    "spearman",
    "spearman_correlation",
    "top1_agreement",
    "topk_overlap",
]
