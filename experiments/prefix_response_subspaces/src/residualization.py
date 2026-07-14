"""Split-local double centering for interaction contrasts.

For every token set S this module computes
r_ij = (z_ij - mean_q-in-S z_iq) - (mean_k-in-A z_kj - mean_k,q-in-A,S z_kq).
Training and evaluation sets must be passed in separate calls.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


class ResidualizationError(ValueError):
    pass


@dataclass(frozen=True)
class DoubleCentered:
    residuals: np.ndarray
    prefix_mean: np.ndarray
    auxiliary_token_mean: np.ndarray
    auxiliary_grand_mean: np.ndarray
    token_indices: np.ndarray


def _indices(values: Iterable[int], size: int, name: str) -> np.ndarray:
    result = np.asarray(list(values), dtype=np.int64)
    if result.ndim != 1 or not len(result) or len(np.unique(result)) != len(result):
        raise ResidualizationError(f"{name} must be a non-empty unique 1D index set")
    if result.min() < 0 or result.max() >= size:
        raise ResidualizationError(f"{name} contains an out-of-range index")
    return result


def double_center(evaluation_z: np.ndarray, auxiliary_z: np.ndarray, token_indices: Iterable[int]) -> DoubleCentered:
    evaluation = np.asarray(evaluation_z)
    auxiliary = np.asarray(auxiliary_z)
    if evaluation.ndim != 3 or auxiliary.ndim != 3 or evaluation.shape[1:] != auxiliary.shape[1:]:
        raise ResidualizationError("arrays must have shapes [prefix, token, hidden] with matching token/hidden axes")
    if not np.issubdtype(evaluation.dtype, np.floating) or not np.issubdtype(auxiliary.dtype, np.floating):
        raise ResidualizationError("hidden states must be floating point")
    evaluation = evaluation.astype(np.float32, copy=False)
    auxiliary = auxiliary.astype(np.float32, copy=False)
    if not np.isfinite(evaluation).all() or not np.isfinite(auxiliary).all():
        raise ResidualizationError("hidden states contain NaN or Inf")
    indices = _indices(token_indices, evaluation.shape[1], "token_indices")
    z = evaluation[:, indices]
    a = auxiliary[:, indices]
    prefix_mean = z.mean(axis=1)
    token_mean = a.mean(axis=0)
    grand_mean = a.mean(axis=(0, 1))
    residuals = z - prefix_mean[:, None] - token_mean[None] + grand_mean
    # Float32 cancellation can leave an absolute row mean around 1e-4 even
    # though the contrast is mathematically centered.  Remove the measured
    # float64 row mean before storage; this changes only the additive null mode.
    correction = residuals.mean(axis=1,dtype=np.float64).astype(np.float32)
    residuals -= correction[:, None]
    assert_row_centered(residuals)
    return DoubleCentered(residuals, prefix_mean, token_mean, grand_mean, indices)


def center_train_and_evaluation(evaluation_z: np.ndarray, auxiliary_z: np.ndarray, train_indices: Iterable[int], evaluation_indices: Iterable[int]) -> tuple[DoubleCentered, DoubleCentered]:
    train = _indices(train_indices, evaluation_z.shape[1], "train_indices")
    heldout = _indices(evaluation_indices, evaluation_z.shape[1], "evaluation_indices")
    if set(train.tolist()) & set(heldout.tolist()):
        raise ResidualizationError("training and evaluation token sets overlap")
    return double_center(evaluation_z, auxiliary_z, train), double_center(evaluation_z, auxiliary_z, heldout)


def inductive_center(
    evaluation_z: np.ndarray,
    auxiliary_z: np.ndarray,
    train_indices: Iterable[int],
    evaluation_indices: Iterable[int],
) -> DoubleCentered:
    """Apply evaluation-fold-independent target-context centering.

    The target-context mean and auxiliary grand mean are estimated only from
    ``train_indices``.  The candidate-specific auxiliary mean is evaluated at
    each held-out token, which is available without observing the held-out
    target context.  This implements

    r_ij = z_ij - mean_{j' in train} z_ij'
           - mean_{a in A} z_aj
           + mean_{a in A,j' in train} z_aj'.

    Unlike :func:`double_center`, the resulting held-out rows are not forced to
    have zero mean: doing so would reintroduce the held-out target-context mean
    that this sensitivity analysis is designed to avoid.
    """
    evaluation = np.asarray(evaluation_z)
    auxiliary = np.asarray(auxiliary_z)
    if evaluation.ndim != 3 or auxiliary.ndim != 3 or evaluation.shape[1:] != auxiliary.shape[1:]:
        raise ResidualizationError("arrays must have shapes [prefix, token, hidden] with matching token/hidden axes")
    if not np.issubdtype(evaluation.dtype, np.floating) or not np.issubdtype(auxiliary.dtype, np.floating):
        raise ResidualizationError("hidden states must be floating point")
    evaluation = evaluation.astype(np.float32, copy=False)
    auxiliary = auxiliary.astype(np.float32, copy=False)
    if not np.isfinite(evaluation).all() or not np.isfinite(auxiliary).all():
        raise ResidualizationError("hidden states contain NaN or Inf")
    train = _indices(train_indices, evaluation.shape[1], "train_indices")
    heldout = _indices(evaluation_indices, evaluation.shape[1], "evaluation_indices")
    if set(train.tolist()) & set(heldout.tolist()):
        raise ResidualizationError("training and evaluation token sets overlap")
    prefix_mean = evaluation[:, train].mean(axis=1)
    token_mean = auxiliary[:, heldout].mean(axis=0)
    grand_mean = auxiliary[:, train].mean(axis=(0, 1))
    residuals = evaluation[:, heldout] - prefix_mean[:, None] - token_mean[None] + grand_mean
    return DoubleCentered(residuals, prefix_mean, token_mean, grand_mean, heldout)


def assert_row_centered(residuals: np.ndarray, atol: float = 2e-5) -> None:
    row_means = np.mean(residuals,axis=1,dtype=np.float64)
    scale = max(1.0, float(np.abs(residuals).max(initial=0.0)))
    if float(np.abs(row_means).max(initial=0.0)) > atol * scale:
        raise ResidualizationError("double-centered residual row means are not zero")


def explicit_contrast_ev(residuals: np.ndarray, basis: np.ndarray, eps: float = 1e-12) -> float:
    r = np.asarray(residuals, dtype=np.float64)
    u = np.asarray(basis, dtype=np.float64)
    differences = r[:, None, :] - r[None, :, :]
    numerator = np.square(differences @ u).sum()
    denominator = np.square(differences).sum()
    if denominator <= eps:
        raise ResidualizationError("contrast EV denominator is zero")
    return float(numerator / denominator)


def centered_residual_ev(residuals: np.ndarray, basis: np.ndarray, eps: float = 1e-12) -> float:
    r = np.asarray(residuals, dtype=np.float64)
    r = r - r.mean(axis=0, keepdims=True)
    numerator = np.square(r @ np.asarray(basis, dtype=np.float64)).sum()
    denominator = np.square(r).sum()
    if denominator <= eps:
        raise ResidualizationError("centered EV denominator is zero")
    return float(numerator / denominator)
