"""Leakage-safe cross-fitted removal of prefix and candidate-token effects.

The canonical array layout in this experiment is ``[prefix, token, hidden]``.
All means in this module are accumulated in float32 or float64, even when the
branch cache is stored in float16.

For an evaluation prefix ``i`` and token fold with training set ``T``, the
implemented residual is

    gamma[i, j] = z[i, j] - m[i] - t[j] + mu

where ``m`` uses only ``T`` from the evaluation prefix, while ``t`` and ``mu``
use a *separate* auxiliary-prefix array.  In particular, held-out evaluation
cells cannot affect any fitted baseline term.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


class ResidualizationError(ValueError):
    """Raised when a residualization request violates the scientific design."""


class LeakageError(ResidualizationError):
    """Raised when auxiliary/evaluation or data-split units overlap."""


def _analysis_array(value: Any, *, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    if ndim is not None and array.ndim != ndim:
        raise ResidualizationError(
            f"{name} must have {ndim} dimensions, got shape {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.number):
        raise ResidualizationError(f"{name} must be numeric, got {array.dtype}")
    if array.dtype.itemsize < np.dtype(np.float32).itemsize or not np.issubdtype(
        array.dtype, np.floating
    ):
        array = array.astype(np.float32)
    if not np.isfinite(array).all():
        raise ResidualizationError(f"{name} contains NaN or Inf")
    return array


def _indices(
    values: Iterable[int], *, size: int, name: str, allow_empty: bool = False
) -> np.ndarray:
    raw = np.asarray(list(values))
    if raw.ndim != 1:
        raise ResidualizationError(f"{name} must be one-dimensional")
    if raw.size == 0 and not allow_empty:
        raise ResidualizationError(f"{name} must not be empty")
    if raw.size and not np.issubdtype(raw.dtype, np.integer):
        # Reject silently truncated floating-point indices.
        raise ResidualizationError(f"{name} must contain integers")
    result = raw.astype(np.int64, copy=False)
    if len(np.unique(result)) != len(result):
        raise ResidualizationError(f"{name} contains duplicates")
    if result.size and (int(result.min()) < 0 or int(result.max()) >= size):
        raise ResidualizationError(
            f"{name} contains an index outside [0, {size}): {result.tolist()}"
        )
    return result


def _id_set(values: Sequence[Any], *, name: str) -> set[str]:
    normalized = [str(value) for value in values]
    if any(not value for value in normalized):
        raise LeakageError(f"{name} contains an empty ID")
    if len(normalized) != len(set(normalized)):
        raise LeakageError(f"{name} contains duplicate IDs")
    return set(normalized)


def assert_disjoint_prefix_sets(
    evaluation_prefix_ids: Sequence[Any], auxiliary_prefix_ids: Sequence[Any]
) -> None:
    """Fail if a prefix is used both for evaluation and auxiliary main effects."""

    evaluation = _id_set(evaluation_prefix_ids, name="evaluation_prefix_ids")
    auxiliary = _id_set(auxiliary_prefix_ids, name="auxiliary_prefix_ids")
    overlap = evaluation & auxiliary
    if overlap:
        raise LeakageError(
            "Evaluation prefixes leaked into auxiliary main-effect estimation: "
            f"{sorted(overlap)[:10]}"
        )


def assert_no_problem_leakage(
    split_problem_ids: Mapping[str, Iterable[Any]],
) -> None:
    """Assert that every original problem ID belongs to exactly one split.

    Duplicate prefixes within one split are allowed (a problem may yield several
    prefixes), but the same problem may never occur under two split names.
    """

    seen: dict[str, str] = {}
    for split, problem_ids in split_problem_ids.items():
        split_name = str(split)
        if not split_name:
            raise LeakageError("Split names must be non-empty")
        for value in problem_ids:
            problem_id = str(value)
            if not problem_id:
                raise LeakageError(f"Split {split_name!r} contains an empty problem ID")
            previous = seen.get(problem_id)
            if previous is not None and previous != split_name:
                raise LeakageError(
                    f"problem_id={problem_id!r} occurs in both {previous!r} "
                    f"and {split_name!r}"
                )
            seen[problem_id] = split_name


@dataclass(frozen=True)
class TokenFold:
    """One deterministic candidate-token cross-validation fold."""

    fold_index: int
    train_indices: np.ndarray
    heldout_indices: np.ndarray
    train_token_ids: np.ndarray
    heldout_token_ids: np.ndarray

    def as_dict(self) -> dict[str, Any]:
        return {
            "fold_index": int(self.fold_index),
            "train_indices": self.train_indices.tolist(),
            "heldout_indices": self.heldout_indices.tolist(),
            "train_token_ids": self.train_token_ids.tolist(),
            "heldout_token_ids": self.heldout_token_ids.tolist(),
        }


def make_token_folds(
    token_ids_or_count: int | Sequence[Any],
    *,
    n_folds: int = 5,
    seed: int = 0,
    shuffle: bool = True,
) -> list[TokenFold]:
    """Construct deterministic folds in which every token is held out once."""

    if isinstance(token_ids_or_count, bool):
        raise ResidualizationError("token_ids_or_count must not be boolean")
    if isinstance(token_ids_or_count, int):
        if token_ids_or_count <= 0:
            raise ResidualizationError("The number of tokens must be positive")
        token_ids = np.arange(token_ids_or_count, dtype=np.int64)
    else:
        token_ids = np.asarray(list(token_ids_or_count))
        if token_ids.ndim != 1 or token_ids.size == 0:
            raise ResidualizationError("token_ids must be a non-empty 1D sequence")
        # Candidate token IDs must identify unique columns.
        if len(set(token_ids.tolist())) != len(token_ids):
            raise ResidualizationError("token_ids contains duplicates")

    if isinstance(n_folds, bool) or not isinstance(n_folds, int) or n_folds < 2:
        raise ResidualizationError("n_folds must be an integer >= 2")
    if n_folds > len(token_ids):
        raise ResidualizationError(
            f"n_folds={n_folds} exceeds n_tokens={len(token_ids)}"
        )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ResidualizationError("seed must be an integer")

    positions = np.arange(len(token_ids), dtype=np.int64)
    if shuffle:
        positions = np.random.default_rng(seed).permutation(positions)
    heldout_parts = np.array_split(positions, n_folds)
    folds: list[TokenFold] = []
    all_positions = np.arange(len(token_ids), dtype=np.int64)
    for fold_index, heldout in enumerate(heldout_parts):
        # Sorting makes saved manifests stable and indexing easier to audit.
        heldout = np.sort(heldout.astype(np.int64, copy=False))
        train = np.setdiff1d(all_positions, heldout, assume_unique=True)
        folds.append(
            TokenFold(
                fold_index=fold_index,
                train_indices=train,
                heldout_indices=heldout,
                train_token_ids=token_ids[train].copy(),
                heldout_token_ids=token_ids[heldout].copy(),
            )
        )
    validate_token_folds(folds, n_tokens=len(token_ids))
    return folds


def validate_token_folds(folds: Sequence[TokenFold], *, n_tokens: int) -> None:
    """Fail if folds overlap, omit tokens, or leak held-out tokens into fitting."""

    if not folds:
        raise ResidualizationError("At least one token fold is required")
    observed: list[int] = []
    expected = set(range(n_tokens))
    for expected_fold_index, fold in enumerate(folds):
        if int(fold.fold_index) != expected_fold_index:
            raise ResidualizationError("Token fold indices must be contiguous from zero")
        train = _indices(fold.train_indices, size=n_tokens, name="train_indices")
        heldout = _indices(fold.heldout_indices, size=n_tokens, name="heldout_indices")
        if set(train.tolist()) & set(heldout.tolist()):
            raise LeakageError(f"Token leakage in fold {fold.fold_index}")
        if set(train.tolist()) | set(heldout.tolist()) != expected:
            raise ResidualizationError(
                f"Fold {fold.fold_index} does not partition all candidate tokens"
            )
        observed.extend(heldout.tolist())
    if sorted(observed) != list(range(n_tokens)):
        raise LeakageError("Each candidate token must be held out exactly once")


@dataclass(frozen=True)
class CrossFittedResidualizer:
    """Fitted, audit-friendly cross-fitted additive baseline."""

    prefix_effects: np.ndarray
    token_effects: np.ndarray
    grand_mean: np.ndarray
    train_token_indices: np.ndarray
    heldout_token_indices: np.ndarray

    @property
    def n_prefixes(self) -> int:
        return int(self.prefix_effects.shape[0])

    @property
    def n_tokens(self) -> int:
        return int(self.token_effects.shape[0])

    @property
    def hidden_shape(self) -> tuple[int, ...]:
        return tuple(self.grand_mean.shape)

    def additive_baseline(self, token_indices: Iterable[int] | None = None) -> np.ndarray:
        """Return ``m_i + t_j - mu`` for requested token columns."""

        if token_indices is None:
            indices = np.arange(self.n_tokens, dtype=np.int64)
        else:
            indices = _indices(
                token_indices, size=self.n_tokens, name="token_indices", allow_empty=True
            )
        return (
            self.prefix_effects[:, None, ...]
            + self.token_effects[indices][None, ...]
            - self.grand_mean[None, None, ...]
        )

    def transform(
        self, evaluation_z: Any, token_indices: Iterable[int] | None = None
    ) -> np.ndarray:
        """Subtract the fitted additive baseline without refitting any mean."""

        z = _analysis_array(evaluation_z, name="evaluation_z")
        expected_shape = (self.n_prefixes, self.n_tokens, *self.hidden_shape)
        if z.shape != expected_shape:
            raise ResidualizationError(
                f"evaluation_z has shape {z.shape}, expected {expected_shape}"
            )
        if token_indices is None:
            indices = np.arange(self.n_tokens, dtype=np.int64)
        else:
            indices = _indices(
                token_indices, size=self.n_tokens, name="token_indices", allow_empty=True
            )
        return z[:, indices, ...] - self.additive_baseline(indices)


@dataclass(frozen=True)
class CrossFittedResiduals:
    """Residual tensor plus every term needed to audit its construction."""

    residuals: np.ndarray
    additive_baseline: np.ndarray
    prefix_effects: np.ndarray
    token_effects: np.ndarray
    grand_mean: np.ndarray
    train_token_indices: np.ndarray
    heldout_token_indices: np.ndarray

    @property
    def heldout_residuals(self) -> np.ndarray:
        return self.residuals[:, self.heldout_token_indices, ...]

    @property
    def train_residuals(self) -> np.ndarray:
        return self.residuals[:, self.train_token_indices, ...]

    @property
    def heldout_baseline(self) -> np.ndarray:
        return self.additive_baseline[:, self.heldout_token_indices, ...]

    @property
    def train_baseline(self) -> np.ndarray:
        return self.additive_baseline[:, self.train_token_indices, ...]


def _validate_prefix_identity_separation(
    *,
    evaluation_prefix_ids: Sequence[Any] | None,
    auxiliary_prefix_ids: Sequence[Any] | None,
    evaluation_problem_ids: Sequence[Any] | None,
    auxiliary_problem_ids: Sequence[Any] | None,
    n_evaluation: int,
    n_auxiliary: int,
) -> None:
    if (evaluation_prefix_ids is None) != (auxiliary_prefix_ids is None):
        raise ResidualizationError(
            "evaluation_prefix_ids and auxiliary_prefix_ids must be provided together"
        )
    if evaluation_prefix_ids is not None and auxiliary_prefix_ids is not None:
        if len(evaluation_prefix_ids) != n_evaluation:
            raise ResidualizationError("evaluation_prefix_ids length mismatch")
        if len(auxiliary_prefix_ids) != n_auxiliary:
            raise ResidualizationError("auxiliary_prefix_ids length mismatch")
        assert_disjoint_prefix_sets(evaluation_prefix_ids, auxiliary_prefix_ids)

    if (evaluation_problem_ids is None) != (auxiliary_problem_ids is None):
        raise ResidualizationError(
            "evaluation_problem_ids and auxiliary_problem_ids must be provided together"
        )
    if evaluation_problem_ids is not None and auxiliary_problem_ids is not None:
        if len(evaluation_problem_ids) != n_evaluation:
            raise ResidualizationError("evaluation_problem_ids length mismatch")
        if len(auxiliary_problem_ids) != n_auxiliary:
            raise ResidualizationError("auxiliary_problem_ids length mismatch")
        overlap = set(map(str, evaluation_problem_ids)) & set(
            map(str, auxiliary_problem_ids)
        )
        if overlap:
            raise LeakageError(
                "Evaluation problem IDs leaked into auxiliary prefixes: "
                f"{sorted(overlap)[:10]}"
            )


def fit_cross_fitted_residualizer(
    evaluation_z: Any,
    auxiliary_z: Any,
    train_token_indices: Iterable[int],
    heldout_token_indices: Iterable[int] | None = None,
    *,
    evaluation_prefix_ids: Sequence[Any] | None = None,
    auxiliary_prefix_ids: Sequence[Any] | None = None,
    evaluation_problem_ids: Sequence[Any] | None = None,
    auxiliary_problem_ids: Sequence[Any] | None = None,
) -> CrossFittedResidualizer:
    """Fit leakage-safe prefix/token effects for one token fold.

    ``evaluation_z`` and ``auxiliary_z`` must have shape ``[P, K, ...]`` and
    share the candidate-token order and hidden dimensions.  The prefix mean is
    fitted from evaluation prefixes but training token columns only.  Token
    effects and the grand mean are fitted from the separate auxiliary array.
    """

    z = _analysis_array(evaluation_z, name="evaluation_z")
    aux = _analysis_array(auxiliary_z, name="auxiliary_z")
    if z.ndim < 3 or aux.ndim != z.ndim:
        raise ResidualizationError(
            "evaluation_z and auxiliary_z must have shape [prefix, token, hidden...]"
        )
    if z.shape[0] == 0 or aux.shape[0] == 0 or z.shape[1] == 0:
        raise ResidualizationError("Prefix and token axes must be non-empty")
    if z.shape[1:] != aux.shape[1:]:
        raise ResidualizationError(
            "Auxiliary and evaluation arrays must share token/hidden shape: "
            f"{z.shape} vs {aux.shape}"
        )
    # Avoid float32/float64 promotion surprises between the two sources.
    dtype = np.result_type(z.dtype, aux.dtype, np.float32)
    z = z.astype(dtype, copy=False)
    aux = aux.astype(dtype, copy=False)

    n_tokens = z.shape[1]
    train = _indices(train_token_indices, size=n_tokens, name="train_token_indices")
    if heldout_token_indices is None:
        heldout = np.setdiff1d(
            np.arange(n_tokens, dtype=np.int64), train, assume_unique=True
        )
    else:
        heldout = _indices(
            heldout_token_indices,
            size=n_tokens,
            name="heldout_token_indices",
            allow_empty=True,
        )
    if set(train.tolist()) & set(heldout.tolist()):
        raise LeakageError("Held-out candidate tokens appear in train_token_indices")
    if set(train.tolist()) | set(heldout.tolist()) != set(range(n_tokens)):
        raise ResidualizationError(
            "train_token_indices and heldout_token_indices must partition all tokens"
        )

    _validate_prefix_identity_separation(
        evaluation_prefix_ids=evaluation_prefix_ids,
        auxiliary_prefix_ids=auxiliary_prefix_ids,
        evaluation_problem_ids=evaluation_problem_ids,
        auxiliary_problem_ids=auxiliary_problem_ids,
        n_evaluation=z.shape[0],
        n_auxiliary=aux.shape[0],
    )

    # Each quantity is deliberately expressed separately to make leakage audits
    # and saved NPZ artifacts transparent.
    prefix_effects = z[:, train, ...].mean(axis=1)
    token_effects = aux.mean(axis=0)
    grand_mean = aux[:, train, ...].mean(axis=(0, 1))
    return CrossFittedResidualizer(
        prefix_effects=prefix_effects,
        token_effects=token_effects,
        grand_mean=grand_mean,
        train_token_indices=train.copy(),
        heldout_token_indices=heldout.copy(),
    )


def compute_cross_fitted_residuals(
    evaluation_z: Any,
    auxiliary_z: Any,
    train_token_indices: Iterable[int],
    heldout_token_indices: Iterable[int] | None = None,
    **identity_kwargs: Any,
) -> CrossFittedResiduals:
    """Fit and apply the cross-fitted baseline for one token fold."""

    residualizer = fit_cross_fitted_residualizer(
        evaluation_z,
        auxiliary_z,
        train_token_indices,
        heldout_token_indices,
        **identity_kwargs,
    )
    residuals = residualizer.transform(evaluation_z)
    baseline = residualizer.additive_baseline()
    return CrossFittedResiduals(
        residuals=residuals,
        additive_baseline=baseline,
        prefix_effects=residualizer.prefix_effects,
        token_effects=residualizer.token_effects,
        grand_mean=residualizer.grand_mean,
        train_token_indices=residualizer.train_token_indices,
        heldout_token_indices=residualizer.heldout_token_indices,
    )


def compute_cross_fitted_residuals_from_full(
    z: Any,
    evaluation_prefix_indices: Iterable[int],
    auxiliary_prefix_indices: Iterable[int],
    train_token_indices: Iterable[int],
    heldout_token_indices: Iterable[int] | None = None,
    *,
    prefix_ids: Sequence[Any] | None = None,
    problem_ids: Sequence[Any] | None = None,
) -> CrossFittedResiduals:
    """Convenience wrapper when evaluation and auxiliary prefixes share one cube."""

    array = _analysis_array(z, name="z")
    if array.ndim < 3:
        raise ResidualizationError("z must have shape [prefix, token, hidden...]")
    evaluation = _indices(
        evaluation_prefix_indices,
        size=array.shape[0],
        name="evaluation_prefix_indices",
    )
    auxiliary = _indices(
        auxiliary_prefix_indices,
        size=array.shape[0],
        name="auxiliary_prefix_indices",
    )
    if set(evaluation.tolist()) & set(auxiliary.tolist()):
        raise LeakageError("Evaluation and auxiliary prefix indices overlap")
    kwargs: dict[str, Any] = {}
    if prefix_ids is not None:
        if len(prefix_ids) != array.shape[0]:
            raise ResidualizationError("prefix_ids length mismatch")
        kwargs["evaluation_prefix_ids"] = [prefix_ids[index] for index in evaluation]
        kwargs["auxiliary_prefix_ids"] = [prefix_ids[index] for index in auxiliary]
    if problem_ids is not None:
        if len(problem_ids) != array.shape[0]:
            raise ResidualizationError("problem_ids length mismatch")
        kwargs["evaluation_problem_ids"] = [problem_ids[index] for index in evaluation]
        kwargs["auxiliary_problem_ids"] = [problem_ids[index] for index in auxiliary]
    return compute_cross_fitted_residuals(
        array[evaluation],
        array[auxiliary],
        train_token_indices,
        heldout_token_indices,
        **kwargs,
    )


def cross_fit_all_token_folds(
    evaluation_z: Any,
    auxiliary_z: Any,
    folds: Sequence[TokenFold],
    **identity_kwargs: Any,
) -> list[CrossFittedResiduals]:
    """Compute an independently fitted residual artifact for every token fold."""

    z = _analysis_array(evaluation_z, name="evaluation_z")
    if z.ndim < 3:
        raise ResidualizationError("evaluation_z must have at least three dimensions")
    validate_token_folds(folds, n_tokens=z.shape[1])
    return [
        compute_cross_fitted_residuals(
            z,
            auxiliary_z,
            fold.train_indices,
            fold.heldout_indices,
            **identity_kwargs,
        )
        for fold in folds
    ]


# A concise alias used by a few analysis scripts and external notebooks.
cross_fitted_residuals = compute_cross_fitted_residuals


__all__ = [
    "CrossFittedResidualizer",
    "CrossFittedResiduals",
    "LeakageError",
    "ResidualizationError",
    "TokenFold",
    "assert_disjoint_prefix_sets",
    "assert_no_problem_leakage",
    "compute_cross_fitted_residuals",
    "compute_cross_fitted_residuals_from_full",
    "cross_fit_all_token_folds",
    "cross_fitted_residuals",
    "fit_cross_fitted_residualizer",
    "make_token_folds",
    "validate_token_folds",
]
