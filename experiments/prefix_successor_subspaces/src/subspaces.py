"""Leakage-conscious linear-subspace estimation and geometry primitives.

Every fitting function consumes training rows explicitly.  Evaluation helpers
never fit a center or a basis.  Inputs may be float16 on disk, but all linear
algebra is performed in float32 or float64.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np


class SubspaceError(ValueError):
    """Base class for invalid or scientifically unsafe subspace operations."""


class RankDeficiencyError(SubspaceError):
    """Raised when the requested equal-rank comparison cannot be estimated."""


class DegenerateEnergyError(SubspaceError):
    """Raised when explained variance has a zero or near-zero denominator."""


def _float_array(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise SubspaceError(f"{name} must be numeric, got {array.dtype}")
    if not np.issubdtype(array.dtype, np.floating) or array.dtype.itemsize < 4:
        array = array.astype(np.float32)
    if not np.isfinite(array).all():
        raise SubspaceError(f"{name} contains NaN or Inf")
    return array


def _matrix(value: Any, *, name: str) -> np.ndarray:
    array = _float_array(value, name=name)
    if array.ndim < 2:
        raise SubspaceError(f"{name} must have shape [..., hidden], got {array.shape}")
    if array.shape[-1] == 0 or np.prod(array.shape[:-1], dtype=np.int64) == 0:
        raise SubspaceError(f"{name} must have non-empty sample and hidden axes")
    return array.reshape(-1, array.shape[-1])


def _validate_rank(rank: int, *, n_samples: int, dimension: int) -> None:
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise RankDeficiencyError("rank must be a positive integer")
    maximum = min(n_samples, dimension)
    if rank > maximum:
        raise RankDeficiencyError(
            f"rank={rank} cannot be fit from n_samples={n_samples}, dimension={dimension}; "
            f"maximum is {maximum}"
        )


def _canonicalize_basis_signs(basis: np.ndarray) -> np.ndarray:
    """Choose deterministic signs without changing the represented subspace."""

    result = basis.copy()
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        if result[pivot, column] < 0:
            result[:, column] *= -1
    return result


def _check_orthonormal(
    basis: Any, *, name: str = "basis", atol: float = 2e-5, rtol: float = 2e-5
) -> np.ndarray:
    array = _float_array(basis, name=name)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise SubspaceError(f"{name} must have shape [hidden, rank]")
    gram = array.T @ array
    identity = np.eye(array.shape[1], dtype=gram.dtype)
    if not np.allclose(gram, identity, atol=atol, rtol=rtol):
        maximum_error = float(np.max(np.abs(gram - identity)))
        raise SubspaceError(
            f"{name} is not orthonormal (maximum Gram error {maximum_error:.6g})"
        )
    return array


@dataclass(frozen=True)
class SubspaceFit:
    basis: np.ndarray
    singular_values: np.ndarray
    center: np.ndarray
    n_samples: int
    rank: int
    effective_rank: int


def fit_top_svd_subspace(
    x: Any,
    rank: int,
    *,
    center: bool = False,
    require_full_numerical_rank: bool = True,
) -> SubspaceFit:
    """Fit a top-right-singular-vector basis from training observations only."""

    matrix = _matrix(x, name="x")
    _validate_rank(rank, n_samples=matrix.shape[0], dimension=matrix.shape[1])
    fitted_center = (
        matrix.mean(axis=0) if center else np.zeros(matrix.shape[1], dtype=matrix.dtype)
    )
    working = matrix - fitted_center
    total_energy = float(np.sum(np.square(working, dtype=np.float64)))
    if not np.isfinite(total_energy) or total_energy <= np.finfo(np.float32).tiny:
        raise RankDeficiencyError("Cannot fit a subspace from zero-energy observations")
    _, singular_values, vh = np.linalg.svd(working, full_matrices=False)
    if singular_values.size == 0 or float(singular_values[0]) <= 0.0:
        raise RankDeficiencyError("SVD returned no positive singular value")
    tolerance = (
        max(working.shape)
        * np.finfo(singular_values.dtype).eps
        * float(singular_values[0])
    )
    effective_rank = int(np.count_nonzero(singular_values > tolerance))
    if require_full_numerical_rank and effective_rank < rank:
        raise RankDeficiencyError(
            f"Requested rank={rank}, but training observations have numerical "
            f"rank={effective_rank} (tolerance={tolerance:.6g})"
        )
    basis = _canonicalize_basis_signs(vh[:rank].T)
    return SubspaceFit(
        basis=basis,
        singular_values=singular_values[:rank].copy(),
        center=fitted_center,
        n_samples=int(matrix.shape[0]),
        rank=int(rank),
        effective_rank=effective_rank,
    )


def top_svd_subspace(
    x: Any,
    rank: int,
    *,
    center: bool = False,
    require_full_numerical_rank: bool = True,
) -> np.ndarray:
    """Return ``[hidden, rank]`` top-SVD basis fit from ``x``."""

    return fit_top_svd_subspace(
        x,
        rank,
        center=center,
        require_full_numerical_rank=require_full_numerical_rank,
    ).basis


def project_onto_subspace(x: Any, basis: Any) -> np.ndarray:
    values = _float_array(x, name="x")
    orthonormal = _check_orthonormal(basis)
    if values.shape[-1] != orthonormal.shape[0]:
        raise SubspaceError(
            f"x hidden dimension {values.shape[-1]} != basis dimension "
            f"{orthonormal.shape[0]}"
        )
    return (values @ orthonormal) @ orthonormal.T


def reconstruction_cosine(
    x: Any, basis: Any, *, eps: float = 1e-12
) -> np.ndarray:
    values = _float_array(x, name="x")
    reconstruction = project_onto_subspace(values, basis)
    numerator = np.sum(values * reconstruction, axis=-1)
    denominator = np.linalg.norm(values, axis=-1) * np.linalg.norm(
        reconstruction, axis=-1
    )
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=np.result_type(values, np.float32)),
        where=denominator > eps,
    )


@dataclass(frozen=True)
class ExplainedVarianceResult:
    value: float
    projected_energy: float
    total_energy: float
    n_observations: int
    valid: bool
    exclusion_reason: str | None = None


def explained_variance_details(
    x: Any, basis: Any, *, eps: float = 1e-12
) -> ExplainedVarianceResult:
    """Evaluate retained energy without fitting either a center or a basis."""

    matrix = _matrix(x, name="x")
    orthonormal = _check_orthonormal(basis)
    if matrix.shape[1] != orthonormal.shape[0]:
        raise SubspaceError("x and basis hidden dimensions differ")
    total = float(np.sum(np.square(matrix, dtype=np.float64)))
    if not np.isfinite(total) or total <= eps:
        return ExplainedVarianceResult(
            value=float("nan"),
            projected_energy=0.0,
            total_energy=total,
            n_observations=int(matrix.shape[0]),
            valid=False,
            exclusion_reason=f"total_energy <= eps ({total:.6g} <= {eps:.6g})",
        )
    coefficients = matrix @ orthonormal
    projected = float(np.sum(np.square(coefficients, dtype=np.float64)))
    value = projected / total
    # Numerical noise may exceed one by a few ulps only.
    if value < -1e-8 or value > 1.0 + 1e-6:
        raise SubspaceError(f"Explained variance outside [0, 1]: {value}")
    value = float(np.clip(value, 0.0, 1.0))
    return ExplainedVarianceResult(
        value=value,
        projected_energy=projected,
        total_energy=total,
        n_observations=int(matrix.shape[0]),
        valid=True,
    )


def explained_variance(x: Any, basis: Any, *, eps: float = 1e-12) -> float:
    """Return retained energy, failing explicitly for a degenerate denominator."""

    result = explained_variance_details(x, basis, eps=eps)
    if not result.valid:
        raise DegenerateEnergyError(result.exclusion_reason or "Invalid denominator")
    return result.value


def subspace_overlap(u: Any, v: Any) -> float:
    """Return ``||U^T V||_F^2 / r`` for equal-rank orthonormal bases."""

    left = _check_orthonormal(u, name="u")
    right = _check_orthonormal(v, name="v")
    if left.shape[0] != right.shape[0]:
        raise SubspaceError("Subspaces have different ambient dimensions")
    if left.shape[1] != right.shape[1]:
        raise RankDeficiencyError(
            "Subspace overlap requires equal ranks: "
            f"{left.shape[1]} vs {right.shape[1]}"
        )
    overlap = float(np.sum(np.square(left.T @ right, dtype=np.float64)) / left.shape[1])
    return float(np.clip(overlap, 0.0, 1.0))


def principal_angles(u: Any, v: Any) -> np.ndarray:
    """Return rotation/sign-invariant principal angles in radians."""

    left = _check_orthonormal(u, name="u")
    right = _check_orthonormal(v, name="v")
    if left.shape[0] != right.shape[0] or left.shape[1] != right.shape[1]:
        raise RankDeficiencyError("Principal angles require equal ambient size and rank")
    singular_values = np.linalg.svd(left.T @ right, compute_uv=False)
    return np.arccos(np.clip(singular_values, 0.0, 1.0))


@dataclass(frozen=True)
class AffineSubspace:
    center: np.ndarray
    basis: np.ndarray
    singular_values: np.ndarray
    train_token_indices: np.ndarray

    @property
    def rank(self) -> int:
        return int(self.basis.shape[1])

    def centered(self, z: Any) -> np.ndarray:
        values = _float_array(z, name="z")
        if values.shape[-1] != self.center.shape[0]:
            raise SubspaceError("z and affine center dimensions differ")
        return values - self.center

    def reconstruct(self, z: Any) -> np.ndarray:
        centered = self.centered(z)
        return self.center + project_onto_subspace(centered, self.basis)


def fit_local_affine_subspace(
    z: Any,
    train_token_indices: Iterable[int],
    rank: int,
) -> AffineSubspace:
    """Fit a prefix-specific affine subspace from training token branches only."""

    values = _float_array(z, name="z")
    if values.ndim != 2:
        raise SubspaceError("z must have shape [token, hidden] for one prefix")
    raw_indices = np.asarray(list(train_token_indices))
    if raw_indices.ndim != 1 or raw_indices.size == 0 or not np.issubdtype(
        raw_indices.dtype, np.integer
    ):
        raise SubspaceError("train_token_indices must be a non-empty integer sequence")
    indices = raw_indices.astype(np.int64, copy=False)
    if len(np.unique(indices)) != len(indices):
        raise SubspaceError("train_token_indices contains duplicates")
    if int(indices.min()) < 0 or int(indices.max()) >= values.shape[0]:
        raise SubspaceError("train_token_indices out of bounds")
    training = values[indices]
    fit = fit_top_svd_subspace(training, rank, center=True)
    return AffineSubspace(
        center=fit.center,
        basis=fit.basis,
        singular_values=fit.singular_values,
        train_token_indices=indices.copy(),
    )


def affine_explained_variance(
    z: Any, affine_subspace: AffineSubspace, *, eps: float = 1e-12
) -> float:
    return explained_variance(affine_subspace.centered(z), affine_subspace.basis, eps=eps)


def orientation_only_explained_variance(
    z: Any, evaluation_center: Any, comparison_basis: Any, *, eps: float = 1e-12
) -> float:
    """Compare directions while holding the evaluation prefix's center fixed."""

    values = _float_array(z, name="z")
    center = _float_array(evaluation_center, name="evaluation_center")
    if center.ndim != 1 or center.shape[0] != values.shape[-1]:
        raise SubspaceError("evaluation_center must have shape [hidden]")
    return explained_variance(values - center, comparison_basis, eps=eps)


def fit_prefix_subspaces(
    responses: Any,
    train_token_indices: Iterable[int],
    rank: int,
    *,
    center: bool = False,
) -> list[SubspaceFit]:
    """Fit one equal-rank space per prefix from training token columns only."""

    array = _float_array(responses, name="responses")
    if array.ndim != 3:
        raise SubspaceError("responses must have shape [prefix, token, hidden]")
    indices = np.asarray(list(train_token_indices))
    if indices.ndim != 1 or indices.size == 0 or not np.issubdtype(
        indices.dtype, np.integer
    ):
        raise SubspaceError("train_token_indices must be non-empty integers")
    indices = indices.astype(np.int64, copy=False)
    if len(np.unique(indices)) != len(indices):
        raise SubspaceError("train_token_indices contains duplicates")
    if int(indices.min()) < 0 or int(indices.max()) >= array.shape[1]:
        raise SubspaceError("train_token_indices out of bounds")
    return [
        fit_top_svd_subspace(prefix[indices], rank, center=center)
        for prefix in array
    ]


def global_subspace(x: Any, rank: int, *, center: bool = False) -> np.ndarray:
    """Fit the ordinary global space using every supplied training response."""

    return top_svd_subspace(x, rank, center=center)


def sample_matched_global_subspace(
    x: Any,
    rank: int,
    *,
    n_samples: int,
    seed: int,
    center: bool = False,
) -> np.ndarray:
    """Fit a global control from exactly ``n_samples`` training vectors."""

    matrix = _matrix(x, name="x")
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples <= 0:
        raise SubspaceError("n_samples must be a positive integer")
    if n_samples > matrix.shape[0]:
        raise RankDeficiencyError(
            f"Requested {n_samples} sample-matched rows from a pool of {matrix.shape[0]}"
        )
    selected = np.random.default_rng(seed).choice(
        matrix.shape[0], size=n_samples, replace=False
    )
    return top_svd_subspace(matrix[selected], rank, center=center)


def random_orthonormal_subspace(
    dimension: int,
    rank: int,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw a uniform orthonormal basis in the supplied coordinate system."""

    if dimension <= 0:
        raise SubspaceError("dimension must be positive")
    _validate_rank(rank, n_samples=dimension, dimension=dimension)
    if rng is not None and seed is not None:
        raise SubspaceError("Pass either rng or seed, not both")
    generator = rng if rng is not None else np.random.default_rng(seed)
    q, r = np.linalg.qr(generator.standard_normal((dimension, rank)), mode="reduced")
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    return _canonicalize_basis_signs(q * signs[None, :])


def split_half_subspaces(
    x: Any,
    rank: int,
    *,
    seed: int,
    center: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit independent same-prefix bases from two disjoint token samples."""

    matrix = _matrix(x, name="x")
    if matrix.shape[0] < 2 * rank:
        raise RankDeficiencyError(
            f"Need at least 2*rank={2 * rank} rows for split-half fitting, "
            f"got {matrix.shape[0]}"
        )
    order = np.random.default_rng(seed).permutation(matrix.shape[0])
    half = matrix.shape[0] // 2
    first_indices = np.sort(order[:half])
    second_indices = np.sort(order[half:])
    first = top_svd_subspace(matrix[first_indices], rank, center=center)
    second = top_svd_subspace(matrix[second_indices], rank, center=center)
    return first, second, first_indices, second_indices


@dataclass(frozen=True)
class WhiteningTransform:
    """Compact symmetric inverse-square-root covariance transform."""

    mean: np.ndarray
    eigenvectors: np.ndarray
    eigenvalues: np.ndarray
    clipped_eigenvalues: np.ndarray
    eigenvalue_floor: float
    n_samples: int
    ddof: int

    @property
    def dimension(self) -> int:
        return int(self.mean.shape[0])

    def _linear(self, x: np.ndarray, *, inverse: bool) -> np.ndarray:
        if x.shape[-1] != self.dimension:
            raise SubspaceError(
                f"x dimension {x.shape[-1]} != whitener dimension {self.dimension}"
            )
        original_shape = x.shape
        matrix = x.reshape(-1, self.dimension)
        if inverse:
            floor_scale = np.sqrt(self.eigenvalue_floor)
            retained_scales = np.sqrt(self.clipped_eigenvalues)
        else:
            floor_scale = 1.0 / np.sqrt(self.eigenvalue_floor)
            retained_scales = 1.0 / np.sqrt(self.clipped_eigenvalues)
        # The compact eigensystem applies retained-direction corrections to an
        # isotropic floor transform in the orthogonal complement.
        transformed = matrix * floor_scale
        if self.eigenvectors.shape[1]:
            coefficients = matrix @ self.eigenvectors
            corrections = retained_scales - floor_scale
            transformed = transformed + (coefficients * corrections) @ self.eigenvectors.T
        return transformed.reshape(original_shape)

    def transform(self, x: Any, *, center: bool = False) -> np.ndarray:
        """Apply ``C^-1/2``; set center=True for natural activations.

        Difference vectors and interaction residuals should use the default
        ``center=False`` because their origin is already meaningful.
        """

        values = _float_array(x, name="x")
        if center:
            values = values - self.mean
        return self._linear(values, inverse=False)

    def inverse_transform(self, x: Any, *, add_mean: bool = False) -> np.ndarray:
        values = _float_array(x, name="x")
        result = self._linear(values, inverse=True)
        return result + self.mean if add_mean else result


def fit_whitener(
    x: Any,
    eigenvalue_floor_ratio: float = 1.0e-5,
    *,
    max_rank: int | None = None,
    ddof: int = 1,
) -> WhiteningTransform:
    """Estimate a stable covariance transform from independent auxiliary data."""

    matrix = _matrix(x, name="x")
    if matrix.shape[0] <= ddof:
        raise RankDeficiencyError(
            f"Need more than ddof={ddof} auxiliary observations, got {matrix.shape[0]}"
        )
    if (
        not np.isfinite(eigenvalue_floor_ratio)
        or eigenvalue_floor_ratio <= 0.0
        or eigenvalue_floor_ratio > 1.0
    ):
        raise SubspaceError("eigenvalue_floor_ratio must be in (0, 1]")
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    eigenvalues = np.square(singular_values, dtype=np.float64) / (
        matrix.shape[0] - ddof
    )
    if eigenvalues.size == 0 or not np.isfinite(eigenvalues[0]) or eigenvalues[0] <= 0:
        raise RankDeficiencyError("Auxiliary covariance has zero total variance")
    floor = float(eigenvalues[0] * eigenvalue_floor_ratio)
    if max_rank is None:
        retained = len(eigenvalues)
    else:
        if isinstance(max_rank, bool) or not isinstance(max_rank, int) or max_rank <= 0:
            raise SubspaceError("max_rank must be a positive integer or None")
        retained = min(max_rank, len(eigenvalues), matrix.shape[1])
    values = eigenvalues[:retained]
    vectors = vh[:retained].T
    clipped = np.maximum(values, floor)
    return WhiteningTransform(
        mean=mean,
        eigenvectors=vectors,
        eigenvalues=values,
        clipped_eigenvalues=clipped,
        eigenvalue_floor=floor,
        n_samples=int(matrix.shape[0]),
        ddof=int(ddof),
    )


def apply_whitening(
    x: Any, whitener: WhiteningTransform, *, center: bool = False
) -> np.ndarray:
    return whitener.transform(x, center=center)


def whiten_basis(basis: Any, whitener: WhiteningTransform) -> np.ndarray:
    """Map a basis to whitened coordinates and re-orthonormalize it."""

    original = _check_orthonormal(basis)
    mapped = whitener.transform(original.T, center=False).T
    q, _ = np.linalg.qr(mapped, mode="reduced")
    return _canonicalize_basis_signs(q)


# Backward-friendly aliases with explicit scientific semantics.
fit_global_subspace = global_subspace
fit_sample_matched_global_subspace = sample_matched_global_subspace
overlap = subspace_overlap


__all__ = [
    "AffineSubspace",
    "DegenerateEnergyError",
    "ExplainedVarianceResult",
    "RankDeficiencyError",
    "SubspaceError",
    "SubspaceFit",
    "WhiteningTransform",
    "affine_explained_variance",
    "apply_whitening",
    "explained_variance",
    "explained_variance_details",
    "fit_global_subspace",
    "fit_local_affine_subspace",
    "fit_prefix_subspaces",
    "fit_sample_matched_global_subspace",
    "fit_top_svd_subspace",
    "fit_whitener",
    "global_subspace",
    "orientation_only_explained_variance",
    "overlap",
    "principal_angles",
    "project_onto_subspace",
    "random_orthonormal_subspace",
    "reconstruction_cosine",
    "sample_matched_global_subspace",
    "split_half_subspaces",
    "subspace_overlap",
    "top_svd_subspace",
    "whiten_basis",
]
