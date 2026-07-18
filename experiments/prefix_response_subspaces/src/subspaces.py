from __future__ import annotations

import numpy as np


class RankError(ValueError):
    pass


def top_svd(matrix: np.ndarray, rank: int, *, allow_rank_reduction: bool = False) -> np.ndarray:
    samples = np.asarray(matrix, dtype=np.float64)
    if samples.ndim != 2 or not np.isfinite(samples).all():
        raise ValueError("SVD samples must be a finite matrix")
    if rank <= 0 or (rank > min(samples.shape) and not allow_rank_reduction):
        raise RankError(f"requested rank {rank}, maximum shape rank {min(samples.shape)}")
    wide = samples.shape[0] <= samples.shape[1]
    if wide:
        # Local response matrices are typically 192 x 1536.  Solving the
        # 192 x 192 Gram eigenproblem is exact for the right singular subspace
        # and substantially cheaper than a full rectangular SVD.
        eigenvalues,eigenvectors=np.linalg.eigh(samples@samples.T)
        order=np.argsort(eigenvalues)[::-1]; eigenvalues=np.clip(eigenvalues[order],0.0,None); eigenvectors=eigenvectors[:,order]
        singular_values=np.sqrt(eigenvalues)
    else:
        _, singular_values, vt = np.linalg.svd(samples, full_matrices=False)
    if wide:
        tolerance=(eigenvalues[0]*max(samples.shape)*np.finfo(samples.dtype).eps) if len(eigenvalues) else 0.0
        numerical_rank=int(np.sum(eigenvalues>tolerance))
    else:
        tolerance = (singular_values[0] * max(samples.shape) * np.finfo(samples.dtype).eps) if len(singular_values) else 0.0
        numerical_rank = int(np.sum(singular_values > tolerance))
    effective_rank = min(int(rank), numerical_rank) if allow_rank_reduction else int(rank)
    if effective_rank <= 0 or (numerical_rank < rank and not allow_rank_reduction):
        raise RankError(f"requested rank {rank}, available rank {numerical_rank}")
    if wide:
        basis=samples.T@eigenvectors[:,:effective_rank]
        basis/=singular_values[:effective_rank][None,:]
        basis=np.linalg.qr(basis,mode="reduced")[0]
    else:
        basis = vt[:effective_rank].T
    if not np.allclose(basis.T @ basis, np.eye(effective_rank), atol=2e-7):
        raise RuntimeError("SVD basis is not orthonormal")
    return basis


def explained_variance(samples: np.ndarray, basis: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(samples, dtype=np.float64)
    u = np.asarray(basis, dtype=np.float64)
    if x.ndim != 2 or u.ndim != 2 or x.shape[1] != u.shape[0]:
        raise ValueError("incompatible sample and basis shapes")
    denominator = float(np.square(x).sum())
    if denominator <= eps:
        return float("nan")
    return float(np.square(x @ u).sum() / denominator)


def principal_angle_cosines_squared(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return cos² principal angles in descending-alignment order."""
    u = np.asarray(left, dtype=np.float64)
    v = np.asarray(right, dtype=np.float64)
    if u.ndim != 2 or v.ndim != 2 or u.shape[0] != v.shape[0]:
        raise ValueError("bases must have compatible [hidden, rank] shapes")
    rank = min(u.shape[1], v.shape[1])
    if rank == 0:
        return np.empty(0, dtype=np.float64)
    singular = np.linalg.svd(u[:, :rank].T @ v[:, :rank], compute_uv=False)
    return np.clip(np.square(singular), 0.0, 1.0)


def normalized_projection_distance(left: np.ndarray, right: np.ndarray) -> float:
    cos2 = principal_angle_cosines_squared(left, right)
    return float(1.0 - cos2.mean()) if len(cos2) else float("nan")


def content_subspace(prefix_hidden: np.ndarray, rank: int, *, excluded_positions: set[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    hidden = np.asarray(prefix_hidden, dtype=np.float64)
    if hidden.ndim != 2:
        raise ValueError("prefix hidden states must have shape [position, hidden]")
    excluded = set(excluded_positions or ())
    positions = np.asarray([i for i in range(len(hidden)) if i not in excluded], dtype=np.int64)
    if len(positions) < 2:
        raise RankError("too few content positions after BOS/sink exclusion")
    centered = hidden[positions] - hidden[positions].mean(axis=0, keepdims=True)
    return top_svd(centered, rank, allow_rank_reduction=True), positions


def remove_directions(values: np.ndarray, directions: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    u = np.asarray(directions, dtype=np.float64)
    return x - (x @ u) @ u.T


def mean_projection_eigensystem(bases: list[np.ndarray], eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Eigenpairs of ``mean_i U_i U_i.T`` for orthonormal local bases.

    The eigenvalues lie in [0, 1] up to numerical error and quantify the
    fraction of fitted local projectors containing each shared direction.
    """
    if not bases:
        raise ValueError("at least one local basis is required")
    prepared = [np.asarray(basis, dtype=np.float64) for basis in bases]
    hidden = prepared[0].shape[0]
    if any(basis.ndim != 2 or basis.shape[0] != hidden for basis in prepared):
        raise ValueError("all local bases must have shape [hidden, rank]")
    projector = np.zeros((hidden, hidden), dtype=np.float64)
    for basis in prepared:
        gram = basis.T @ basis
        if not np.allclose(gram, np.eye(basis.shape[1]), atol=2e-7):
            raise ValueError("local bases must be orthonormal")
        projector += basis @ basis.T
    projector /= len(prepared)
    eigenvalues, eigenvectors = np.linalg.eigh(projector)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0.0, 1.0)
    keep = eigenvalues > eps
    return eigenvalues, eigenvectors[:, order][:, keep]


def randomized_mean_projection_eigensystem(
    bases: list[np.ndarray], components: int, *, seed: int, oversampling: int = 16, power_iterations: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Leading eigenpairs of the mean projector without forming hidden².

    If ``A=[U_1 ... U_N]/sqrt(N)``, the target is ``A A.T``.  A deterministic
    randomized range finder makes the paper-scale 1536 x 1536 decomposition
    substantially cheaper while preserving the leading shared directions.
    """
    if not bases:
        raise ValueError("at least one local basis is required")
    prepared = [np.asarray(basis, dtype=np.float32) for basis in bases]
    hidden = prepared[0].shape[0]
    if any(basis.ndim != 2 or basis.shape[0] != hidden for basis in prepared):
        raise ValueError("all local bases must have shape [hidden, rank]")
    target = min(int(components), hidden, sum(basis.shape[1] for basis in prepared))
    if target <= 0:
        raise ValueError("components must be positive")
    q = min(hidden, sum(basis.shape[1] for basis in prepared), target + int(oversampling))
    stack = np.concatenate(prepared, axis=1) / np.sqrt(float(len(prepared)))
    rng = np.random.default_rng(int(seed))
    omega = rng.normal(size=(hidden, q)).astype(np.float32)
    sample = stack @ (stack.T @ omega)
    basis_q = np.linalg.qr(sample, mode="reduced")[0]
    for _ in range(int(power_iterations)):
        sample = stack @ (stack.T @ basis_q)
        basis_q = np.linalg.qr(sample, mode="reduced")[0]
    compressed = basis_q.T @ stack
    left, singular_values, _ = np.linalg.svd(compressed, full_matrices=False)
    eigenvectors = basis_q @ left[:, :target]
    eigenvalues = np.clip(np.square(singular_values[:target], dtype=np.float64), 0.0, 1.0)
    return eigenvalues, np.asarray(eigenvectors[:, :target], dtype=np.float64)


def remove_shared_subspace(local_basis: np.ndarray, shared_basis: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Orthonormalize the component of a local basis outside a shared space."""
    local = np.asarray(local_basis, dtype=np.float64)
    shared = np.asarray(shared_basis, dtype=np.float64)
    if local.ndim != 2 or shared.ndim != 2 or local.shape[0] != shared.shape[0]:
        raise ValueError("local and shared bases must have compatible hidden axes")
    residual = local - shared @ (shared.T @ local) if shared.shape[1] else local.copy()
    if not residual.size:
        return np.empty((local.shape[0], 0), dtype=np.float64)
    gram = residual.T @ residual
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0.0, None)
    eigenvectors = eigenvectors[:, order]
    singular_values = np.sqrt(eigenvalues)
    if not len(singular_values):
        return np.empty((local.shape[0], 0), dtype=np.float64)
    tolerance = max(float(eps), float(singular_values[0]) * max(residual.shape) * np.finfo(np.float64).eps)
    rank = int(np.sum(singular_values > tolerance))
    if rank <= 0:
        return np.empty((local.shape[0], 0), dtype=np.float64)
    basis = residual @ eigenvectors[:, :rank]
    basis /= singular_values[:rank][None, :]
    return np.linalg.qr(basis, mode="reduced")[0]
