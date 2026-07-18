"""Shared-basis covariance models for candidate-response residuals.

The implementation never materializes a hidden-size by hidden-size covariance.
For ``R_i`` with shape ``[candidate, hidden]`` it evaluates the Frobenius
reconstruction objective through candidate-space Gram matrices and projected
variances.  This keeps the paper-scale calculation practical while remaining
algebraically identical to fitting ``S_i = R_i.T @ R_i / n``.
"""
from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np


ModelKind = Literal["cpc", "dictionary"]


@dataclass(frozen=True)
class OptimizationConfig:
    learning_rate: float = 0.02
    maximum_steps: int = 2000
    patience: int = 200
    restarts: int = 5
    seed: int = 0
    coherence_penalty: float = 0.0
    epsilon: float = 1e-12
    improvement_tolerance: float = 1e-9
    initialization_noise: float = 0.02
    context_batch_size: int = 0
    full_loss_interval: int = 25

    def validate(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.maximum_steps <= 0 or self.patience <= 0:
            raise ValueError("maximum_steps and patience must be positive")
        if self.restarts < 1:
            raise ValueError("restarts must be positive")
        if self.coherence_penalty < 0:
            raise ValueError("coherence_penalty must be non-negative")
        if self.epsilon <= 0 or self.improvement_tolerance < 0:
            raise ValueError("epsilon must be positive and improvement_tolerance non-negative")
        if self.context_batch_size < 0 or self.full_loss_interval <= 0:
            raise ValueError("context_batch_size must be non-negative and full_loss_interval positive")


@dataclass
class SharedCovarianceFit:
    kind: ModelKind
    basis: Any
    weights: Any
    loss: float
    selected_restart: int
    diagnostics: dict[str, Any]


def torch_dtype(name: str):
    import torch

    values = {"float32": torch.float32, "float64": torch.float64}
    if name not in values:
        raise ValueError(f"unsupported dtype {name!r}; choose float32 or float64")
    return values[name]


def as_residual_tensor(residuals: np.ndarray | Any, *, device: str, dtype: str):
    import torch

    value = torch.as_tensor(residuals, device=device, dtype=torch_dtype(dtype))
    if value.ndim != 3 or min(value.shape) <= 0:
        raise ValueError("residuals must have shape [context, candidate, hidden]")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("residuals contain NaN or Inf")
    return value


def covariance_frobenius_squared(residuals):
    """Return ``||R.T R / n||_F^2`` for each context."""
    import torch

    if residuals.ndim != 3:
        raise ValueError("residuals must be three-dimensional")
    candidate_count = residuals.shape[1]
    gram = torch.bmm(residuals, residuals.transpose(1, 2)) / float(candidate_count)
    return torch.square(gram).sum(dim=(1, 2))


def covariance_reconstruction_errors(
    residuals, basis, weights, *, kind: ModelKind, epsilon: float = 1e-12, covariance_norm=None,
):
    """Return normalized per-context covariance reconstruction errors."""
    import torch

    if residuals.ndim != 3 or basis.ndim != 2 or weights.ndim != 2:
        raise ValueError("invalid residual, basis, or weight dimensions")
    if residuals.shape[0] != weights.shape[0] or residuals.shape[2] != basis.shape[0] or basis.shape[1] != weights.shape[1]:
        raise ValueError("incompatible residual, basis, and weight shapes")
    if not bool(torch.isfinite(residuals).all() and torch.isfinite(basis).all() and torch.isfinite(weights).all()):
        raise ValueError("reconstruction inputs contain NaN or Inf")
    sample_count = float(residuals.shape[1])
    projected = torch.einsum("cnd,dk->cnk", residuals, basis)
    projected_variance = torch.square(projected).sum(dim=1) / sample_count
    if covariance_norm is None:
        covariance_norm = covariance_frobenius_squared(residuals)
    elif covariance_norm.shape != (residuals.shape[0],):
        raise ValueError("covariance_norm must have one value per context")
    cross = 2.0 * (weights * projected_variance).sum(dim=1)
    if kind == "cpc":
        model_norm = torch.square(weights).sum(dim=1)
    elif kind == "dictionary":
        squared_gram = torch.square(basis.transpose(0, 1) @ basis)
        model_norm = torch.einsum("ck,kl,cl->c", weights, squared_gram, weights)
    else:
        raise ValueError(f"unknown model kind: {kind}")
    errors = (covariance_norm - cross + model_norm) / (covariance_norm + float(epsilon))
    return torch.clamp(errors, min=0.0)


def _inverse_softplus(values):
    import torch

    clipped = torch.clamp(values, min=torch.finfo(values.dtype).eps)
    return clipped + torch.log(-torch.expm1(-clipped))


def _normalize_columns(value, epsilon: float):
    import torch

    norms = torch.linalg.vector_norm(value, dim=0, keepdim=True)
    if not bool(torch.isfinite(norms).all()) or bool((norms <= epsilon).any()):
        raise RuntimeError("dictionary column normalization failed")
    return value / norms


def _orthogonalize(value):
    import torch

    try:
        basis, _ = torch.linalg.qr(value, mode="reduced")
    except RuntimeError as exc:
        raise RuntimeError(f"QR decomposition failed: {exc}") from exc
    if not bool(torch.isfinite(basis).all()):
        raise RuntimeError("QR decomposition returned NaN or Inf")
    return basis


def pooled_pca_basis(residuals, dictionary_size: int):
    """Top pooled covariance directions, initialized without evaluation data."""
    import torch

    if residuals.ndim != 3:
        raise ValueError("residuals must have shape [context, candidate, hidden]")
    maximum = min(int(residuals.shape[0] * residuals.shape[1]), int(residuals.shape[2]))
    if dictionary_size <= 0 or dictionary_size > maximum:
        raise ValueError(f"dictionary_size {dictionary_size} exceeds pooled rank bound {maximum}")
    flattened = residuals.reshape(-1, residuals.shape[-1])
    try:
        covariance = flattened.transpose(0, 1) @ flattened / float(flattened.shape[0])
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    except RuntimeError as exc:
        raise RuntimeError(f"pooled covariance eigendecomposition failed: {exc}") from exc
    order = torch.argsort(eigenvalues, descending=True)[:dictionary_size]
    basis = eigenvectors[:, order].contiguous()
    if not bool(torch.isfinite(basis).all()):
        raise RuntimeError("pooled covariance eigendecomposition returned NaN or Inf")
    return basis


def context_projection_variances(residuals, basis):
    import torch

    projected = torch.einsum("cnd,dk->cnk", residuals, basis)
    return torch.square(projected).mean(dim=1)


def select_pooled_subspace(residuals, pooled_basis, context_index: int, rank: int):
    import torch

    if rank <= 0 or rank > pooled_basis.shape[1]:
        raise ValueError("rank must be between one and dictionary size")
    variance = context_projection_variances(residuals[context_index : context_index + 1], pooled_basis)[0]
    indices = torch.argsort(variance, descending=True)[:rank]
    return pooled_basis[:, indices], indices


def covariance_subspace(basis, weights, rank: int, *, kind: ModelKind):
    """Return the requested subspace of ``B diag(lambda) B.T``."""
    import torch

    if weights.ndim != 1 or basis.shape[1] != weights.shape[0]:
        raise ValueError("basis and weights are incompatible")
    if rank <= 0 or rank > basis.shape[1]:
        raise ValueError("rank must be between one and dictionary size")
    if bool((weights < 0).any()) or not bool(torch.isfinite(weights).all()):
        raise ValueError("weights must be finite and non-negative")
    if kind == "cpc":
        indices = torch.argsort(weights, descending=True)[:rank]
        return basis[:, indices], indices
    if kind != "dictionary":
        raise ValueError(f"unknown model kind: {kind}")
    weighted = basis * torch.sqrt(torch.clamp(weights, min=0.0))[None, :]
    try:
        left, singular, _ = torch.linalg.svd(weighted, full_matrices=False)
    except RuntimeError as exc:
        raise RuntimeError(f"estimated-covariance eigenspace failed: {exc}") from exc
    tolerance = singular.max() * max(weighted.shape) * torch.finfo(singular.dtype).eps
    numerical_rank = int((singular > tolerance).sum().detach().cpu())
    if numerical_rank < rank:
        raise RuntimeError(
            f"estimated covariance has numerical rank {numerical_rank}, below requested rank {rank}"
        )
    subspace = left[:, :rank]
    if not bool(torch.isfinite(subspace).all()):
        raise RuntimeError("estimated-covariance eigenspace contains NaN or Inf")
    return subspace, None


def estimated_covariance(basis, weights):
    import torch

    result = (basis * weights[None, :]) @ basis.transpose(0, 1)
    return 0.5 * (result + result.transpose(0, 1))


def _model_state(raw_basis, raw_weights, kind: ModelKind, epsilon: float):
    import torch

    basis = _orthogonalize(raw_basis) if kind == "cpc" else _normalize_columns(raw_basis, epsilon)
    weights = torch.nn.functional.softplus(raw_weights)
    return basis, weights


def _loss(residuals, raw_basis, raw_weights, kind: ModelKind, config: OptimizationConfig, covariance_norm=None):
    import torch

    basis, weights = _model_state(raw_basis, raw_weights, kind, config.epsilon)
    reconstruction = covariance_reconstruction_errors(
        residuals, basis, weights, kind=kind, epsilon=config.epsilon, covariance_norm=covariance_norm,
    ).mean()
    if kind == "dictionary" and config.coherence_penalty:
        identity = torch.eye(basis.shape[1], device=basis.device, dtype=basis.dtype)
        coherence = torch.square(basis.transpose(0, 1) @ basis - identity).sum() / float(basis.shape[1] ** 2)
        reconstruction = reconstruction + config.coherence_penalty * coherence
    return reconstruction, basis, weights


def _gradient_norm(parameters) -> float:
    total = None
    for parameter in parameters:
        if parameter.grad is not None:
            squared = parameter.grad.detach().double().square().sum()
            total = squared if total is None else total + squared
    return 0.0 if total is None else math.sqrt(float(total.cpu()))


def _basis_diagnostics(basis, kind: ModelKind) -> dict[str, float | int]:
    import torch

    gram = basis.transpose(0, 1) @ basis
    identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    if kind == "cpc":
        return {"orthogonality_error_frobenius": float(torch.linalg.matrix_norm(gram - identity).detach().cpu())}
    off_diagonal = gram - torch.diag(torch.diagonal(gram))
    singular = torch.linalg.svdvals(basis)
    squared = torch.square(singular)
    probabilities = squared / torch.clamp(squared.sum(), min=torch.finfo(squared.dtype).eps)
    entropy = -(probabilities * torch.log(torch.clamp(probabilities, min=torch.finfo(probabilities.dtype).eps))).sum()
    tolerance = singular.max() * max(basis.shape) * torch.finfo(singular.dtype).eps
    return {
        "maximum_column_coherence": float(torch.abs(off_diagonal).max().detach().cpu()),
        "gram_identity_error_frobenius": float(torch.linalg.matrix_norm(gram - identity).detach().cpu()),
        "numerical_rank": int((singular > tolerance).sum().detach().cpu()),
        "entropy_effective_rank": float(torch.exp(entropy).detach().cpu()),
    }


def _make_data_parallel_objective(raw_basis, raw_weights, kind: ModelKind, config: OptimizationConfig):
    """Build the shared-parameter objective wrapped by ``torch.nn.DataParallel``."""
    import torch

    class _SharedCovarianceObjective(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.raw_basis = torch.nn.Parameter(raw_basis)
            self.raw_weights = torch.nn.Parameter(raw_weights)

        def forward(self, residuals, context_indices, covariance_norm):
            basis, all_weights = _model_state(
                self.raw_basis, self.raw_weights, kind, config.epsilon,
            )
            weights = all_weights.index_select(0, context_indices)
            errors = covariance_reconstruction_errors(
                residuals, basis, weights, kind=kind, epsilon=config.epsilon,
                covariance_norm=covariance_norm,
            )
            if kind == "dictionary" and config.coherence_penalty:
                identity = torch.eye(basis.shape[1], device=basis.device, dtype=basis.dtype)
                coherence = torch.square(basis.transpose(0, 1) @ basis - identity).sum()
                errors = errors + config.coherence_penalty * coherence / float(basis.shape[1] ** 2)
            return errors

        def constrained_state(self):
            return _model_state(
                self.raw_basis, self.raw_weights, kind, config.epsilon,
            )

    return _SharedCovarianceObjective()


def _fit_data_parallel_restarts(
    residuals,
    pooled,
    initial_raw_weights,
    full_covariance_norm,
    *,
    kind: ModelKind,
    config: OptimizationConfig,
    device_ids: list[int],
) -> tuple[list[dict[str, Any]], list[tuple[float, int, Any, Any, dict[str, Any]]]]:
    """Fit every restart with context batches scattered across CUDA devices."""
    import torch

    if len(device_ids) < 2:
        raise ValueError("DataParallel requires at least two CUDA device IDs")
    if residuals.device.type != "cuda" or residuals.device.index != int(device_ids[0]):
        raise ValueError("DataParallel residuals must live on the first configured CUDA device")
    if len(set(device_ids)) != len(device_ids):
        raise ValueError("DataParallel device IDs must be unique")
    visible = torch.cuda.device_count()
    if any(device < 0 or device >= visible for device in device_ids):
        raise ValueError(f"DataParallel device IDs {device_ids} exceed {visible} visible CUDA devices")
    context_count = int(residuals.shape[0])
    batch_size = context_count if config.context_batch_size <= 0 else min(context_count, config.context_batch_size)
    if batch_size < len(device_ids):
        raise ValueError("context_batch_size must be at least the DataParallel device count")
    restart_rows: list[dict[str, Any]] = []
    successful: list[tuple[float, int, Any, Any, dict[str, Any]]] = []
    primary = torch.device(f"cuda:{device_ids[0]}")
    device_label = "DataParallel(" + ",".join(f"cuda:{item}" for item in device_ids) + ")"

    for restart in range(config.restarts):
        restart_started = time.monotonic()
        print(f"[shared_dictionary] RESTART start kind={kind} restart={restart} device={device_label}", flush=True)
        generator = torch.Generator(device=primary)
        generator.manual_seed(int(config.seed) + 10007 * restart + (0 if kind == "cpc" else 1000003))
        try:
            noise_scale = config.initialization_noise
            noise = torch.randn(pooled.shape, generator=generator, device=primary, dtype=pooled.dtype)
            basis_start = pooled.clone() + noise_scale * noise
            weight_noise = torch.randn(
                initial_raw_weights.shape, generator=generator, device=primary, dtype=pooled.dtype,
            )
            weights_start = initial_raw_weights.clone() + noise_scale * weight_noise
            objective = _make_data_parallel_objective(basis_start, weights_start, kind, config).to(primary)
            parallel = torch.nn.DataParallel(
                objective, device_ids=device_ids, output_device=device_ids[0], dim=0,
            )
            optimizer = torch.optim.Adam(parallel.module.parameters(), lr=config.learning_rate)
            best_loss = float("inf")
            best_step = -1
            best_basis = None
            best_weights = None
            stale = 0
            last_gradient_norm = float("nan")
            all_indices = torch.arange(context_count, device=primary, dtype=torch.long)
            for step in range(config.maximum_steps):
                optimizer.zero_grad(set_to_none=True)
                measure_full = step == 0 or (step + 1) % config.full_loss_interval == 0 or step + 1 == config.maximum_steps
                if batch_size < context_count:
                    batch_indices = torch.randint(
                        0, context_count, (batch_size,), generator=generator, device=primary,
                    )
                    batch_residuals = residuals.index_select(0, batch_indices)
                    batch_covariance_norm = full_covariance_norm.index_select(0, batch_indices)
                else:
                    batch_indices = all_indices
                    batch_residuals = residuals
                    batch_covariance_norm = full_covariance_norm
                losses = parallel(batch_residuals, batch_indices, batch_covariance_norm)
                loss = losses.mean()
                if measure_full and not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"non-finite DataParallel training loss at step {step}")
                loss.backward()
                if measure_full:
                    last_gradient_norm = _gradient_norm(parallel.module.parameters())
                    if not math.isfinite(last_gradient_norm):
                        raise FloatingPointError(f"non-finite DataParallel gradient norm at step {step}")
                optimizer.step()
                if measure_full:
                    with torch.no_grad():
                        current_loss = parallel(residuals, all_indices, full_covariance_norm).mean()
                        full_basis, full_weights = parallel.module.constrained_state()
                    if not bool(torch.isfinite(current_loss)):
                        raise FloatingPointError(f"non-finite full DataParallel training loss at step {step}")
                    current = float(current_loss.detach().cpu())
                    if current < best_loss - config.improvement_tolerance:
                        best_loss = current
                        best_step = step
                        best_basis = full_basis.detach().clone()
                        best_weights = full_weights.detach().clone()
                        stale = 0
                    else:
                        stale += config.full_loss_interval
                if stale >= config.patience:
                    break
            if best_basis is None or best_weights is None:
                raise RuntimeError("DataParallel optimizer did not produce a finite state")
            diagnostics = {
                "restart": restart,
                "restart_device": device_label,
                "parallelism": "torch.nn.DataParallel",
                "data_parallel_device_ids": list(map(int, device_ids)),
                "status": "success",
                "training_loss": best_loss,
                "convergence_step": best_step,
                "steps_executed": step + 1,
                "elapsed_seconds": time.monotonic() - restart_started,
                "final_gradient_norm": last_gradient_norm,
                "context_batch_size": batch_size,
                "full_training_loss_interval": config.full_loss_interval,
                **_basis_diagnostics(best_basis, kind),
            }
            restart_rows.append(diagnostics)
            successful.append((best_loss, restart, best_basis, best_weights, diagnostics))
            print(
                f"[shared_dictionary] RESTART done kind={kind} restart={restart} "
                f"steps={step + 1} loss={best_loss:.8g} elapsed={diagnostics['elapsed_seconds']:.1f}s",
                flush=True,
            )
        except (FloatingPointError, RuntimeError, ValueError) as exc:
            restart_rows.append({
                "restart": restart,
                "restart_device": device_label,
                "parallelism": "torch.nn.DataParallel",
                "data_parallel_device_ids": list(map(int, device_ids)),
                "status": "failed",
                "failure_type": type(exc).__name__,
                "failure": str(exc),
            })
            print(f"[shared_dictionary] RESTART failed kind={kind} restart={restart}: {exc}", flush=True)
        finally:
            if "parallel" in locals():
                del parallel
            if "objective" in locals():
                del objective
            torch.cuda.empty_cache()
    return restart_rows, successful


def fit_shared_covariance_model(
    residuals,
    dictionary_size: int,
    *,
    kind: ModelKind,
    config: OptimizationConfig,
    pooled_basis=None,
    restart_devices: list[str] | tuple[str, ...] | None = None,
    data_parallel_device_ids: list[int] | tuple[int, ...] | None = None,
) -> SharedCovarianceFit:
    """Fit CPC or a non-orthogonal shared dictionary using training rows only."""
    import torch

    config.validate()
    if kind not in {"cpc", "dictionary"}:
        raise ValueError(f"unknown model kind: {kind}")
    if residuals.ndim != 3 or not bool(torch.isfinite(residuals).all()):
        raise ValueError("residuals must be a finite [context, candidate, hidden] tensor")
    if dictionary_size > residuals.shape[2] or dictionary_size <= 0:
        raise ValueError("dictionary_size must be within the hidden width")
    pooled = pooled_pca_basis(residuals, dictionary_size) if pooled_basis is None else pooled_basis
    if pooled.shape != (residuals.shape[2], dictionary_size):
        raise ValueError("pooled_basis has an unexpected shape")
    initial_weights = torch.clamp(context_projection_variances(residuals, pooled), min=config.epsilon)
    initial_raw_weights = _inverse_softplus(initial_weights)
    full_covariance_norm = covariance_frobenius_squared(residuals).detach()
    context_count = int(residuals.shape[0])
    batch_size = context_count if config.context_batch_size <= 0 else min(context_count, config.context_batch_size)
    if data_parallel_device_ids:
        restart_rows, successful = _fit_data_parallel_restarts(
            residuals, pooled, initial_raw_weights, full_covariance_norm,
            kind=kind, config=config, device_ids=list(map(int, data_parallel_device_ids)),
        )
        requested_devices = [f"cuda:{item}" for item in data_parallel_device_ids]
        if not successful:
            raise RuntimeError(f"all {config.restarts} {kind} DataParallel restarts failed: {restart_rows}")
        best_loss, selected_restart, basis, weights, _ = min(successful, key=lambda item: (item[0], item[1]))
        diagnostics = {
            "kind": kind,
            "dictionary_size": int(dictionary_size),
            "optimization": asdict(config),
            "selected_restart": int(selected_restart),
            "selected_training_loss": float(best_loss),
            "restart_devices": requested_devices,
            "parallelism": "torch.nn.DataParallel",
            "data_parallel_device_ids": list(map(int, data_parallel_device_ids)),
            "restarts": restart_rows,
            **_basis_diagnostics(basis, kind),
        }
        return SharedCovarianceFit(kind, basis, weights, float(best_loss), int(selected_restart), diagnostics)
    source_device = residuals.device
    requested_devices = list(restart_devices or [str(source_device)])
    if not requested_devices:
        raise ValueError("restart_devices must contain at least one device")
    requested_devices = list(dict.fromkeys(map(str, requested_devices)))
    assignments = {device: [] for device in requested_devices}
    for restart in range(config.restarts):
        assignments[requested_devices[restart % len(requested_devices)]].append(restart)

    def run_device_restarts(target_device: str, restart_ids: list[int]):
        if not restart_ids:
            return [], []
        target = torch.device(target_device)
        if target.type == "cuda":
            torch.cuda.set_device(target)
        local_residuals = residuals if target == source_device else residuals.to(target)
        local_pooled = pooled if target == source_device else pooled.to(target)
        local_initial_raw_weights = initial_raw_weights if target == source_device else initial_raw_weights.to(target)
        local_covariance_norm = full_covariance_norm if target == source_device else full_covariance_norm.to(target)
        local_rows: list[dict[str, Any]] = []
        local_successful: list[tuple[float, int, Any, Any, dict[str, Any]]] = []
        for restart in restart_ids:
            restart_started = time.monotonic()
            print(f"[shared_dictionary] RESTART start kind={kind} restart={restart} device={target}", flush=True)
            generator = torch.Generator(device=target)
            generator.manual_seed(int(config.seed) + 10007 * restart + (0 if kind == "cpc" else 1000003))
            try:
                # Every restart is a deterministic random perturbation of the
                # pooled-covariance eigenvector initialization.
                noise_scale = config.initialization_noise
                noise = torch.randn(local_pooled.shape, generator=generator, device=target, dtype=local_pooled.dtype)
                raw_basis = torch.nn.Parameter(local_pooled.clone() + noise_scale * noise)
                weight_noise = torch.randn(
                    local_initial_raw_weights.shape, generator=generator, device=target, dtype=local_pooled.dtype,
                )
                raw_weights = torch.nn.Parameter(local_initial_raw_weights.clone() + noise_scale * weight_noise)
                optimizer = torch.optim.Adam([raw_basis, raw_weights], lr=config.learning_rate)
                best_loss = float("inf")
                best_step = -1
                best_basis = None
                best_weights = None
                stale = 0
                last_gradient_norm = float("nan")
                for step in range(config.maximum_steps):
                    optimizer.zero_grad(set_to_none=True)
                    measure_full = step == 0 or (step + 1) % config.full_loss_interval == 0 or step + 1 == config.maximum_steps
                    if batch_size < context_count:
                        batch_indices = torch.randint(
                            0, context_count, (batch_size,), generator=generator, device=target,
                        )
                        batch_residuals = local_residuals.index_select(0, batch_indices)
                        batch_raw_weights = raw_weights.index_select(0, batch_indices)
                        batch_covariance_norm = local_covariance_norm.index_select(0, batch_indices)
                    else:
                        batch_residuals = local_residuals
                        batch_raw_weights = raw_weights
                        batch_covariance_norm = local_covariance_norm
                    loss, _, _ = _loss(
                        batch_residuals, raw_basis, batch_raw_weights, kind, config, batch_covariance_norm,
                    )
                    if measure_full and not bool(torch.isfinite(loss)):
                        raise FloatingPointError(f"non-finite training loss at step {step}")
                    loss.backward()
                    if measure_full:
                        last_gradient_norm = _gradient_norm([raw_basis, raw_weights])
                        if not math.isfinite(last_gradient_norm):
                            raise FloatingPointError(f"non-finite gradient norm at step {step}")
                    optimizer.step()
                    if measure_full:
                        with torch.no_grad():
                            full_loss, full_basis, full_weights = _loss(
                                local_residuals, raw_basis, raw_weights, kind, config, local_covariance_norm,
                            )
                        if not bool(torch.isfinite(full_loss)):
                            raise FloatingPointError(f"non-finite full training loss at step {step}")
                        current = float(full_loss.detach().cpu())
                        if current < best_loss - config.improvement_tolerance:
                            best_loss = current
                            best_step = step
                            best_basis = full_basis.detach().clone()
                            best_weights = full_weights.detach().clone()
                            stale = 0
                        else:
                            stale += config.full_loss_interval
                    if stale >= config.patience:
                        break
                if best_basis is None or best_weights is None:
                    raise RuntimeError("optimizer did not produce a finite state")
                diagnostics = {
                    "restart": restart,
                    "restart_device": str(target),
                    "status": "success",
                    "training_loss": best_loss,
                    "convergence_step": best_step,
                    "steps_executed": step + 1,
                    "elapsed_seconds": time.monotonic() - restart_started,
                    "final_gradient_norm": last_gradient_norm,
                    "context_batch_size": batch_size,
                    "full_training_loss_interval": config.full_loss_interval,
                    **_basis_diagnostics(best_basis, kind),
                }
                local_rows.append(diagnostics)
                local_successful.append((
                    best_loss, restart, best_basis.to(source_device), best_weights.to(source_device), diagnostics,
                ))
                print(
                    f"[shared_dictionary] RESTART done kind={kind} restart={restart} device={target} "
                    f"steps={step + 1} loss={best_loss:.8g} elapsed={diagnostics['elapsed_seconds']:.1f}s",
                    flush=True,
                )
            except (FloatingPointError, RuntimeError, ValueError) as exc:
                local_rows.append({
                    "restart": restart,
                    "restart_device": str(target),
                    "status": "failed",
                    "failure_type": type(exc).__name__,
                    "failure": str(exc),
                })
                print(
                    f"[shared_dictionary] RESTART failed kind={kind} restart={restart} device={target}: {exc}",
                    flush=True,
                )
        if target.type == "cuda" and target != source_device:
            del local_residuals, local_pooled, local_initial_raw_weights, local_covariance_norm
            torch.cuda.empty_cache()
        return local_rows, local_successful

    if len(requested_devices) == 1:
        results = [run_device_restarts(requested_devices[0], assignments[requested_devices[0]])]
    else:
        with ThreadPoolExecutor(max_workers=len(requested_devices), thread_name_prefix="shared-dictionary-gpu") as executor:
            futures = [
                executor.submit(run_device_restarts, device, assignments[device])
                for device in requested_devices
            ]
            results = [future.result() for future in futures]
    restart_rows = sorted(
        [row for rows, _ in results for row in rows], key=lambda row: int(row["restart"]),
    )
    successful = [item for _, values in results for item in values]

    if not successful:
        raise RuntimeError(f"all {config.restarts} {kind} restarts failed: {restart_rows}")
    best_loss, selected_restart, basis, weights, _ = min(successful, key=lambda item: (item[0], item[1]))
    diagnostics = {
        "kind": kind,
        "dictionary_size": int(dictionary_size),
        "optimization": asdict(config),
        "selected_restart": int(selected_restart),
        "selected_training_loss": float(best_loss),
        "restart_devices": requested_devices,
        "restarts": restart_rows,
        **_basis_diagnostics(basis, kind),
    }
    return SharedCovarianceFit(kind, basis, weights, float(best_loss), int(selected_restart), diagnostics)


def fit_fixed_dictionary_weights(
    residuals,
    basis,
    *,
    learning_rate: float,
    maximum_steps: int,
    patience: int,
    seed: int,
    epsilon: float = 1e-12,
    full_loss_interval: int = 25,
) -> tuple[Any, dict[str, Any]]:
    """Fit non-negative diagonal weights for a fixed non-orthogonal basis."""
    import torch

    if residuals.ndim != 3 or residuals.shape[0] != 1:
        raise ValueError("fixed-weight fitting expects one held-out context's training residuals")
    normalized = _normalize_columns(basis.detach(), epsilon)
    initial = torch.clamp(context_projection_variances(residuals, normalized), min=epsilon)
    raw = torch.nn.Parameter(_inverse_softplus(initial))
    optimizer = torch.optim.Adam([raw], lr=float(learning_rate))
    covariance_norm = covariance_frobenius_squared(residuals).detach()
    projected_variance = context_projection_variances(residuals, normalized).detach()
    squared_gram = torch.square(normalized.transpose(0, 1) @ normalized).detach()
    best_loss = float("inf")
    best_weights = None
    best_step = -1
    stale = 0
    last_gradient = float("nan")
    torch.manual_seed(int(seed))
    for step in range(int(maximum_steps)):
        optimizer.zero_grad(set_to_none=True)
        weights = torch.nn.functional.softplus(raw)
        cross = 2.0 * (weights * projected_variance).sum(dim=1)
        model_norm = torch.einsum("ck,kl,cl->c", weights, squared_gram, weights)
        loss = torch.clamp(
            (covariance_norm - cross + model_norm) / (covariance_norm + float(epsilon)), min=0.0,
        ).mean()
        measure_full = step == 0 or (step + 1) % int(full_loss_interval) == 0 or step + 1 == int(maximum_steps)
        if measure_full and not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite fixed-basis loss at step {step}")
        loss.backward()
        if measure_full:
            last_gradient = _gradient_norm([raw])
        optimizer.step()
        if measure_full:
            with torch.no_grad():
                current_weights = torch.nn.functional.softplus(raw)
                current_cross = 2.0 * (current_weights * projected_variance).sum(dim=1)
                current_model_norm = torch.einsum(
                    "ck,kl,cl->c", current_weights, squared_gram, current_weights,
                )
                current_loss = torch.clamp(
                    (covariance_norm - current_cross + current_model_norm) / (covariance_norm + float(epsilon)),
                    min=0.0,
                ).mean()
            current = float(current_loss.detach().cpu())
            if current < best_loss - 1e-10:
                best_loss = current
                best_weights = current_weights.detach().clone()[0]
                best_step = step
                stale = 0
            else:
                stale += int(full_loss_interval)
        if stale >= int(patience):
            break
    if best_weights is None:
        raise RuntimeError("fixed-basis weight optimization did not produce a finite state")
    return best_weights, {
        "training_loss": best_loss,
        "convergence_step": best_step,
        "steps_executed": step + 1,
        "final_gradient_norm": last_gradient,
        "full_training_loss_interval": int(full_loss_interval),
    }


def fitted_state_numpy(fit: SharedCovarianceFit) -> dict[str, Any]:
    return {
        "kind": fit.kind,
        "basis": fit.basis.detach().cpu().numpy(),
        "weights": fit.weights.detach().cpu().numpy(),
        "loss": fit.loss,
        "selected_restart": fit.selected_restart,
        "diagnostics": fit.diagnostics,
    }
