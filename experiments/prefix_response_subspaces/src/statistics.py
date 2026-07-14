from __future__ import annotations

import numpy as np


def problem_bootstrap(values: np.ndarray, problem_ids: np.ndarray, *, replicates: int, seed: int, ci: float = 0.95) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    problem_ids = np.asarray(problem_ids)
    valid = np.isfinite(values)
    values, problem_ids = values[valid], problem_ids[valid]
    unique = np.unique(problem_ids)
    if not len(unique):
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_problems": 0}
    grouped = np.asarray([values[problem_ids == item].mean() for item in unique])
    rng = np.random.default_rng(seed)
    samples = np.asarray([rng.choice(grouped, size=len(grouped), replace=True).mean() for _ in range(replicates)])
    alpha = (1.0 - ci) / 2.0
    return {"mean": float(grouped.mean()), "ci_low": float(np.quantile(samples, alpha)), "ci_high": float(np.quantile(samples, 1-alpha)), "n_problems": int(len(unique))}


def problem_ratio_bootstrap(
    numerators: np.ndarray,
    denominators: np.ndarray,
    problem_ids: np.ndarray,
    *,
    replicates: int,
    seed: int,
    ci: float = 0.95,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Bootstrap a ratio of totals while resampling independent problems.

    The point estimate is ``sum(numerator) / sum(denominator)``.  Cells are
    first summed within problem, then whole problem blocks are resampled.  This
    avoids the instability and implicit reweighting caused by averaging
    cell-wise ratios when some denominators are close to zero.
    """
    numerators = np.asarray(numerators, dtype=np.float64)
    denominators = np.asarray(denominators, dtype=np.float64)
    problem_ids = np.asarray(problem_ids)
    if numerators.shape != denominators.shape or numerators.shape != problem_ids.shape:
        raise ValueError("numerators, denominators, and problem_ids must have identical shapes")
    valid = np.isfinite(numerators) & np.isfinite(denominators)
    numerators, denominators, problem_ids = numerators[valid], denominators[valid], problem_ids[valid]
    unique = np.unique(problem_ids)
    if not len(unique):
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_problems": 0}
    grouped_num = np.asarray([numerators[problem_ids == item].sum() for item in unique])
    grouped_den = np.asarray([denominators[problem_ids == item].sum() for item in unique])
    total_den = float(grouped_den.sum())
    point = float(grouped_num.sum() / total_den) if abs(total_den) > eps else float("nan")
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(replicates):
        indices = rng.integers(0, len(unique), size=len(unique))
        denominator = float(grouped_den[indices].sum())
        if abs(denominator) > eps:
            samples.append(float(grouped_num[indices].sum() / denominator))
    alpha = (1.0 - ci) / 2.0
    if not samples:
        low = high = float("nan")
    else:
        low, high = np.quantile(np.asarray(samples), [alpha, 1 - alpha])
    return {"mean": point, "ci_low": float(low), "ci_high": float(high), "n_problems": int(len(unique))}


def permutation_pvalue(observed: float, null_values: np.ndarray) -> float:
    null = np.asarray(null_values, dtype=np.float64)
    null = null[np.isfinite(null)]
    return float((1 + np.sum(null >= observed)) / (1 + len(null)))
