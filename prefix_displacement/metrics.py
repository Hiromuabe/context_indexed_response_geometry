from __future__ import annotations

import math
import random
from typing import Any, Callable, Iterable, Mapping, Sequence


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    x_mean, y_mean = mean(x), mean(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    x_energy = sum((a - x_mean) ** 2 for a in x)
    y_energy = sum((b - y_mean) ** 2 for b in y)
    denominator = math.sqrt(x_energy * y_energy)
    return numerator / denominator if denominator > 0 else float("nan")


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + end - 1) / 2.0
        for offset in range(start, end):
            ranks[order[offset]] = average_rank
        start = end
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    return pearson(_average_ranks(x), _average_ranks(y))


def functional_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    true = [float(row["s_true"]) for row in rows]
    predicted = [float(row["s_hat"]) for row in rows]
    if not true:
        return {name: float("nan") for name in (
            "functional_r2", "pearson", "spearman", "sign_agreement",
            "normalized_mae", "calibration_intercept", "calibration_slope"
        )}
    true_mean = mean(true)
    denominator = sum((value - true_mean) ** 2 for value in true)
    residual = sum((a - b) ** 2 for a, b in zip(true, predicted))
    r2 = 1.0 - residual / denominator if denominator > 0 else float("nan")
    pred_mean = mean(predicted)
    pred_variance = sum((value - pred_mean) ** 2 for value in predicted)
    covariance = sum((a - true_mean) * (b - pred_mean) for a, b in zip(true, predicted))
    slope = covariance / pred_variance if pred_variance > 0 else float("nan")
    intercept = true_mean - slope * pred_mean if math.isfinite(slope) else float("nan")
    scale = mean([abs(value) for value in true])
    return {
        "functional_r2": r2,
        "pearson": pearson(true, predicted),
        "spearman": spearman(true, predicted),
        "sign_agreement": mean([float((a >= 0) == (b >= 0)) for a, b in zip(true, predicted)]),
        "normalized_mae": mean([abs(a - b) for a, b in zip(true, predicted)]) / max(scale, 1e-12),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def geometric_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    return {
        "mean_reconstruction_cosine": mean(
            [float(row["reconstruction_cosine"]) for row in rows]
        ),
        "mean_relative_error": mean([float(row["relative_error"]) for row in rows]),
        "mean_retained_energy": mean([float(row["retained_energy"]) for row in rows]),
    }


def aggregate_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    return {**geometric_metrics(rows), **functional_metrics(rows), "num_rows": len(rows)}


def problem_cluster_bootstrap_difference(
    rows_a: Sequence[Mapping[str, Any]],
    rows_b: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    grouped_a: dict[str, list[Mapping[str, Any]]] = {}
    grouped_b: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows_a:
        grouped_a.setdefault(str(row["problem_id"]), []).append(row)
    for row in rows_b:
        grouped_b.setdefault(str(row["problem_id"]), []).append(row)
    problem_ids = sorted(set(grouped_a) & set(grouped_b))
    if set(grouped_a) != set(grouped_b):
        raise ValueError("Paired methods do not cover the same problem IDs")
    rng = random.Random(seed)
    differences = []
    metric_function: Callable[[Sequence[Mapping[str, Any]]], dict[str, float]] = aggregate_metrics
    for _ in range(replicates):
        sampled = [rng.choice(problem_ids) for _ in problem_ids]
        sample_a = [row for problem_id in sampled for row in grouped_a[problem_id]]
        sample_b = [row for problem_id in sampled for row in grouped_b[problem_id]]
        differences.append(
            metric_function(sample_a)[metric] - metric_function(sample_b)[metric]
        )
    differences.sort()
    lower_index = max(0, int(0.025 * replicates))
    upper_index = min(replicates - 1, int(0.975 * replicates))
    point = aggregate_metrics(rows_a)[metric] - aggregate_metrics(rows_b)[metric]
    return {
        "point_difference": point,
        "ci_lower": differences[lower_index],
        "ci_upper": differences[upper_index],
        "replicates": replicates,
        "unit": "problem_id",
    }
