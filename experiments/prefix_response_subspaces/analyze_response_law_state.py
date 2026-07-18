from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .src.data import pad_token_rows
from .src.matching import match_prefixes, paired_js_from_logits
from .src.metrics import LOG_TWO
from .src.model import assert_output_order, load_next_token_model
from .src.review_experiments import review_roots
from .src.storage import load_residual_entry
from .src.subspaces import normalized_projection_distance, top_svd
from .src.utils import (
    atomic_json,
    file_sha256,
    load_config,
    read_json,
    read_jsonl,
    stable_hash,
    stage_is_complete,
)


IMPLEMENTATION_VERSION = "response_law_contrast_geometry_v2"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        keys.extend(key for key in row if key not in keys)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def top_token_ids(logits: np.ndarray, vocabulary_size: int, count: int = 20) -> np.ndarray:
    result = np.empty((len(logits), int(count)), dtype=np.int64)
    for start in range(0, len(logits), 32):
        values = np.asarray(logits[start : start + 32, : int(vocabulary_size)], dtype=np.float32)
        unordered = np.argpartition(values, -int(count), axis=1)[:, -int(count) :]
        scores = np.take_along_axis(values, unordered, axis=1)
        order = np.argsort(-scores, axis=1, kind="stable")
        result[start : start + len(values)] = np.take_along_axis(unordered, order, axis=1)
    return result


def select_distribution_matched_pairs(
    records: list[dict[str, Any]],
    logits: np.ndarray,
    top_tokens: np.ndarray,
    vocabulary_size: int,
    *,
    group: str,
    limit: int,
) -> list[dict[str, Any]]:
    indices = np.asarray([index for index, row in enumerate(records) if row["problem_group"] == group], dtype=np.int64)
    matches = match_prefixes(
        records,
        query_indices=indices,
        candidate_indices=indices,
        logits=logits,
        top_token_ids=top_tokens,
        tokenizer_vocabulary_size=int(vocabulary_size),
    )
    axis = {str(row["prefix_id"]): index for index, row in enumerate(records)}
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for match in matches:
        if not match.get("matched"):
            continue
        left_id = str(match["prefix_id"])
        right_id = str(match["matched_prefix_id"])
        key = tuple(sorted((left_id, right_id)))
        row = {
            "left_prefix_id": key[0],
            "right_prefix_id": key[1],
            "left_axis": int(axis[key[0]]),
            "right_axis": int(axis[key[1]]),
            "left_problem_id": str(records[axis[key[0]]]["problem_id"]),
            "right_problem_id": str(records[axis[key[1]]]["problem_id"]),
            "current_js": float(match["js_distance"]),
            "current_normalized_js": float(match["js_distance"] / LOG_TWO),
            "top5_overlap": int(match.get("top5_overlap", 0)),
            "top20_overlap": int(match.get("top20_overlap", 0)),
            "same_top1_token_id": int(match["same_top1_token_id"]),
            "prefix_length_difference": abs(
                int(records[axis[key[0]]]["prefix_length"]) - int(records[axis[key[1]]]["prefix_length"])
            ),
        }
        if key not in unique or row["current_js"] < unique[key]["current_js"]:
            unique[key] = row
    ordered = sorted(unique.values(), key=lambda row: (row["current_js"], row["left_prefix_id"], row["right_prefix_id"]))
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for row in ordered:
        if row["left_prefix_id"] in used or row["right_prefix_id"] in used:
            continue
        selected.append(row)
        used.update((row["left_prefix_id"], row["right_prefix_id"]))
        if len(selected) == int(limit):
            break
    for number, row in enumerate(selected):
        row["pair_id"] = f"{group}:{number:03d}"
        row["split"] = "development" if group == "analysis_dev" else "evaluation"
    return selected


def response_matrix_cosine_distance(left: np.ndarray, right: np.ndarray, eps: float = 1e-12) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= eps:
        return float("nan")
    return float(1.0 - np.clip(np.sum(a * b) / denominator, -1.0, 1.0))


def attach_response_distances(
    pairs: list[dict[str, Any]],
    residual_manifest: dict[str, Any],
    hidden_prefixes: list[dict[str, Any]],
    *,
    layer: int,
    rank: int,
) -> None:
    prefix_axis = {str(row["prefix_id"]): index for index, row in enumerate(hidden_prefixes)}
    selected_entries = [row for row in residual_manifest["entries"] if int(row["layer"]) == int(layer)]
    if not selected_entries:
        raise RuntimeError(f"no residual entries for selected layer {layer}")
    accumulators = {row["pair_id"]: {"subspace": [], "law": [], "rank": []} for row in pairs}
    for entry in selected_entries:
        bundle = load_residual_entry(entry)
        nonauxiliary = np.asarray(bundle["nonauxiliary_prefix_indices"], dtype=np.int64)
        local_axis = {int(full): local for local, full in enumerate(nonauxiliary)}
        for pair in pairs:
            try:
                left = local_axis[prefix_axis[pair["left_prefix_id"]]]
                right = local_axis[prefix_axis[pair["right_prefix_id"]]]
            except KeyError:
                continue
            left_response = np.asarray(bundle["train_residuals"][left], dtype=np.float64)
            right_response = np.asarray(bundle["train_residuals"][right], dtype=np.float64)
            effective_rank = min(int(rank), left_response.shape[0] - 1, right_response.shape[0] - 1)
            left_basis = top_svd(left_response, effective_rank, allow_rank_reduction=True)
            right_basis = top_svd(right_response, effective_rank, allow_rank_reduction=True)
            target = accumulators[pair["pair_id"]]
            target["subspace"].append(normalized_projection_distance(left_basis, right_basis))
            target["law"].append(response_matrix_cosine_distance(left_response, right_response))
            target["rank"].append(min(left_basis.shape[1], right_basis.shape[1]))
    for pair in pairs:
        values = accumulators[pair["pair_id"]]
        if not values["subspace"]:
            raise RuntimeError(f"response residuals unavailable for pair {pair['pair_id']}")
        pair["response_subspace_distance"] = float(np.mean(values["subspace"]))
        pair["response_law_distance"] = float(np.mean(values["law"]))
        pair["response_distance_folds"] = len(values["subspace"])
        pair["response_effective_rank"] = int(min(values["rank"]))


def common_forced_tokens(
    left_logits: np.ndarray,
    right_logits: np.ndarray,
    *,
    count: int,
    vocabulary_size: int,
    special_ids: set[int],
    intersection_top_k: int = 128,
    allowed_token_ids: list[int] | None = None,
) -> list[int]:
    left = np.asarray(left_logits[: int(vocabulary_size)], dtype=np.float32)
    right = np.asarray(right_logits[: int(vocabulary_size)], dtype=np.float32)
    k = min(int(intersection_top_k), len(left))
    left_top = set(np.argpartition(left, -k)[-k:].tolist())
    right_top = set(np.argpartition(right, -k)[-k:].tolist())
    scores = left + right
    common = sorted(left_top & right_top, key=lambda token_id: (-float(scores[token_id]), int(token_id)))
    allowed = set(map(int, allowed_token_ids)) if allowed_token_ids is not None else set(range(len(scores)))
    selected = [
        int(token_id) for token_id in common
        if int(token_id) not in special_ids and int(token_id) in allowed
    ]
    if len(selected) < int(count):
        proposals = np.asarray(sorted(allowed), dtype=np.int64)
        ordered = sorted(map(int, proposals), key=lambda token_id: (-float(scores[token_id]), token_id))
        selected.extend(token_id for token_id in ordered if token_id not in special_ids and token_id not in selected)
    if len(selected) < int(count):
        raise RuntimeError(f"only {len(selected)} valid common forced tokens for requested {count}")
    return selected[: int(count)]


def fixed_candidate_panel(
    logits: np.ndarray,
    context_axes: list[int],
    candidate_token_ids: list[int],
    *,
    count: int,
    special_ids: set[int],
) -> list[int]:
    eligible = np.asarray([
        int(token_id) for token_id in candidate_token_ids if int(token_id) not in special_ids
    ], dtype=np.int64)
    if len(eligible) < int(count):
        raise RuntimeError(f"only {len(eligible)} eligible studied candidates for requested panel {count}")
    score = np.zeros(len(eligible), dtype=np.float64)
    axes = np.asarray(sorted(set(map(int, context_axes))), dtype=np.int64)
    for start in range(0, len(axes), 16):
        block = np.asarray(logits[axes[start : start + 16]][:, eligible], dtype=np.float32)
        score += block.sum(axis=0, dtype=np.float64)
    score /= float(len(axes))
    order = sorted(range(len(eligible)), key=lambda index: (-float(score[index]), int(eligible[index])))
    return [int(eligible[index]) for index in order[: int(count)]]


def center_candidate_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("candidate profiles must be a finite matrix")
    return matrix - matrix.mean(axis=0, keepdims=True)


def candidate_gram(values: np.ndarray) -> np.ndarray:
    centered = center_candidate_rows(values)
    return centered @ centered.T


def gram_alignment(left: np.ndarray, right: np.ndarray, eps: float = 1e-18) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Gram matrices must have the same square shape")
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= eps:
        return float("nan")
    return float(np.clip(np.sum(a * b) / denominator, -1.0, 1.0))


def double_center_hidden_profile(context_values: np.ndarray, auxiliary_token_mean: np.ndarray) -> np.ndarray:
    context = np.asarray(context_values, dtype=np.float64)
    token_mean = np.asarray(auxiliary_token_mean, dtype=np.float64)
    if context.shape != token_mean.shape or context.ndim != 2:
        raise ValueError("context and auxiliary profiles must share [candidate, hidden] shape")
    result = context - context.mean(axis=0, keepdims=True)
    result -= token_mean
    result += token_mean.mean(axis=0, keepdims=True)
    result -= result.mean(axis=0, keepdims=True)
    return result


def row_space_projection_distance(left: np.ndarray, right: np.ndarray, rank: int) -> float:
    a = center_candidate_rows(left)
    b = center_candidate_rows(right)
    if a.shape != b.shape:
        raise ValueError("row-space comparison requires aligned matrices")

    def eigensystem(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gram = matrix @ matrix.T
        values, vectors = np.linalg.eigh(gram)
        order = np.argsort(values)[::-1]
        values = np.clip(values[order], 0.0, None)
        vectors = vectors[:, order]
        tolerance = values[0] * max(matrix.shape) * np.finfo(np.float64).eps if len(values) else 0.0
        available = int(np.sum(values > tolerance))
        effective = min(int(rank), available)
        if effective <= 0:
            raise ValueError("candidate profile has zero numerical rank")
        return np.sqrt(values[:effective]), vectors[:, :effective]

    singular_a, vectors_a = eigensystem(a)
    singular_b, vectors_b = eigensystem(b)
    effective = min(len(singular_a), len(singular_b))
    overlap = (
        (vectors_a[:, :effective].T @ (a @ b.T) @ vectors_b[:, :effective])
        / singular_a[:effective, None]
        / singular_b[None, :effective]
    )
    cosines = np.linalg.svd(overlap, compute_uv=False)
    return float(1.0 - np.mean(np.square(np.clip(cosines, 0.0, 1.0))))


def profile_geometry(
    hidden_left: np.ndarray,
    hidden_right: np.ndarray,
    future_left: np.ndarray,
    future_right: np.ndarray,
    *,
    rank: int,
) -> dict[str, Any]:
    hidden_left = center_candidate_rows(hidden_left)
    hidden_right = center_candidate_rows(hidden_right)
    future_left = center_candidate_rows(future_left)
    future_right = center_candidate_rows(future_right)
    grams = {
        "hidden_left": hidden_left @ hidden_left.T,
        "hidden_right": hidden_right @ hidden_right.T,
        "future_left": future_left @ future_left.T,
        "future_right": future_right @ future_right.T,
    }
    return {
        "left_hidden_future_cka": gram_alignment(grams["hidden_left"], grams["future_left"]),
        "right_hidden_future_cka": gram_alignment(grams["hidden_right"], grams["future_right"]),
        "hidden_contrast_law_distance": response_matrix_cosine_distance(hidden_left, hidden_right),
        "future_contrast_law_distance": response_matrix_cosine_distance(future_left, future_right),
        "hidden_candidate_kernel_distance": 1.0 - gram_alignment(grams["hidden_left"], grams["hidden_right"]),
        "future_candidate_kernel_distance": 1.0 - gram_alignment(grams["future_left"], grams["future_right"]),
        "hidden_contrast_subspace_distance": row_space_projection_distance(hidden_left, hidden_right, rank),
        "future_contrast_subspace_distance": row_space_projection_distance(future_left, future_right, rank),
        "grams": {name: value.tolist() for name, value in grams.items()},
    }


def _standardize(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    scale = float(x.std())
    return (x - x.mean()) / scale if scale > 1e-12 else np.zeros_like(x)


def partial_standardized_beta(rows: list[dict[str, Any]], predictor: str) -> float:
    if len(rows) < 4:
        return float("nan")
    y = _standardize(np.asarray([row["future_normalized_js"] for row in rows]))
    current = _standardize(np.asarray([row["current_normalized_js"] for row in rows]))
    response = _standardize(np.asarray([row[predictor] for row in rows]))
    design = np.column_stack((np.ones(len(rows)), current, response))
    return float(np.linalg.lstsq(design, y, rcond=None)[0][2])


def standardized_beta(
    rows: list[dict[str, Any]], predictor: str, outcome: str, controls: tuple[str, ...] = (),
) -> float:
    if len(rows) < 4:
        return float("nan")
    y = _standardize(np.asarray([row[outcome] for row in rows], dtype=np.float64))
    columns = [np.ones(len(rows))]
    columns.extend(_standardize(np.asarray([row[key] for row in rows], dtype=np.float64)) for key in controls)
    columns.append(_standardize(np.asarray([row[predictor] for row in rows], dtype=np.float64)))
    return float(np.linalg.lstsq(np.column_stack(columns), y, rcond=None)[0][-1])


def _average_ranks(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    order = np.argsort(x, kind="stable")
    result = np.empty(len(x), dtype=np.float64)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and x[order[stop]] == x[order[start]]:
            stop += 1
        result[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return result


def correlation(rows: list[dict[str, Any]], left: str, right: str, *, ranks: bool = False) -> float:
    if len(rows) < 3:
        return float("nan")
    x = np.asarray([row[left] for row in rows], dtype=np.float64)
    y = np.asarray([row[right] for row in rows], dtype=np.float64)
    if ranks:
        x, y = _average_ranks(x), _average_ranks(y)
    x, y = x - x.mean(), y - y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.sum(x * y) / denominator) if denominator > 1e-18 else float("nan")


def bootstrap_statistic(
    rows: list[dict[str, Any]],
    statistic: Callable[[list[dict[str, Any]]], float],
    *,
    replicates: int,
    seed: int,
    ci: float,
) -> dict[str, float]:
    if not rows:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_pairs": 0}
    point = float(statistic(rows))
    rng = np.random.default_rng(int(seed))
    samples = []
    for _ in range(int(replicates)):
        sampled = [rows[int(index)] for index in rng.integers(0, len(rows), size=len(rows))]
        value = float(statistic(sampled))
        if np.isfinite(value):
            samples.append(value)
    alpha = (1.0 - float(ci)) / 2.0
    low, high = np.quantile(samples, [alpha, 1.0 - alpha]) if samples else (float("nan"), float("nan"))
    return {"mean": point, "ci_low": float(low), "ci_high": float(high), "n_pairs": len(rows)}


def threshold_contrast(
    rows: list[dict[str, Any]], predictor: str, threshold: float, *, replicates: int, seed: int, ci: float
) -> dict[str, float]:
    low = [row for row in rows if float(row[predictor]) <= float(threshold)]
    high = [row for row in rows if float(row[predictor]) > float(threshold)]

    def point(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> float:
        return float(np.mean([row["future_normalized_js"] for row in right]) - np.mean([row["future_normalized_js"] for row in left]))

    if not low or not high:
        return {"high_minus_low": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_low": len(low), "n_high": len(high)}
    rng = np.random.default_rng(int(seed))
    samples = []
    for _ in range(int(replicates)):
        sampled_low = [low[int(index)] for index in rng.integers(0, len(low), size=len(low))]
        sampled_high = [high[int(index)] for index in rng.integers(0, len(high), size=len(high))]
        samples.append(point(sampled_low, sampled_high))
    alpha = (1.0 - float(ci)) / 2.0
    lower, upper = np.quantile(samples, [alpha, 1.0 - alpha])
    return {
        "high_minus_low": point(low, high), "ci_low": float(lower), "ci_high": float(upper),
        "n_low": len(low), "n_high": len(high), "threshold_fixed_on_development": float(threshold),
    }


def geometry_association_summary(
    rows: list[dict[str, Any]], predictor: str, outcome: str, *, replicates: int, seed: int, ci: float,
) -> dict[str, Any]:
    pearson = bootstrap_statistic(
        rows, lambda sample: correlation(sample, predictor, outcome),
        replicates=replicates, seed=seed, ci=ci,
    )
    spearman = bootstrap_statistic(
        rows, lambda sample: correlation(sample, predictor, outcome, ranks=True),
        replicates=replicates, seed=seed + 1, ci=ci,
    )
    partial = bootstrap_statistic(
        rows,
        lambda sample: standardized_beta(
            sample, predictor, outcome, controls=("current_normalized_js",),
        ),
        replicates=replicates, seed=seed + 2, ci=ci,
    )
    rng = np.random.default_rng(int(seed) + 3)
    observed = float(pearson["mean"])
    outcomes = np.asarray([row[outcome] for row in rows], dtype=np.float64)
    null = []
    for _ in range(int(replicates)):
        permuted = rng.permutation(outcomes)
        permuted_rows = [{**row, outcome: float(permuted[index])} for index, row in enumerate(rows)]
        value = correlation(permuted_rows, predictor, outcome)
        if np.isfinite(value):
            null.append(float(value))
    p_positive = (1 + sum(value >= observed for value in null)) / (1 + len(null))
    return {
        "predictor": predictor,
        "outcome": outcome,
        "pearson_pair_bootstrap": pearson,
        "spearman_pair_bootstrap": spearman,
        "partial_standardized_beta_controlling_current_js": partial,
        "positive_association_permutation_p": float(p_positive),
        "outcome_mean": float(outcomes.mean()),
        "outcome_sd": float(outcomes.std()),
        "outcome_min": float(outcomes.min()),
        "outcome_max": float(outcomes.max()),
    }


def candidate_alignment_permutation_test(
    rows: list[dict[str, Any]], *, replicates: int, seed: int, ci: float,
) -> dict[str, Any]:
    observed_rows = [
        {**row, "pair_candidate_cka": 0.5 * (row["left_hidden_future_cka"] + row["right_hidden_future_cka"])}
        for row in rows
    ]
    observed = bootstrap_statistic(
        observed_rows, lambda sample: float(np.mean([row["pair_candidate_cka"] for row in sample])),
        replicates=replicates, seed=seed, ci=ci,
    )
    rng = np.random.default_rng(int(seed) + 1)
    null = []
    for _ in range(int(replicates)):
        alignments = []
        permutation = rng.permutation(len(rows[0]["grams"]["hidden_left"]))
        for row in rows:
            for side in ("left", "right"):
                hidden = np.asarray(row["grams"][f"hidden_{side}"], dtype=np.float64)
                future = np.asarray(row["grams"][f"future_{side}"], dtype=np.float64)
                shuffled = future[np.ix_(permutation, permutation)]
                alignments.append(gram_alignment(hidden, shuffled))
        null.append(float(np.mean(alignments)))
    alpha = (1.0 - float(ci)) / 2.0
    low, high = np.quantile(null, [alpha, 1.0 - alpha])
    null_mean = float(np.mean(null))
    p_value = (1 + sum(value >= float(observed["mean"]) for value in null)) / (1 + len(null))
    return {
        "observed_pair_bootstrap": observed,
        "candidate_label_permutation_mean": null_mean,
        "candidate_label_permutation_interval": {"low": float(low), "high": float(high)},
        "observed_minus_permutation_mean": float(observed["mean"] - null_mean),
        "positive_alignment_permutation_p": float(p_value),
        "permutations": int(replicates),
    }


def _aggregate_future_rows(pairs: list[dict[str, Any]], future_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for row in future_rows:
        by_pair.setdefault(str(row["pair_id"]), []).append(row)
    result = []
    for pair in pairs:
        cells = by_pair.get(str(pair["pair_id"]), [])
        if not cells:
            continue
        result.append({
            **pair,
            "future_normalized_js": float(np.mean([row["future_normalized_js"] for row in cells])),
            "future_js": float(np.mean([row["future_js"] for row in cells])),
            "future_top1_agreement": float(np.mean([row["future_top1_agreement"] for row in cells])),
            "future_top5_overlap": float(np.mean([row["future_top5_overlap"] for row in cells])),
            "forced_candidate_count": len(cells),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    source_root, root = review_roots(config)
    paths = {
        "candidate_manifest": source_root / "manifests/candidate_tokens.json",
        "candidate_tokens": source_root / "candidate_tokens/candidate_tokens.json",
        "prefixes": source_root / "prefix_pool/prefixes.jsonl",
        "hidden": source_root / "manifests/hidden_states.json",
        "geometry": source_root / "metrics/paper_geometry_summary.json",
    }
    inputs = {f"{key}_sha256": file_sha256(path) for key, path in paths.items()}
    inputs["implementation_version"] = IMPLEMENTATION_VERSION
    manifest_path = root / "manifests/response_law_state.json"
    if not args.force and manifest_path.is_file():
        existing_manifest = read_json(manifest_path)
        if existing_manifest.get("config_hash") != stable_hash(config):
            raise RuntimeError(f"existing stage has a different configuration: {manifest_path}")
        if existing_manifest.get("implementation_version") == IMPLEMENTATION_VERSION:
            if stage_is_complete(manifest_path, config, inputs):
                print(manifest_path)
                return
        else:
            print(
                f"[response_state] upgrading retired level test to {IMPLEMENTATION_VERSION}",
                flush=True,
            )

    quick = bool(config.get("quick_check"))
    formal = bool(config.get("formal_check"))
    if quick:
        development_pair_limit, evaluation_pair_limit = 8, 16
        forced_candidate_count, contrast_rank = 4, 2
        contrast_auxiliary_contexts = 8
        replicates = 50
    elif formal:
        development_pair_limit, evaluation_pair_limit = 16, 32
        forced_candidate_count, contrast_rank = 8, 4
        contrast_auxiliary_contexts = 32
        replicates = 500
    else:
        development_pair_limit, evaluation_pair_limit = 24, 48
        forced_candidate_count, contrast_rank = 8, 4
        contrast_auxiliary_contexts = 64
        replicates = int(config["statistics"]["bootstrap_replicates"])
    ci = float(config["statistics"]["ci"])
    seed = int(config["seed"]) + 32003

    candidate_manifest = read_json(paths["candidate_manifest"])
    prefix_rows = read_jsonl(paths["prefixes"])
    prefix_by_id = {str(row["prefix_id"]): row for row in prefix_rows}
    axis_ids = list(map(str, candidate_manifest["prefix_axis_ids"]))
    scored_records = [prefix_by_id[prefix_id] for prefix_id in axis_ids]
    logits_path = Path(str(candidate_manifest.get("next_token_logits", source_root / "prefix_pool/next_token_logits.npy")))
    logits = np.load(logits_path, mmap_mode="r")
    vocabulary_size = int(candidate_manifest["tokenizer_vocabulary_size"])
    print(f"[response_state] matching current distributions rows={len(scored_records)} vocab={vocabulary_size}", flush=True)
    top_tokens = top_token_ids(logits, vocabulary_size, 20)
    development_pairs = select_distribution_matched_pairs(
        scored_records, logits, top_tokens, vocabulary_size, group="analysis_dev", limit=development_pair_limit,
    )
    evaluation_pairs = select_distribution_matched_pairs(
        scored_records, logits, top_tokens, vocabulary_size, group="analysis_test", limit=evaluation_pair_limit,
    )
    if len(development_pairs) < 4 or len(evaluation_pairs) < 8:
        raise RuntimeError(
            f"too few disjoint current-distribution matches: development={len(development_pairs)} "
            f"evaluation={len(evaluation_pairs)}"
        )
    pairs = development_pairs + evaluation_pairs
    hidden = read_json(paths["hidden"])
    hidden_prefixes = read_jsonl(hidden["prefix_snapshot"])
    hidden_prefix_axis = {str(row["prefix_id"]): index for index, row in enumerate(hidden_prefixes)}
    geometry = read_json(paths["geometry"])
    selected_layer = int(geometry["selected_layer"])
    layer_entries = [entry for entry in hidden["layers"] if int(entry["layer"]) == selected_layer]
    if len(layer_entries) != 1:
        raise RuntimeError(f"expected one hidden-state entry for selected layer {selected_layer}")
    successor_states = np.load(layer_entries[0]["successor_path"], mmap_mode="r")
    candidate_data = read_json(paths["candidate_tokens"])
    candidate_token_ids = list(map(int, candidate_data["candidate_token_ids"]))
    if successor_states.shape[1] != len(candidate_token_ids):
        raise RuntimeError("candidate-token order does not match the hidden-state candidate axis")
    candidate_axis = {token_id: index for index, token_id in enumerate(candidate_token_ids)}
    loaded = load_next_token_model(config, args.model_path)
    special_ids = set(map(int, loaded.tokenizer.all_special_ids))
    panel_context_axes = [
        int(pair[side]) for pair in development_pairs for side in ("left_axis", "right_axis")
    ]
    forced_panel = fixed_candidate_panel(
        logits, panel_context_axes, candidate_token_ids,
        count=forced_candidate_count, special_ids=special_ids,
    )
    for pair in pairs:
        pair["forced_token_ids"] = list(forced_panel)

    all_auxiliary_indices = [
        index for index, row in enumerate(hidden_prefixes) if row["problem_group"] == "auxiliary"
    ]
    all_auxiliary_indices.sort(key=lambda index: str(hidden_prefixes[index]["prefix_id"]))
    auxiliary_indices = np.asarray(all_auxiliary_indices[:contrast_auxiliary_contexts], dtype=np.int64)
    if not len(auxiliary_indices):
        raise RuntimeError("response-law contrasts require auxiliary prefixes")
    auxiliary_sum = np.zeros((len(candidate_token_ids), successor_states.shape[2]), dtype=np.float64)
    chunk_size = max(1, int(config.get("extraction", {}).get("prefix_chunk_size", 8)))
    print(
        f"[response_state] auxiliary centering contexts={len(auxiliary_indices)} "
        f"candidates={len(candidate_token_ids)} layer={selected_layer}",
        flush=True,
    )
    for start in range(0, len(auxiliary_indices), chunk_size):
        block = np.asarray(successor_states[auxiliary_indices[start : start + chunk_size]], dtype=np.float32)
        auxiliary_sum += block.sum(axis=0, dtype=np.float64)
    auxiliary_token_mean = auxiliary_sum / float(len(auxiliary_indices))

    output_root = root / "functional_state"
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_root / "response_law_future_checkpoint.json"
    checkpoint_key = stable_hash({
        "version": IMPLEMENTATION_VERSION, "config_hash": stable_hash(config), "inputs": inputs,
        "pairs": [(row["pair_id"], row["forced_token_ids"]) for row in pairs],
        "contrast_auxiliary_prefix_ids": [hidden_prefixes[index]["prefix_id"] for index in auxiliary_indices],
        "model_identity": {
            "checkpoint": config["model"]["checkpoint"],
            "configured_revision": config["model"].get("revision", "main"),
            "resolved_revision": loaded.resolved_revision,
            "hidden_size": loaded.hidden_size,
        },
    })
    import torch

    auxiliary_future_path = output_root / "response_law_auxiliary_future_contrast.npy"
    auxiliary_future_meta_path = output_root / "response_law_auxiliary_future_contrast.json"
    auxiliary_cache_key = stable_hash({
        "checkpoint_key": checkpoint_key,
        "auxiliary_indices": auxiliary_indices.tolist(),
        "forced_panel": forced_panel,
    })
    auxiliary_meta = (
        read_json(auxiliary_future_meta_path)
        if auxiliary_future_meta_path.is_file() and not args.force
        else {}
    )
    if auxiliary_meta.get("cache_key") == auxiliary_cache_key and auxiliary_future_path.is_file():
        future_auxiliary_contrast_mean = np.load(auxiliary_future_path, mmap_mode="r")
        print(f"[response_state] reuse auxiliary future contrast {auxiliary_future_path}", flush=True)
    else:
        profile_sum = np.zeros((forced_candidate_count, vocabulary_size), dtype=np.float64)
        global_sequences = max(
            1, int(config.get("extraction", {}).get("per_device_batch_size", 16)) * max(1, len(loaded.device_ids))
        )
        contexts_per_batch = max(1, global_sequences // forced_candidate_count)
        print(
            f"[response_state] future auxiliary centering contexts={len(auxiliary_indices)} "
            f"candidates={forced_candidate_count}",
            flush=True,
        )
        with torch.no_grad():
            for start in range(0, len(auxiliary_indices), contexts_per_batch):
                context_indices = auxiliary_indices[start : start + contexts_per_batch]
                sequences = [
                    list(map(int, hidden_prefixes[int(context_index)]["prefix_token_ids"])) + [token_id]
                    for context_index in context_indices for token_id in forced_panel
                ]
                ids, mask, positions = pad_token_rows(sequences, loaded.tokenizer.pad_token_id)
                sample = torch.arange(len(sequences), dtype=torch.long)
                output_logits, observed = loaded.model(
                    ids.to(loaded.device), mask.to(loaded.device), positions.to(loaded.device), sample.to(loaded.device)
                )
                assert_output_order(sample, observed.cpu())
                probabilities = torch.softmax(output_logits[:, :vocabulary_size].float(), dim=-1).cpu().numpy()
                profiles = probabilities.reshape(len(context_indices), forced_candidate_count, vocabulary_size)
                profiles -= profiles.mean(axis=1, keepdims=True)
                profile_sum += profiles.sum(axis=0, dtype=np.float64)
        future_auxiliary_contrast_mean = (
            profile_sum / float(len(auxiliary_indices))
        ).astype(np.float32)
        np.save(auxiliary_future_path, future_auxiliary_contrast_mean, allow_pickle=False)
        atomic_json(auxiliary_future_meta_path, {
            "cache_key": auxiliary_cache_key,
            "path": str(auxiliary_future_path),
            "sha256": file_sha256(auxiliary_future_path),
            "shape": list(future_auxiliary_contrast_mean.shape),
        })

    checkpoint = read_json(checkpoint_path) if checkpoint_path.is_file() and not args.force else {}
    valid_checkpoint = checkpoint.get("checkpoint_key") == checkpoint_key
    future_rows = list(checkpoint.get("rows", [])) if valid_checkpoint else []
    profile_rows = list(checkpoint.get("profiles", [])) if valid_checkpoint else []
    completed_pairs = {str(row["pair_id"]) for row in profile_rows}
    resumed_profiles = len(profile_rows)
    started = time.monotonic()

    with torch.no_grad():
        for pair_number, pair in enumerate(pairs, start=1):
            pair_id = str(pair["pair_id"])
            if pair_id in completed_pairs:
                continue
            sequences: list[list[int]] = []
            for token_id in pair["forced_token_ids"]:
                sequences.append(list(map(int, prefix_by_id[pair["left_prefix_id"]]["prefix_token_ids"])) + [token_id])
                sequences.append(list(map(int, prefix_by_id[pair["right_prefix_id"]]["prefix_token_ids"])) + [token_id])
            ids, mask, positions = pad_token_rows(sequences, loaded.tokenizer.pad_token_id)
            sample = torch.arange(len(sequences), dtype=torch.long)
            output_logits, observed = loaded.model(
                ids.to(loaded.device), mask.to(loaded.device), positions.to(loaded.device), sample.to(loaded.device)
            )
            assert_output_order(sample, observed.cpu())
            valid_logits = output_logits[:, :vocabulary_size].float()
            probabilities = torch.softmax(valid_logits, dim=-1).cpu().numpy()
            values = valid_logits.cpu().numpy()
            left_logits, right_logits = values[0::2], values[1::2]
            left_probabilities, right_probabilities = probabilities[0::2], probabilities[1::2]
            divergences = paired_js_from_logits(left_logits, right_logits)
            left_top5 = np.argpartition(left_logits, -5, axis=1)[:, -5:]
            right_top5 = np.argpartition(right_logits, -5, axis=1)[:, -5:]
            for cell_index, token_id in enumerate(pair["forced_token_ids"]):
                overlap = float(np.mean([
                    token in set(map(int, right_top5[cell_index])) for token in map(int, left_top5[cell_index])
                ]))
                future_rows.append({
                    "pair_id": pair_id,
                    "split": pair["split"],
                    "forced_token_id": token_id,
                    "forced_token_text": loaded.tokenizer.decode(
                        [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False,
                    ),
                    "future_js": float(divergences[cell_index]),
                    "future_normalized_js": float(divergences[cell_index] / LOG_TWO),
                    "future_top1_agreement": bool(left_logits[cell_index].argmax() == right_logits[cell_index].argmax()),
                    "future_top5_overlap": overlap,
                })

            token_axes = np.asarray([candidate_axis[int(token_id)] for token_id in pair["forced_token_ids"]], dtype=np.int64)
            token_mean = auxiliary_token_mean[token_axes]
            left_hidden = double_center_hidden_profile(
                np.asarray(successor_states[hidden_prefix_axis[pair["left_prefix_id"]], token_axes], dtype=np.float64),
                token_mean,
            )
            right_hidden = double_center_hidden_profile(
                np.asarray(successor_states[hidden_prefix_axis[pair["right_prefix_id"]], token_axes], dtype=np.float64),
                token_mean,
            )
            profile_rows.append({
                "pair_id": pair_id,
                "split": pair["split"],
                "forced_token_ids": pair["forced_token_ids"],
                "contrast_rank": contrast_rank,
                **profile_geometry(
                    left_hidden, right_hidden,
                    center_candidate_rows(left_probabilities) - future_auxiliary_contrast_mean,
                    center_candidate_rows(right_probabilities) - future_auxiliary_contrast_mean,
                    rank=contrast_rank,
                ),
            })
            atomic_json(checkpoint_path, {
                "checkpoint_key": checkpoint_key, "complete": False,
                "rows": future_rows, "profiles": profile_rows,
            })
            completed = len(profile_rows)
            rate = (completed - resumed_profiles) / max(time.monotonic() - started, 1e-9)
            print(
                f"[response_state] contrast profiles={completed}/{len(pairs)} "
                f"rate={rate:.2f}/s eta={(len(pairs)-completed)/max(rate,1e-9)/60:.1f}m",
                flush=True,
            )

    atomic_json(checkpoint_path, {
        "checkpoint_key": checkpoint_key, "complete": True,
        "rows": future_rows, "profiles": profile_rows,
    })
    level_rows = _aggregate_future_rows(pairs, future_rows)
    profiles_by_pair = {str(row["pair_id"]): row for row in profile_rows}
    pair_rows = []
    for row in level_rows:
        profile = profiles_by_pair[str(row["pair_id"])]
        pair_rows.append({**row, **{key: value for key, value in profile.items() if key != "grams"}})
    development = [row for row in pair_rows if row["split"] == "development"]
    evaluation = [row for row in pair_rows if row["split"] == "evaluation"]
    evaluation_profiles = [row for row in profile_rows if row["split"] == "evaluation"]
    summary = {
        "claim_tested": "current-distribution sufficiency and one-step propagation of candidate-contrast geometry",
        "future_definition": "one-step next-token distribution after forcing the same candidate x: p(. | context + x)",
        "pair_selection": "different problems; same top-1 token, prefix-length bin, and reasoning-progress bin; maximize top-5/top-20 overlap then minimize full-vocabulary JS",
        "forced_candidate_selection": "one fixed high-average-logit panel selected on development contexts only, restricted to the studied candidate set, and fixed before evaluation outcomes",
        "selected_layer": selected_layer,
        "contrast_rank": contrast_rank,
        "contrast_auxiliary_contexts": len(auxiliary_indices),
        "development_pairs": len(development),
        "evaluation_pairs": len(evaluation),
        "forced_candidates_per_pair": forced_candidate_count,
        "current_distribution_match": bootstrap_statistic(
            evaluation, lambda rows: float(np.mean([row["current_normalized_js"] for row in rows])),
            replicates=replicates, seed=seed + 1, ci=ci,
        ),
        "future_divergence": bootstrap_statistic(
            evaluation, lambda rows: float(np.mean([row["future_normalized_js"] for row in rows])),
            replicates=replicates, seed=seed + 2, ci=ci,
        ),
        "post_x_minus_current_js": bootstrap_statistic(
            evaluation,
            lambda rows: float(np.mean([
                row["future_normalized_js"] - row["current_normalized_js"] for row in rows
            ])),
            replicates=replicates, seed=seed + 7, ci=ci,
        ),
        "contrast_geometry": {
            "definition": "double-centered hidden responses R_i(X) versus double-centered one-step probability profiles H Phi_i(X) - mean_aux H Phi_a(X)",
            "candidate_identity_alignment": candidate_alignment_permutation_test(
                evaluation_profiles, replicates=replicates, seed=seed + 10, ci=ci,
            ),
            "response_law_distance_correspondence": geometry_association_summary(
                evaluation, "hidden_contrast_law_distance", "future_contrast_law_distance",
                replicates=replicates, seed=seed + 20, ci=ci,
            ),
            "candidate_kernel_distance_correspondence": geometry_association_summary(
                evaluation, "hidden_candidate_kernel_distance", "future_candidate_kernel_distance",
                replicates=replicates, seed=seed + 30, ci=ci,
            ),
            "subspace_distance_correspondence": geometry_association_summary(
                evaluation, "hidden_contrast_subspace_distance", "future_contrast_subspace_distance",
                replicates=replicates, seed=seed + 40, ci=ci,
            ),
        },
        "retired_level_test": {
            "status": "not_computed_or_interpreted",
            "reason": "double-centered interaction geometry cannot be tested against absolute between-problem future-distribution levels",
        },
        "inference_units": "evaluation uses context-disjoint pairs; pair bootstrap for intervals; within-context candidate-label permutation for contrast alignment",
        "interpretation": {
            "positive_candidate_alignment": "candidate contrast geometry at the selected hidden layer propagates to the next probability-profile geometry",
            "positive_distance_correspondence": "contexts with more different hidden response fields also have more different one-step output contrast fields",
            "scope": "tests one-step post-candidate functional state, not complete multi-token reasoning sufficiency",
        },
    }
    pairs_path = output_root / "response_law_pairs.csv"
    futures_path = output_root / "response_law_future_cells.csv"
    profiles_path = output_root / "response_law_contrast_profiles.json"
    summary_path = root / "metrics/response_law_state_summary.json"
    _write_csv(pairs_path, pair_rows)
    _write_csv(futures_path, future_rows)
    atomic_json(profiles_path, {"implementation_version": IMPLEMENTATION_VERSION, "profiles": profile_rows})
    atomic_json(summary_path, summary)
    atomic_json(manifest_path, {
        "complete": True, "config_hash": stable_hash(config), **inputs,
        "pairs": str(pairs_path), "pairs_sha256": file_sha256(pairs_path),
        "future_cells": str(futures_path), "future_cells_sha256": file_sha256(futures_path),
        "contrast_profiles": str(profiles_path), "contrast_profiles_sha256": file_sha256(profiles_path),
        "auxiliary_future_contrast": str(auxiliary_future_path),
        "auxiliary_future_contrast_sha256": file_sha256(auxiliary_future_path),
        "summary": str(summary_path), "summary_sha256": file_sha256(summary_path),
        "model": loaded.metadata,
    })
    print(manifest_path)


if __name__ == "__main__":
    main()
