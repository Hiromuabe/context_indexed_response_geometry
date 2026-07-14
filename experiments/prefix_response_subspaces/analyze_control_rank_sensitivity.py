from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .analyze_paper_geometry import _equal_energy_global_basis, _resolve_conditional_basis, _stratum
from .src.statistics import problem_bootstrap, problem_ratio_bootstrap
from .src.storage import load_residual_entry
from .src.utils import (
    atomic_json,
    file_sha256,
    load_config,
    read_json,
    read_jsonl,
    result_root,
    stable_hash,
    stage_is_complete,
)


DEFAULT_RANK_CURVE = [8, 16, 32, 64, 96, 127]
DEFAULT_MATCH_GRID = list(range(64, 192))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _exact_for_control(row: dict[str, Any], control: str) -> bool:
    if control == "matched_common":
        return _truthy(row.get("matched_common_exact_bin", True))
    if control == "wrong_context":
        return _truthy(row.get("wrong_context_exact_bin", True))
    raise ValueError(f"unknown control: {control}")


def _sample_eigensystem(samples: np.ndarray, maximum_rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Return left singular vectors/values for projecting without hidden-size bases."""
    matrix = np.asarray(samples, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("training samples must be a finite matrix")
    eigenvalues, eigenvectors = np.linalg.eigh(matrix @ matrix.T)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0.0, None)
    eigenvectors = eigenvectors[:, order]
    tolerance = eigenvalues[0] * max(matrix.shape) * np.finfo(np.float64).eps if len(eigenvalues) else 0.0
    effective = min(int(maximum_rank), int(np.sum(eigenvalues > tolerance)))
    if effective <= 0:
        raise RuntimeError("training residuals have zero numerical rank")
    return eigenvectors[:, :effective], np.sqrt(eigenvalues[:effective])


def _sample_projection_curve(
    target: np.ndarray,
    training: np.ndarray,
    eigensystem: tuple[np.ndarray, np.ndarray],
    ranks: list[int],
) -> dict[int, float]:
    target64 = np.asarray(target, dtype=np.float64)
    training64 = np.asarray(training, dtype=np.float64)
    left, singular = eigensystem
    denominator = float(np.square(target64).sum())
    if denominator <= 1e-12:
        return {int(rank): float("nan") for rank in ranks}
    coefficients = (target64 @ training64.T @ left) / singular[None, :]
    cumulative = np.cumsum(np.square(coefficients), axis=1).sum(axis=0) / denominator
    return {
        int(rank): float(cumulative[int(rank) - 1]) if int(rank) <= len(cumulative) else float("nan")
        for rank in ranks
    }


def _basis_projection_curve(target: np.ndarray, basis: np.ndarray | None, ranks: list[int]) -> dict[int, float]:
    if basis is None:
        return {int(rank): float("nan") for rank in ranks}
    target64 = np.asarray(target, dtype=np.float64)
    basis64 = np.asarray(basis, dtype=np.float64)
    denominator = float(np.square(target64).sum())
    if denominator <= 1e-12:
        return {int(rank): float("nan") for rank in ranks}
    cumulative = np.cumsum(np.square(target64 @ basis64), axis=1).sum(axis=0) / denominator
    return {
        int(rank): float(cumulative[int(rank) - 1]) if int(rank) <= len(cumulative) else float("nan")
        for rank in ranks
    }


def _conditional_bases(train: np.ndarray, prefixes: list[dict[str, Any]], nonaux: np.ndarray, rank: int) -> dict[tuple[int, int], np.ndarray | None]:
    positions: dict[tuple[int, int], list[int]] = defaultdict(list)
    for local_index, full_index in enumerate(nonaux):
        prefix = prefixes[int(full_index)]
        if prefix["problem_group"] == "analysis_train":
            positions[_stratum(prefix)].append(local_index)
    return {
        key: _equal_energy_global_basis(train[indices], rank)
        for key, indices in positions.items()
    }


def _auxiliary_terms(
    z: np.ndarray,
    auxiliary: np.ndarray,
    train_tokens: np.ndarray,
    evaluation_tokens: np.ndarray,
    chunk_size: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    hidden = int(z.shape[-1])
    token_sum = np.zeros((len(evaluation_tokens), hidden), dtype=np.float64)
    grand_sum = np.zeros(hidden, dtype=np.float64)
    for start in range(0, len(auxiliary), chunk_size):
        chunk = auxiliary[start : start + chunk_size]
        token_sum += np.asarray(z[np.ix_(chunk, evaluation_tokens)], dtype=np.float64).sum(axis=0)
        grand_sum += np.asarray(z[np.ix_(chunk, train_tokens)], dtype=np.float64).sum(axis=(0, 1))
    return token_sum / len(auxiliary), grand_sum / (len(auxiliary) * len(train_tokens))


def _problem_mean(rows: list[dict[str, Any]], key: str) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = float(row[key])
        if np.isfinite(value):
            grouped[str(row["problem_id"])].append(value)
    return float(np.mean([np.mean(values) for values in grouped.values()])) if grouped else float("nan")


def mean_shift_energy_fraction(samples: np.ndarray, eps: float = 1e-12) -> float:
    """Fraction of residual energy carried by the candidate-constant row mean."""
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("samples must be a finite [candidate, hidden] matrix")
    denominator = float(np.square(values).sum())
    if denominator <= eps:
        return float("nan")
    mean = values.mean(axis=0)
    return float(len(values) * np.square(mean).sum() / denominator)


def select_ev_matching_rank(rows: list[dict[str, Any]], control: str, ranks: list[int]) -> dict[str, Any]:
    """Select one global control rank from exact-bin development problems only."""
    candidates = []
    for rank in ranks:
        selected = [
            row for row in rows
            if row["split"] == "development"
            and row["control"] == control
            and int(row["control_rank"]) == int(rank)
            and _exact_for_control(row, control)
        ]
        target_mean = _problem_mean(selected, "ev_target_rank64")
        control_mean = _problem_mean(selected, "ev_control")
        candidates.append({
            "rank": int(rank),
            "target_rank64_mean_ev": target_mean,
            "control_mean_ev": control_mean,
            "target_minus_control_ev": target_mean - control_mean,
            "absolute_ev_gap": abs(target_mean - control_mean),
            "n_problems": len({row["problem_id"] for row in selected}),
        })
    eligible = [row for row in candidates if np.isfinite(row["absolute_ev_gap"])]
    if not eligible:
        raise RuntimeError(f"no finite development EV values for {control}")
    selected = min(eligible, key=lambda row: (row["absolute_ev_gap"], row["rank"]))
    return {"selected_rank": int(selected["rank"]), "selected": selected, "candidates": candidates}


def _bootstrap(rows: list[dict[str, Any]], key: str, config: dict[str, Any], seed_offset: int) -> dict[str, float]:
    return problem_bootstrap(
        np.asarray([float(row[key]) for row in rows], dtype=np.float64),
        np.asarray([row["problem_id"] for row in rows]),
        replicates=int(config["statistics"]["bootstrap_replicates"]),
        seed=int(config["seed"]) + int(seed_offset),
        ci=float(config["statistics"]["ci"]),
    )


def _summarize_selected_match(
    rows: list[dict[str, Any]], control: str, rank: int, config: dict[str, Any], seed_offset: int
) -> dict[str, Any]:
    selected = [
        row for row in rows
        if row["split"] == "evaluation"
        and row["control"] == control
        and int(row["control_rank"]) == int(rank)
        and _exact_for_control(row, control)
    ]
    target = np.asarray([float(row["ev_target_rank64"]) for row in selected], dtype=np.float64)
    gap = np.asarray([float(row["target_minus_control_ev"]) for row in selected], dtype=np.float64)
    problem_ids = np.asarray([row["problem_id"] for row in selected])
    return {
        "control_rank": int(rank),
        "population": "exact-bin evaluation problems",
        "target_rank64_ev": _bootstrap(selected, "ev_target_rank64", config, seed_offset),
        "control_ev": _bootstrap(selected, "ev_control", config, seed_offset + 1),
        "target_minus_control_ev": _bootstrap(selected, "target_minus_control_ev", config, seed_offset + 2),
        "relative_target_minus_control_ev": problem_ratio_bootstrap(
            gap,
            target,
            problem_ids,
            replicates=int(config["statistics"]["bootstrap_replicates"]),
            seed=int(config["seed"]) + seed_offset + 3,
            ci=float(config["statistics"]["ci"]),
        ),
        "n_rows": len(selected),
    }


def _row_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return str(row["problem_id"]), str(row["prefix_id"]), int(row["fold"])


def _summarize_cached_rows(
    curve_rows: list[dict[str, Any]],
    match_rows: list[dict[str, Any]],
    inductive_rows: list[dict[str, Any]],
    rank_curve: list[int],
    match_grid: list[int],
    target_rank: int,
    inductive_rank: int,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    rank_summary: dict[str, Any] = {}
    for rank in rank_curve:
        selected = [row for row in curve_rows if int(row["rank"]) == rank]
        common_exact = [row for row in selected if _exact_for_control(row, "matched_common")]
        wrong_exact = [row for row in selected if _exact_for_control(row, "wrong_context")]
        rank_summary[str(rank)] = {
            "population": {
                "ev_target": "all evaluation problems",
                "delta_common": "matched-common exact-bin evaluation problems",
                "delta_wrong": "wrong-context exact-bin evaluation problems",
            },
            "ev_target": _bootstrap(selected, "ev_target", config, 2100 + rank),
            "delta_common": _bootstrap(common_exact, "delta_common", config, 2200 + rank),
            "delta_wrong": _bootstrap(wrong_exact, "delta_wrong", config, 2300 + rank),
        }

    common_selection = select_ev_matching_rank(match_rows, "matched_common", match_grid)
    wrong_selection = select_ev_matching_rank(match_rows, "wrong_context", match_grid)
    matched_evaluation = {
        "matched_common": _summarize_selected_match(match_rows, "matched_common", common_selection["selected_rank"], config, 2401),
        "wrong_context": _summarize_selected_match(match_rows, "wrong_context", wrong_selection["selected_rank"], config, 2411),
    }

    primary_rank_rows = {
        _row_key(row): row for row in curve_rows if int(row["rank"]) == int(inductive_rank)
    }
    paired_rows: list[dict[str, Any]] = []
    for row in inductive_rows:
        primary = primary_rank_rows.get(_row_key(row))
        if primary is None:
            continue
        paired_rows.append({
            **row,
            "inductive_minus_primary_ev_target": float(row["ev_target"]) - float(primary["ev_target"]),
            "inductive_minus_primary_delta_common": float(row["delta_common"]) - float(primary["delta_common"]),
            "inductive_minus_primary_delta_wrong": float(row["delta_wrong"]) - float(primary["delta_wrong"]),
        })
    common_inductive = [row for row in inductive_rows if _exact_for_control(row, "matched_common")]
    wrong_inductive = [row for row in inductive_rows if _exact_for_control(row, "wrong_context")]
    common_paired = [row for row in paired_rows if _exact_for_control(row, "matched_common")]
    wrong_paired = [row for row in paired_rows if _exact_for_control(row, "wrong_context")]
    inductive_summary = {
        "name": "evaluation-fold-independent target-context centering",
        "role": "sensitivity analysis; not a replacement for the split-local centered primary estimand",
        "rank": inductive_rank,
        "rho_definition": "rho_i = |S_test| * ||mean_j r_ind_ij||_2^2 / sum_j ||r_ind_ij||_2^2",
        "population": {
            "ev_target": "all evaluation problems",
            "delta_common": "matched-common exact-bin evaluation problems",
            "delta_wrong": "wrong-context exact-bin evaluation problems",
        },
        "ev_target": _bootstrap(inductive_rows, "ev_target", config, 2501),
        "delta_common": _bootstrap(common_inductive, "delta_common", config, 2502),
        "delta_wrong": _bootstrap(wrong_inductive, "delta_wrong", config, 2503),
        "mean_shift_energy_fraction_rho": _bootstrap(inductive_rows, "mean_shift_energy_fraction_rho", config, 2504),
        "paired_inductive_minus_primary": {
            "ev_target": _bootstrap(paired_rows, "inductive_minus_primary_ev_target", config, 2511),
            "delta_common": _bootstrap(common_paired, "inductive_minus_primary_delta_common", config, 2512),
            "delta_wrong": _bootstrap(wrong_paired, "inductive_minus_primary_delta_wrong", config, 2513),
        },
        "n_rows": len(inductive_rows),
    }
    selection_summary = {
        "definition": {
            "selection_split": "analysis_dev exact-bin problems only",
            "selection_unit": "global rank per control after equal-weight problem aggregation",
            "criterion": "minimum absolute difference between mean target-rank-64 EV and mean control EV; smaller rank breaks ties",
            "evaluation_split_used_for_selection": False,
            "target_rank": target_rank,
            "control_rank_grid": match_grid,
            "maximum_control_rank_reason": "Each fold has 192 fit candidates; centering reduces the algebraic rank ceiling to 191.",
            "relative_gap_definition": "sum(target_rank64_ev - control_ev) / sum(target_rank64_ev), with whole problems resampled for the CI",
            "wrong_context_aggregation": "mean over the five donor-context EV values within each problem-fold before problem-level aggregation and bootstrap",
            "recommended_manuscript_label": "maximum-rank control stress test; the controls did not achieve EV matching",
            "control_definitions": {"matched_common": "conditioned common space; exact bins only", "wrong_context": "mean over five exact-bin wrong contexts within cell"},
        },
        "matched_common": common_selection,
        "wrong_context": wrong_selection,
        "evaluation_achieved_match": matched_evaluation,
    }
    return rank_summary, selection_summary, inductive_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--results-root", help="override config results_root; useful for fixed replications")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summary-only", action="store_true", help="refresh summaries from saved CSV rows without recomputing SVDs")
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(args.results_root) if args.results_root else result_root(config)
    settings = config.get("additional_experiments", {})
    rank_curve = sorted(set(map(int, settings.get("rank_curve_ranks", DEFAULT_RANK_CURVE))))
    if "ev_match_control_ranks" in settings:
        match_grid = sorted(set(map(int, settings["ev_match_control_ranks"])))
    else:
        minimum = int(settings.get("ev_match_control_rank_min", 64))
        maximum = int(settings.get("ev_match_control_rank_max", 191))
        if minimum > maximum:
            raise ValueError("ev_match_control_rank_min must not exceed ev_match_control_rank_max")
        match_grid = list(range(minimum, maximum + 1))
    target_rank = int(settings.get("ev_match_target_rank", 64))
    inductive_rank = int(settings.get("inductive_rank", 64))
    if target_rank not in rank_curve:
        rank_curve = sorted(set(rank_curve + [target_rank]))
    maximum_rank = max(rank_curve + match_grid + [inductive_rank])

    curve_path = root / "metrics/control_rank_curve_rows.csv"
    match_path = root / "metrics/ev_match_rank_rows.csv"
    selection_path = root / "metrics/ev_matched_rank_selection.json"
    inductive_path = root / "metrics/inductive_centering_rows.csv"
    inductive_summary_path = root / "metrics/inductive_centering_summary.json"
    summary_path = root / "metrics/control_rank_sensitivity_summary.json"
    if args.summary_only:
        rank_summary, selection_summary, inductive_summary = _summarize_cached_rows(
            _read_csv(curve_path),
            _read_csv(match_path),
            _read_csv(inductive_path),
            rank_curve,
            match_grid,
            target_rank,
            inductive_rank,
            config,
        )
        overall = read_json(summary_path)
        overall["rank_curve"] = rank_summary
        overall["ev_matched_rank_selection"] = selection_summary
        overall["inductive_centering"] = inductive_summary
        atomic_json(selection_path, selection_summary)
        atomic_json(inductive_summary_path, inductive_summary)
        atomic_json(summary_path, overall)
        print(summary_path)
        return

    residual_path = root / "manifests/residuals.json"
    hidden_path = root / "manifests/hidden_states.json"
    geometry_path = root / "metrics/paper_geometry_summary.json"
    wrong_path = root / "controls/wrong_prefixes.jsonl"
    inputs = {
        "residuals_sha256": file_sha256(residual_path),
        "hidden_states_sha256": file_sha256(hidden_path),
        "geometry_sha256": file_sha256(geometry_path),
        "wrong_prefixes_sha256": file_sha256(wrong_path),
        "settings_sha256": stable_hash({"rank_curve": rank_curve, "match_grid": match_grid, "target_rank": target_rank, "inductive_rank": inductive_rank}),
    }
    manifest_path = root / "manifests/control_rank_sensitivity.json"
    if not args.force and stage_is_complete(manifest_path, config, inputs):
        print(manifest_path)
        return

    residual_manifest = read_json(residual_path)
    hidden_manifest = read_json(hidden_path)
    geometry = read_json(geometry_path)
    prefixes = read_jsonl(hidden_manifest["prefix_snapshot"])
    wrong_rows = read_jsonl(wrong_path)
    wrong_map = {row["prefix_id"]: list(row["wrong_prefix_ids"]) for row in wrong_rows}
    relaxed_wrong = {row["prefix_id"] for row in wrong_rows if int(row.get("relaxed_length_wrong_prefixes", 0)) > 0}
    layer = int(geometry["selected_layer"])
    layer_entry = next(entry for entry in hidden_manifest["layers"] if int(entry["layer"]) == layer)
    z = np.load(layer_entry["successor_path"], mmap_mode="r")
    auxiliary = np.asarray([i for i, row in enumerate(prefixes) if row["problem_group"] == "auxiliary"], dtype=np.int64)
    expected_wrong = int(config.get("controls", {}).get("wrong_prefixes_per_target", 5))

    curve_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    inductive_rows: list[dict[str, Any]] = []
    cache_entries: list[dict[str, Any]] = []
    entries = [entry for entry in residual_manifest["entries"] if int(entry["layer"]) == layer]
    for entry_number, entry in enumerate(entries, start=1):
        bundle = load_residual_entry(entry)
        train = bundle["train_residuals"]
        evaluation = bundle["evaluation_residuals"]
        nonaux = np.asarray(bundle["nonauxiliary_prefix_indices"], dtype=np.int64)
        fold = int(entry["fold"])
        train_tokens = np.asarray(bundle["train_candidate_indices"], dtype=np.int64)
        evaluation_tokens = np.asarray(bundle["evaluation_candidate_indices"], dtype=np.int64)
        if maximum_rank > len(train_tokens) - 1:
            raise ValueError(f"rank {maximum_rank} exceeds centered fold-{fold} training ceiling {len(train_tokens) - 1}")
        position_by_id = {prefixes[int(full)]["prefix_id"]: index for index, full in enumerate(nonaux)}
        conditional = _conditional_bases(train, prefixes, nonaux, maximum_rank)
        systems: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        def system(position: int) -> tuple[np.ndarray, np.ndarray]:
            if position not in systems:
                systems[position] = _sample_eigensystem(train[position], maximum_rank)
            return systems[position]

        auxiliary_token_mean, auxiliary_train_grand = _auxiliary_terms(z, auxiliary, train_tokens, evaluation_tokens)
        print(f"[control_rank] fold={fold} entry={entry_number}/{len(entries)} contexts={len(nonaux)} max_rank={maximum_rank}", flush=True)
        for local_index, full_index in enumerate(nonaux):
            prefix = prefixes[int(full_index)]
            group = prefix["problem_group"]
            if group not in {"analysis_dev", "analysis_test"}:
                continue
            split = "development" if group == "analysis_dev" else "evaluation"
            prefix_id = prefix["prefix_id"]
            target = np.asarray(evaluation[local_index], dtype=np.float32)
            common_basis, common_exact, length_distance, resolved = _resolve_conditional_basis(conditional, prefix)
            wrong_ids = [wrong_id for wrong_id in wrong_map.get(prefix_id, []) if wrong_id in position_by_id]
            target_curve = _sample_projection_curve(target, train[local_index], system(local_index), rank_curve)
            common_all = _basis_projection_curve(target, common_basis, sorted(set(rank_curve + match_grid)))
            wrong_all = [
                _sample_projection_curve(target, train[position_by_id[wrong_id]], system(position_by_id[wrong_id]), sorted(set(rank_curve + match_grid)))
                for wrong_id in wrong_ids
            ]

            if split == "evaluation":
                for rank in rank_curve:
                    wrong_values = [curve[rank] for curve in wrong_all]
                    wrong_mean = float(np.mean(wrong_values)) if len(wrong_values) == expected_wrong and np.isfinite(wrong_values).all() else float("nan")
                    curve_rows.append({
                        "problem_id": prefix["problem_id"], "prefix_id": prefix_id, "fold": fold, "layer": layer, "rank": rank,
                        "ev_target": target_curve[rank], "ev_matched_common": common_all[rank], "ev_wrong_mean": wrong_mean,
                        "delta_common": target_curve[rank] - common_all[rank], "delta_wrong": target_curve[rank] - wrong_mean,
                        "matched_common_exact_bin": bool(common_exact), "matched_common_length_bin_distance": length_distance,
                        "matched_common_resolved_stratum": f"{resolved[0]}:{resolved[1]}" if resolved is not None else "",
                        "wrong_context_count": len(wrong_ids), "wrong_context_exact_bin": prefix_id not in relaxed_wrong,
                    })

            for control in ("matched_common", "wrong_context"):
                for rank in match_grid:
                    if control == "matched_common":
                        control_ev = common_all[rank]
                    else:
                        wrong_values = [curve[rank] for curve in wrong_all]
                        control_ev = float(np.mean(wrong_values)) if len(wrong_values) == expected_wrong and np.isfinite(wrong_values).all() else float("nan")
                    match_rows.append({
                        "problem_id": prefix["problem_id"], "prefix_id": prefix_id, "split": split, "fold": fold, "layer": layer,
                        "control": control, "target_rank": target_rank, "control_rank": rank,
                        "ev_target_rank64": target_curve[target_rank], "ev_control": control_ev,
                        "target_minus_control_ev": target_curve[target_rank] - control_ev,
                        "matched_common_exact_bin": bool(common_exact), "wrong_context_count": len(wrong_ids),
                        "wrong_context_exact_bin": prefix_id not in relaxed_wrong,
                    })

            if split == "evaluation":
                prefix_train_mean = np.asarray(z[int(full_index), train_tokens], dtype=np.float64).mean(axis=0)
                inductive_target = (
                    np.asarray(z[int(full_index), evaluation_tokens], dtype=np.float64)
                    - prefix_train_mean[None]
                    - auxiliary_token_mean
                    + auxiliary_train_grand[None]
                )
                target_ind = _sample_projection_curve(inductive_target, train[local_index], system(local_index), [inductive_rank])[inductive_rank]
                common_ind = _basis_projection_curve(inductive_target, common_basis, [inductive_rank])[inductive_rank]
                wrong_ind_values = [
                    _sample_projection_curve(inductive_target, train[position_by_id[wrong_id]], system(position_by_id[wrong_id]), [inductive_rank])[inductive_rank]
                    for wrong_id in wrong_ids
                ]
                wrong_ind = float(np.mean(wrong_ind_values)) if len(wrong_ind_values) == expected_wrong and np.isfinite(wrong_ind_values).all() else float("nan")
                inductive_rows.append({
                    "problem_id": prefix["problem_id"], "prefix_id": prefix_id, "fold": fold, "layer": layer, "rank": inductive_rank,
                    "ev_target": target_ind, "ev_matched_common": common_ind, "ev_wrong_mean": wrong_ind,
                    "delta_common": target_ind - common_ind, "delta_wrong": target_ind - wrong_ind,
                    "mean_shift_energy_fraction_rho": mean_shift_energy_fraction(inductive_target),
                    "matched_common_exact_bin": bool(common_exact), "wrong_context_count": len(wrong_ids),
                    "wrong_context_exact_bin": prefix_id not in relaxed_wrong,
                })

        cache_root = root / "subspaces/control_rank_cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        local_positions = np.asarray(sorted(systems), dtype=np.int64)
        local_left = np.zeros((len(local_positions), len(train_tokens), maximum_rank), dtype=np.float32)
        local_singular = np.zeros((len(local_positions), maximum_rank), dtype=np.float64)
        local_effective = np.zeros(len(local_positions), dtype=np.int64)
        for cache_axis, position in enumerate(local_positions):
            left, singular = systems[int(position)]
            effective = len(singular)
            local_left[cache_axis, :, :effective] = left.astype(np.float32)
            local_singular[cache_axis, :effective] = singular
            local_effective[cache_axis] = effective
        common_keys = np.asarray(sorted(conditional), dtype=np.int64)
        common_bases = np.zeros((len(common_keys), int(z.shape[-1]), maximum_rank), dtype=np.float32)
        common_effective = np.zeros(len(common_keys), dtype=np.int64)
        for cache_axis, key in enumerate(map(tuple, common_keys.tolist())):
            basis = conditional[key]
            if basis is None:
                continue
            effective = int(basis.shape[1])
            common_bases[cache_axis, :, :effective] = np.asarray(basis, dtype=np.float32)
            common_effective[cache_axis] = effective
        cache_path = cache_root / f"fold_{fold}.npz"
        np.savez(
            cache_path,
            local_positions=local_positions,
            local_left_singular_vectors=local_left,
            local_singular_values=local_singular,
            local_effective_ranks=local_effective,
            common_strata=common_keys,
            common_bases=common_bases,
            common_effective_ranks=common_effective,
            train_candidate_indices=train_tokens,
            maximum_rank=np.asarray(maximum_rank, dtype=np.int64),
        )
        cache_entries.append({
            "fold": fold, "path": str(cache_path), "sha256": file_sha256(cache_path),
            "maximum_rank": maximum_rank, "local_systems": len(local_positions), "common_strata": len(common_keys),
            "local_left_storage_dtype": "float32", "local_singular_storage_dtype": "float64", "common_basis_storage_dtype": "float32",
        })

    rank_summary, selection_summary, inductive_summary = _summarize_cached_rows(
        curve_rows,
        match_rows,
        inductive_rows,
        rank_curve,
        match_grid,
        target_rank,
        inductive_rank,
        config,
    )
    overall = {
        "model": hidden_manifest.get("model", {}),
        "selected_layer": layer,
        "rank_curve_ranks": rank_curve,
        "rank_curve": rank_summary,
        "ev_matched_rank_selection": selection_summary,
        "inductive_centering": inductive_summary,
        "rank_basis_cache": {"maximum_rank": maximum_rank, "entries": cache_entries},
        "coverage": {
            "rank_curve_rows": len(curve_rows), "ev_match_rows": len(match_rows), "inductive_rows": len(inductive_rows),
            "evaluation_problems": len({row["problem_id"] for row in curve_rows}),
        },
    }

    _write_csv(curve_path, curve_rows)
    _write_csv(match_path, match_rows)
    _write_csv(inductive_path, inductive_rows)
    atomic_json(selection_path, selection_summary)
    atomic_json(inductive_summary_path, inductive_summary)
    atomic_json(summary_path, overall)
    atomic_json(manifest_path, {
        "complete": True,
        "config_hash": stable_hash(config),
        **inputs,
        "curve_rows": str(curve_path),
        "ev_match_rows": str(match_path),
        "rank_selection": str(selection_path),
        "inductive_rows": str(inductive_path),
        "inductive_summary": str(inductive_summary_path),
        "summary": str(summary_path),
        "rank_basis_cache": cache_entries,
    })
    print(summary_path)


if __name__ == "__main__":
    main()
