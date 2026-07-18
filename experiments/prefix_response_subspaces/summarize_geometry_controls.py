from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

from .src.statistics import problem_bootstrap
from .src.utils import atomic_json, ensure_layout, load_config, read_json, read_jsonl


_DESIGN_FIELDS = {
    "seed": ("seed",),
    "prefix_pool_size": ("data", "prefix_pool_size"),
    "evaluation_prefixes": ("data", "evaluation_prefixes"),
    "auxiliary_prefixes": ("data", "auxiliary_prefixes"),
    "analysis_dev_prefixes": ("data", "analysis_dev_prefixes"),
    "analysis_train_prefixes": ("data", "analysis_train_prefixes"),
    "candidate_selection_prefixes": ("data", "candidate_selection_prefixes"),
    "prefixes_per_problem": ("data", "prefixes_per_problem"),
    "min_prefix_tokens": ("data", "min_prefix_tokens"),
    "min_remaining_tokens": ("data", "min_remaining_tokens"),
    "position_strata": ("data", "position_strata"),
    "length_bins": ("data", "length_bins"),
    "candidate_total": ("candidates", "total"),
    "candidate_calibration": ("candidates", "calibration"),
    "candidate_analysis": ("candidates", "analysis"),
    "candidate_folds": ("candidates", "folds"),
    "proposal_top_k": ("candidates", "proposal_top_k"),
    "probability_bands": ("candidates", "probability_bands"),
    "coverage_bands": ("candidates", "coverage_bands"),
    "wrong_prefixes_per_target": ("controls", "wrong_prefixes_per_target"),
    "prefer_same_last_token": ("controls", "prefer_same_last_token"),
    "ranks": ("analysis", "ranks"),
    "permutation_replicates": ("permutation", "replicates"),
    "permutation_minimum_stratum_size": ("permutation", "minimum_stratum_size"),
    "permutation_minimum_exchangeable_fraction": ("permutation", "minimum_exchangeable_prefix_fraction"),
    "bootstrap_replicates": ("statistics", "bootstrap_replicates"),
    "confidence_level": ("statistics", "ci"),
}


_BLUE = "#2F6B9A"
_ORANGE = "#D8873A"
_GREEN = "#3F7D68"
_INK = "#27313B"
_MID = "#707A86"
_GRID = "#E7EBEF"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _is_true(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def _wrong_exact_by_prefix(root: Path) -> dict[str, bool]:
    path = root / "controls/wrong_prefixes.jsonl"
    if not path.is_file():
        return {}
    return {
        row["prefix_id"]: int(row.get("relaxed_length_wrong_prefixes", 0)) == 0
        for row in read_jsonl(path)
    }


def _conditional_global_is_exact(row: dict[str, str]) -> bool:
    if "conditional_global_exact_length_bin" in row:
        return _is_true(row["conditional_global_exact_length_bin"])
    if "conditional_global_length_bin_distance" in row and row["conditional_global_length_bin_distance"] != "":
        return int(float(row["conditional_global_length_bin_distance"])) == 0
    if "conditional_global_resolved_stratum" in row and "conditional_stratum" in row:
        return row["conditional_global_resolved_stratum"] == row["conditional_stratum"]
    # Legacy GSM8K rows predate the recorded fallback.  That implementation
    # emitted a row only when the requested conditional stratum existed.
    return True


def _wrong_control_is_exact(row: dict[str, str], exact_by_prefix: dict[str, bool]) -> bool:
    if "wrong_control_exact_length_bin" in row:
        return _is_true(row["wrong_control_exact_length_bin"])
    return exact_by_prefix.get(row.get("prefix_id", ""), True)


def _nested(config, path):
    value = config
    for key in path:
        value = value[key]
    return value


def _dataset_label(config: dict) -> str:
    profile = config.get("profile", "").lower()
    trajectories = config.get("data", {}).get("trajectories_jsonl", "").lower()
    identity = f"{profile} {trajectories}"
    if "commonsenseqa" in identity:
        return "CommonsenseQA"
    if "gsm8k" in identity:
        return "GSM8K"
    return config.get("profile", "External dataset")


def _design_audit(config):
    reference_path = config.get("design_reference_config")
    if not reference_path:
        return None
    reference = load_config(reference_path)
    fields = {
        name: {
            "current": _nested(config, path),
            "reference": _nested(reference, path),
            "matches": _nested(config, path) == _nested(reference, path),
        }
        for name, path in _DESIGN_FIELDS.items()
    }
    current_generation = load_config(config["data"]["trajectory_generation_config"])
    reference_generation = load_config(reference["data"]["trajectory_generation_config"])
    current_maximum = current_generation["dataset"]["max_problems"]
    reference_maximum = reference_generation["dataset"]["max_problems"]
    fields["trajectory_max_problems"] = {
        "current": current_maximum,
        "reference": reference_maximum,
        "matches": current_maximum == reference_maximum,
    }
    return {
        "reference_config": str(reference_path),
        "group_assignment": "same assign_groups implementation; deterministic problem-level partition with seed",
        "candidate_partition": "same stratified_partition implementation; calibration plus held-out candidate folds",
        "inference": "same problem-level bootstrap and within-stratum fitted-basis label permutation",
        "intentional_fixed_condition_difference": "CommonsenseQA fixes block 0 and rank 64 before evaluation; GSM8K selected its primary layer/rank on development data.",
        "fields": fields,
        "all_fields_match": all(row["matches"] for row in fields.values()),
    }


def _estimate(rows, field, *, replicates, seed, ci):
    return problem_bootstrap(
        np.asarray([float(row[field]) for row in rows], dtype=np.float64),
        np.asarray([row["problem_id"] for row in rows]),
        replicates=replicates,
        seed=seed,
        ci=ci,
    )


def _difference(rows, left, right, *, replicates, seed, ci):
    return problem_bootstrap(
        np.asarray([float(row[left]) - float(row[right]) for row in rows], dtype=np.float64),
        np.asarray([row["problem_id"] for row in rows]),
        replicates=replicates,
        seed=seed,
        ci=ci,
    )


def summarize(config_path: str) -> tuple[Path, dict]:
    config = load_config(config_path)
    root = ensure_layout(config)
    geometry = read_json(root / "metrics/paper_geometry_summary.json")
    selected_layer = int(geometry["selected_layer"])
    expected_wrong = int(config["controls"]["wrong_prefixes_per_target"])
    replicates = int(config["statistics"]["bootstrap_replicates"])
    ci = float(config["statistics"]["ci"])
    seed = int(config["seed"])
    analysis_candidates = int(config["candidates"]["analysis"])
    candidate_folds = int(config["candidates"]["folds"])
    if analysis_candidates % candidate_folds:
        raise ValueError("analysis candidate count must be divisible by candidate folds")
    heldout_candidates = analysis_candidates // candidate_folds

    ev_rows = _read_csv(root / "metrics/paper_geometry_rows.csv")
    wrong_exact_by_prefix = _wrong_exact_by_prefix(root)
    rotation_path = root / "metrics/paper_rotation_rank_rows.csv"
    if not rotation_path.is_file():
        raise FileNotFoundError(
            f"{rotation_path} is missing; set analysis.report_multirank_controls=true "
            "and rerun analyze_paper_geometry"
        )
    rotation_rows = _read_csv(rotation_path)
    reports = []
    for rank in map(int, config["analysis"]["ranks"]):
        selected_ev = [
            row for row in ev_rows
            if row["split"] == "evaluation"
            and int(row["layer"]) == selected_layer
            and int(row["rank"]) == rank
            and _conditional_global_is_exact(row)
            and _wrong_control_is_exact(row, wrong_exact_by_prefix)
            and int(row["wrong_prefix_count"]) == expected_wrong
        ]
        selected_rotation = [
            row for row in rotation_rows
            if int(row["layer"]) == selected_layer
            and int(row["rank"]) == rank
            and _wrong_control_is_exact(row, wrong_exact_by_prefix)
            and int(row["wrong_prefix_count"]) == expected_wrong
        ]
        if not selected_ev or not selected_rotation:
            raise RuntimeError(f"rank {rank} has no complete exact-bin evaluation rows")
        target = _estimate(selected_ev, "ev_local", replicates=replicates, seed=seed + rank, ci=ci)
        common = _estimate(selected_ev, "ev_conditional_global", replicates=replicates, seed=seed + 100 + rank, ci=ci)
        wrong = _estimate(selected_ev, "ev_wrong_mean", replicates=replicates, seed=seed + 200 + rank, ci=ci)
        target_minus_common = _difference(
            selected_ev, "ev_local", "ev_conditional_global",
            replicates=replicates, seed=seed + 300 + rank, ci=ci,
        )
        target_minus_wrong = _difference(
            selected_ev, "ev_local", "ev_wrong_mean",
            replicates=replicates, seed=seed + 400 + rank, ci=ci,
        )
        within = _estimate(selected_rotation, "R_within", replicates=replicates, seed=seed + 500 + rank, ci=ci)
        between = _estimate(selected_rotation, "R_between", replicates=replicates, seed=seed + 600 + rank, ci=ci)
        between_minus_within = _estimate(
            selected_rotation, "R_between_minus_within",
            replicates=replicates, seed=seed + 700 + rank, ci=ci,
        )
        reports.append({
            "rank": rank,
            "n_exact_problems": target["n_problems"],
            "n_distance_problems": within["n_problems"],
            "heldout_ev": {
                "target_context": target,
                "matched_common": common,
                "wrong_context": wrong,
                "target_minus_matched_common": target_minus_common,
                "target_minus_wrong_context": target_minus_wrong,
            },
            "subspace_distance": {
                "within_context": within,
                "between_context": between,
                "between_minus_within": between_minus_within,
            },
            "signs": {
                "target_above_matched_common": target_minus_common["mean"] > 0,
                "target_above_wrong_context": target_minus_wrong["mean"] > 0,
                "between_above_within": between_minus_within["mean"] > 0,
            },
            "ci_excludes_zero": {
                "target_above_matched_common": target_minus_common["ci_low"] > 0,
                "target_above_wrong_context": target_minus_wrong["ci_low"] > 0,
                "between_above_within": between_minus_within["ci_low"] > 0,
            },
        })

    fixed_confirmatory_rank = config.get("replication_confirmatory_fixed_rank")
    primary_rank = int(
        fixed_confirmatory_rank
        if fixed_confirmatory_rank is not None
        else geometry.get("selected_rank", max(map(int, config["analysis"]["ranks"])))
    )
    rank_annotation = (
        f"fixed rank {primary_rank}"
        if fixed_confirmatory_rank is not None
        else f"development-selected rank {primary_rank}"
    )
    result = {
        "profile": config.get("profile", "external_geometry"),
        "dataset_label": _dataset_label(config),
        "dataset": config["data"]["trajectories_jsonl"],
        "model": config["model"]["checkpoint"],
        "layer": selected_layer,
        "confidence_level": ci,
        "candidate_protocol": {
            "analysis_candidates": analysis_candidates,
            "folds": candidate_folds,
            "fit_per_fold": analysis_candidates - heldout_candidates,
            "heldout_per_fold": heldout_candidates,
        },
        "matched_common_definition": (
            "Pooled response covariance matched by prefix-length and reasoning-progress bins; "
            "only exact-bin controls are included."
        ),
        "distance_definition": geometry["rotation_distance_definition"],
        "preregistered_rank": primary_rank,
        "primary_rank": primary_rank,
        "primary_rank_origin": "fixed_confirmatory" if fixed_confirmatory_rank is not None else "development_selected",
        "rank_annotation": rank_annotation,
        "design_audit": _design_audit(config),
        "ranks": reports,
        "all_ranks_same_positive_sign": {
            key: all(report["signs"][key] for report in reports)
            for key in reports[0]["signs"]
        },
        "all_ranks_ci_excludes_zero": {
            key: all(report["ci_excludes_zero"][key] for report in reports)
            for key in reports[0]["ci_excludes_zero"]
        },
    }
    output_path = root / "metrics/external_geometry_controls.json"
    curve_path = root / "metrics/external_geometry_rank_curve.csv"
    figure_path = root / "figures/external_geometry_rank_curve.svg"
    _write_rank_curve(curve_path, reports)
    _write_rank_curve_svg(figure_path, reports, result)
    result["rank_curve_csv"] = str(curve_path)
    result["rank_curve_figure_svg"] = str(figure_path)
    reference_path = config.get("design_reference_config")
    if reference_path:
        reference_config = load_config(reference_path)
        reference_root = ensure_layout(reference_config)
        reference_geometry = read_json(reference_root / "metrics/paper_geometry_summary.json")
        comparison = [
            {
                "dataset": _dataset_label(reference_config),
                "model": reference_config["model"]["checkpoint"],
                "layer": int(reference_geometry["selected_layer"]),
                "reports": _summarize_reference_ev_curve(reference_config, reference_root),
            },
            {
                "dataset": result["dataset_label"],
                "model": result["model"],
                "layer": result["layer"],
                "reports": reports,
            },
        ]
        reference_ranks = [int(row["rank"]) for row in comparison[0]["reports"]]
        external_ranks = [int(row["rank"]) for row in reports]
        if reference_ranks != external_ranks:
            raise ValueError(
                f"combined rank figure requires identical rank grids: "
                f"reference={reference_ranks}, external={external_ranks}"
            )
        comparison_csv = root / "metrics/gsm8k_commonsenseqa_rank_comparison.csv"
        comparison_agreement = root / "metrics/gsm8k_commonsenseqa_curve_agreement.json"
        comparison_figure = root / "figures/gsm8k_commonsenseqa_ev_rank_curves.svg"
        comparison_distance_figure = root / "figures/gsm8k_commonsenseqa_distance_rank_curves.svg"
        comparison_gap_figure = root / "figures/gsm8k_commonsenseqa_rank_gap_comparison.svg"
        _write_cross_dataset_curve_csv(comparison_csv, comparison)
        atomic_json(comparison_agreement, _curve_agreement_summary(comparison))
        _write_cross_dataset_ev_curve_svg(
            comparison_figure,
            comparison,
            fixed_rank=result["preregistered_rank"],
            confidence_level=ci,
            candidate_protocol=result["candidate_protocol"],
        )
        _write_cross_dataset_distance_curve_svg(
            comparison_distance_figure,
            comparison,
            fixed_rank=result["preregistered_rank"],
            confidence_level=ci,
            candidate_protocol=result["candidate_protocol"],
        )
        _write_cross_dataset_curve_svg(
            comparison_gap_figure,
            comparison,
            fixed_rank=result["preregistered_rank"],
            confidence_level=ci,
            candidate_protocol=result["candidate_protocol"],
        )
        result["cross_dataset_rank_curve_csv"] = str(comparison_csv)
        result["cross_dataset_curve_agreement_json"] = str(comparison_agreement)
        result["cross_dataset_rank_curve_figure_svg"] = str(comparison_figure)
        result["cross_dataset_distance_rank_curve_figure_svg"] = str(comparison_distance_figure)
        result["cross_dataset_gap_rank_curve_figure_svg"] = str(comparison_gap_figure)
    atomic_json(output_path, result)
    return output_path, result


def _format_interval(estimate: dict) -> str:
    return f"{estimate['mean']:.4f} [{estimate['ci_low']:.4f}, {estimate['ci_high']:.4f}]"


def _write_rank_curve(path: Path, reports: list[dict]) -> None:
    rows = []
    for report in reports:
        ev = report["heldout_ev"]
        distance = report["subspace_distance"]
        row = {"rank": report["rank"], "n_exact_problems": report["n_exact_problems"]}
        for prefix, estimate in (
            ("target_ev", ev["target_context"]),
            ("matched_common_ev", ev["matched_common"]),
            ("wrong_context_ev", ev["wrong_context"]),
            ("target_minus_common", ev["target_minus_matched_common"]),
            ("target_minus_wrong", ev["target_minus_wrong_context"]),
            ("within_distance", distance["within_context"]),
            ("between_distance", distance["between_context"]),
            ("between_minus_within", distance["between_minus_within"]),
        ):
            row[f"{prefix}_mean"] = estimate["mean"]
            row[f"{prefix}_ci_low"] = estimate["ci_low"]
            row[f"{prefix}_ci_high"] = estimate["ci_high"]
        rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summarize_reference_ev_curve(config: dict, root: Path) -> list[dict]:
    geometry = read_json(root / "metrics/paper_geometry_summary.json")
    selected_layer = int(geometry["selected_layer"])
    expected_wrong = int(config["controls"]["wrong_prefixes_per_target"])
    replicates = int(config["statistics"]["bootstrap_replicates"])
    ci = float(config["statistics"]["ci"])
    seed = int(config["seed"])
    rows = _read_csv(root / "metrics/paper_geometry_rows.csv")
    rotation_path = root / "metrics/paper_rotation_rank_rows.csv"
    if not rotation_path.is_file():
        raise FileNotFoundError(
            f"{rotation_path} is missing; generate it from saved residuals with "
            "refresh_paper_geometry --multirank-rotation-only"
        )
    rotation_rows = _read_csv(rotation_path)
    wrong_exact_by_prefix = _wrong_exact_by_prefix(root)
    reports = []
    for rank in map(int, config["analysis"]["ranks"]):
        selected = [
            row for row in rows
            if row["split"] == "evaluation"
            and int(row["layer"]) == selected_layer
            and int(row["rank"]) == rank
            and _conditional_global_is_exact(row)
            and _wrong_control_is_exact(row, wrong_exact_by_prefix)
            and int(row["wrong_prefix_count"]) == expected_wrong
        ]
        if not selected:
            raise RuntimeError(f"reference dataset rank {rank} has no complete exact-bin evaluation rows")
        selected_rotation = [
            row for row in rotation_rows
            if int(row["layer"]) == selected_layer
            and int(row["rank"]) == rank
            and _wrong_control_is_exact(row, wrong_exact_by_prefix)
            and int(row["wrong_prefix_count"]) == expected_wrong
        ]
        if not selected_rotation:
            raise RuntimeError(f"reference dataset rank {rank} has no complete rotation rows")
        reports.append({
            "rank": rank,
            "n_exact_problems": len({row["problem_id"] for row in selected}),
            "n_distance_problems": len({row["problem_id"] for row in selected_rotation}),
            "heldout_ev": {
                "target_context": _estimate(selected, "ev_local", replicates=replicates, seed=seed + rank, ci=ci),
                "matched_common": _estimate(selected, "ev_conditional_global", replicates=replicates, seed=seed + 100 + rank, ci=ci),
                "wrong_context": _estimate(selected, "ev_wrong_mean", replicates=replicates, seed=seed + 200 + rank, ci=ci),
                "target_minus_matched_common": _difference(
                    selected, "ev_local", "ev_conditional_global",
                    replicates=replicates, seed=seed + 300 + rank, ci=ci,
                ),
                "target_minus_wrong_context": _difference(
                    selected, "ev_local", "ev_wrong_mean",
                    replicates=replicates, seed=seed + 400 + rank, ci=ci,
                ),
            },
            "subspace_distance": {
                "within_context": _estimate(
                    selected_rotation, "R_within",
                    replicates=replicates, seed=seed + 500 + rank, ci=ci,
                ),
                "between_context": _estimate(
                    selected_rotation, "R_between",
                    replicates=replicates, seed=seed + 600 + rank, ci=ci,
                ),
                "between_minus_within": _estimate(
                    selected_rotation, "R_between_minus_within",
                    replicates=replicates, seed=seed + 700 + rank, ci=ci,
                ),
            },
        })
    return reports


def _write_cross_dataset_curve_csv(path: Path, comparison: list[dict]) -> None:
    rows = []
    for dataset in comparison:
        for report in dataset["reports"]:
            ev = report["heldout_ev"]
            row = {
                "dataset": dataset["dataset"],
                "model": dataset["model"],
                "layer": dataset["layer"],
                "rank": report["rank"],
                "n_exact_problems": report["n_exact_problems"],
                "n_distance_problems": report.get("n_distance_problems", report["subspace_distance"]["within_context"]["n_problems"]),
            }
            for prefix, estimate in (
                ("target_ev", ev["target_context"]),
                ("matched_common_ev", ev["matched_common"]),
                ("wrong_context_ev", ev["wrong_context"]),
                ("target_minus_common", ev["target_minus_matched_common"]),
                ("target_minus_wrong", ev["target_minus_wrong_context"]),
            ):
                row[f"{prefix}_mean"] = estimate["mean"]
                row[f"{prefix}_ci_low"] = estimate["ci_low"]
                row[f"{prefix}_ci_high"] = estimate["ci_high"]
            for prefix, estimate in (
                ("within_distance", report["subspace_distance"]["within_context"]),
                ("between_distance", report["subspace_distance"]["between_context"]),
                ("between_minus_within", report["subspace_distance"]["between_minus_within"]),
            ):
                row[f"{prefix}_mean"] = estimate["mean"]
                row[f"{prefix}_ci_low"] = estimate["ci_low"]
                row[f"{prefix}_ci_high"] = estimate["ci_high"]
            rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _curve_agreement_summary(comparison: list[dict]) -> dict:
    if len(comparison) != 2:
        raise ValueError("curve agreement requires exactly two datasets")
    ranks = [int(report["rank"]) for report in comparison[0]["reports"]]
    metrics = {}
    for label, field in (
        ("target_context", "target_context"),
        ("matched_common", "matched_common"),
        ("wrong_context", "wrong_context"),
    ):
        left = np.asarray([float(row["heldout_ev"][field]["mean"]) for row in comparison[0]["reports"]])
        right = np.asarray([float(row["heldout_ev"][field]["mean"]) for row in comparison[1]["reports"]])
        left_normalized = left / left[-1]
        right_normalized = right / right[-1]
        metrics[label] = {
            "pearson_r_across_rank_points": float(np.corrcoef(left, right)[0, 1]),
            "raw_rmse": float(np.sqrt(np.mean(np.square(left - right)))),
            "raw_max_absolute_difference": float(np.max(np.abs(left - right))),
            "rank64_normalized_rmse": float(np.sqrt(np.mean(np.square(left_normalized - right_normalized)))),
            "rank64_normalized_max_absolute_difference": float(np.max(np.abs(left_normalized - right_normalized))),
            "monotone_nondecreasing_in_both_datasets": bool(np.all(np.diff(left) >= 0) and np.all(np.diff(right) >= 0)),
        }
    for label, field in (
        ("within_context_distance", "within_context"),
        ("between_context_distance", "between_context"),
        ("between_minus_within_distance", "between_minus_within"),
    ):
        left = np.asarray([float(row["subspace_distance"][field]["mean"]) for row in comparison[0]["reports"]])
        right = np.asarray([float(row["subspace_distance"][field]["mean"]) for row in comparison[1]["reports"]])
        metrics[label] = {
            "pearson_r_across_rank_points": float(np.corrcoef(left, right)[0, 1]),
            "raw_rmse": float(np.sqrt(np.mean(np.square(left - right)))),
            "raw_max_absolute_difference": float(np.max(np.abs(left - right))),
            "same_direction_from_rank1_to_rank64": bool(np.sign(left[-1] - left[0]) == np.sign(right[-1] - right[0])),
        }
    ordering = {}
    for dataset in comparison:
        ordering[dataset["dataset"]] = {
            "target_above_matched_common_at_all_ranks": all(
                float(row["heldout_ev"]["target_context"]["mean"])
                > float(row["heldout_ev"]["matched_common"]["mean"])
                for row in dataset["reports"]
            ),
            "target_above_wrong_context_at_all_ranks": all(
                float(row["heldout_ev"]["target_context"]["mean"])
                > float(row["heldout_ev"]["wrong_context"]["mean"])
                for row in dataset["reports"]
            ),
        }
    return {
        "datasets": [dataset["dataset"] for dataset in comparison],
        "ranks": ranks,
        "metrics": metrics,
        "ordering": ordering,
        "interpretation": (
            "Descriptive curve-shape agreement only. Pearson correlation is expected to be high for "
            "monotone rank curves; normalized deviations and ordering should be reported alongside it."
        ),
        "equivalence_test": "not performed; no pre-specified equivalence margin",
    }


def _nice_axis(maximum: float, *, minimum: float = 0.0, ticks: int = 4) -> tuple[float, float, list[float]]:
    span = max(maximum - minimum, 1e-6)
    raw_step = span / ticks
    magnitude = 10 ** math.floor(math.log10(raw_step))
    normalized = raw_step / magnitude
    step_multiplier = 1 if normalized <= 1 else 2 if normalized <= 2 else 5 if normalized <= 5 else 10
    step = step_multiplier * magnitude
    lower = math.floor(minimum / step) * step
    upper = math.ceil(maximum / step) * step
    values = []
    value = lower
    while value <= upper + step * 0.01:
        values.append(value)
        value += step
    return lower, upper, values


def _svg_marker(kind: str, x: float, y: float, color: str) -> str:
    if kind == "square":
        return f'<rect x="{x - 4:.2f}" y="{y - 4:.2f}" width="8" height="8" fill="white" stroke="{color}" stroke-width="1.8"/>'
    if kind == "diamond":
        points = f"{x:.2f},{y - 5:.2f} {x + 5:.2f},{y:.2f} {x:.2f},{y + 5:.2f} {x - 5:.2f},{y:.2f}"
        return f'<polygon points="{points}" fill="white" stroke="{color}" stroke-width="1.8"/>'
    if kind == "triangle":
        points = f"{x:.2f},{y - 5:.2f} {x + 5:.2f},{y + 4.5:.2f} {x - 5:.2f},{y + 4.5:.2f}"
        return f'<polygon points="{points}" fill="white" stroke="{color}" stroke-width="1.8"/>'
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="white" stroke="{color}" stroke-width="1.8"/>'


def _write_rank_curve_svg(path: Path, reports: list[dict], result: dict) -> None:
    """Write the external-dataset rank curve without adding a plotting dependency."""
    if not reports:
        raise ValueError("rank-curve figure requires at least one rank")
    ranks = [int(report["rank"]) for report in reports]
    if any(rank <= 0 for rank in ranks):
        raise ValueError("rank-curve figure requires positive ranks")

    panels = [
        {
            "title": "Held-out explained variance",
            "ylabel": "Held-out EV",
            "series": [
                ("Target context", "heldout_ev", "target_context", _BLUE, "", "circle"),
                ("Matched common", "heldout_ev", "matched_common", _GREEN, "7 4", "square"),
                ("Wrong context", "heldout_ev", "wrong_context", _ORANGE, "2 3", "diamond"),
            ],
        },
        {
            "title": "Target-versus-control specificity",
            "ylabel": "Target minus control EV",
            "series": [
                ("Matched common", "heldout_ev", "target_minus_matched_common", _BLUE, "", "circle"),
                ("Wrong context", "heldout_ev", "target_minus_wrong_context", _ORANGE, "7 4", "square"),
            ],
        },
        {
            "title": "Subspace separation",
            "ylabel": "Between minus within distance",
            "series": [
                ("Between minus within", "subspace_distance", "between_minus_within", _GREEN, "", "triangle"),
            ],
        },
    ]

    width, height = 1180, 450
    left, right, gap = 82, 30, 72
    top, bottom = 100, 88
    panel_width = (width - left - right - 2 * gap) / 3
    plot_height = height - top - bottom
    minimum_log = math.log2(min(ranks))
    maximum_log = math.log2(max(ranks))
    log_span = max(maximum_log - minimum_log, 1.0)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{escape(result["dataset_label"])} geometry controls across evaluation ranks</title>',
        '<desc id="desc">Held-out explained variance, target-versus-control differences, and between-minus-within subspace distance with problem-bootstrap 95 percent confidence intervals.</desc>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="DejaVu Sans, Arial, sans-serif" fill="#27313B">',
        f'<text x="590" y="28" text-anchor="middle" font-size="20" font-weight="700">{escape(result["dataset_label"])} geometry controls across rank</text>',
        f'<text x="590" y="53" text-anchor="middle" font-size="12" fill="{_MID}">{escape(result["model"].split("/")[-1])}, block {int(result["layer"])}; means and problem-bootstrap {100 * float(result["confidence_level"]):g}% CIs</text>',
    ]

    for panel_index, panel in enumerate(panels):
        x0 = left + panel_index * (panel_width + gap)
        x1 = x0 + panel_width
        y0 = top
        y1 = top + plot_height
        estimates = [
            report[group][field]
            for report in reports
            for _, group, field, _, _, _ in panel["series"]
        ]
        observed_minimum = min(float(estimate["ci_low"]) for estimate in estimates)
        observed_maximum = max(float(estimate["ci_high"]) for estimate in estimates)
        axis_minimum, axis_maximum, y_ticks = _nice_axis(
            observed_maximum * 1.06,
            minimum=min(0.0, observed_minimum * 1.06),
        )
        y_span = max(axis_maximum - axis_minimum, 1e-6)

        def x_position(rank: int) -> float:
            return x0 + (math.log2(rank) - minimum_log) / log_span * panel_width

        def y_position(value: float) -> float:
            return y1 - (value - axis_minimum) / y_span * plot_height

        svg.append(f'<text x="{x0:.2f}" y="76" font-size="14" font-weight="700">{chr(65 + panel_index)}  {escape(panel["title"])}</text>')
        for tick in y_ticks:
            y = y_position(tick)
            svg.append(f'<line x1="{x0:.2f}" y1="{y:.2f}" x2="{x1:.2f}" y2="{y:.2f}" stroke="{_GRID}" stroke-width="1"/>')
            svg.append(f'<text x="{x0 - 9:.2f}" y="{y + 4:.2f}" text-anchor="end" font-size="11">{tick:.2f}</text>')
        svg.append(f'<line x1="{x0:.2f}" y1="{y1:.2f}" x2="{x1:.2f}" y2="{y1:.2f}" stroke="{_INK}" stroke-width="1"/>')
        svg.append(f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x0:.2f}" y2="{y1:.2f}" stroke="{_INK}" stroke-width="1"/>')
        if axis_minimum < 0 < axis_maximum:
            zero_y = y_position(0.0)
            svg.append(f'<line x1="{x0:.2f}" y1="{zero_y:.2f}" x2="{x1:.2f}" y2="{zero_y:.2f}" stroke="{_INK}" stroke-width="1"/>')

        fixed_rank = int(result["primary_rank"])
        if fixed_rank in ranks:
            fixed_x = x_position(fixed_rank)
            svg.append(f'<line x1="{fixed_x:.2f}" y1="{y0:.2f}" x2="{fixed_x:.2f}" y2="{y1:.2f}" stroke="{_MID}" stroke-width="1" stroke-dasharray="2 3"/>')
            if panel_index == 1:
                svg.append(f'<text x="{fixed_x - 5:.2f}" y="{y0 + 13:.2f}" text-anchor="end" font-size="10" fill="{_MID}">{escape(result["rank_annotation"])}</text>')

        for rank in ranks:
            x = x_position(rank)
            svg.append(f'<line x1="{x:.2f}" y1="{y1:.2f}" x2="{x:.2f}" y2="{y1 + 4:.2f}" stroke="{_INK}" stroke-width="1"/>')
            svg.append(f'<text x="{x:.2f}" y="{y1 + 20:.2f}" text-anchor="middle" font-size="11">{rank}</text>')
        svg.append(f'<text x="{(x0 + x1) / 2:.2f}" y="{y1 + 45:.2f}" text-anchor="middle" font-size="12">Rank</text>')
        ylabel_x = x0 - 57
        ylabel_y = (y0 + y1) / 2
        svg.append(f'<text x="{ylabel_x:.2f}" y="{ylabel_y:.2f}" text-anchor="middle" font-size="12" transform="rotate(-90 {ylabel_x:.2f} {ylabel_y:.2f})">{escape(panel["ylabel"])}</text>')

        legend_x = x1 - 178 if panel_index == 2 else x0 + 10
        legend_y = y0 + 20
        for series_index, (label, group, field, color, dash, marker) in enumerate(panel["series"]):
            y_legend = legend_y + 20 * series_index
            dash_attribute = f' stroke-dasharray="{dash}"' if dash else ""
            svg.append(f'<line x1="{legend_x:.2f}" y1="{y_legend:.2f}" x2="{legend_x + 20:.2f}" y2="{y_legend:.2f}" stroke="{color}" stroke-width="1.8"{dash_attribute}/>')
            svg.append(_svg_marker(marker, legend_x + 10, y_legend, color))
            svg.append(f'<text x="{legend_x + 30:.2f}" y="{y_legend + 4:.2f}" font-size="11">{escape(label)}</text>')

            points = []
            for report in reports:
                estimate = report[group][field]
                x = x_position(int(report["rank"]))
                y = y_position(float(estimate["mean"]))
                low_y = y_position(float(estimate["ci_low"]))
                high_y = y_position(float(estimate["ci_high"]))
                points.append(f"{x:.2f},{y:.2f}")
                svg.append(f'<line x1="{x:.2f}" y1="{high_y:.2f}" x2="{x:.2f}" y2="{low_y:.2f}" stroke="{color}" stroke-width="1"/>')
                svg.append(f'<line x1="{x - 3:.2f}" y1="{high_y:.2f}" x2="{x + 3:.2f}" y2="{high_y:.2f}" stroke="{color}" stroke-width="1"/>')
                svg.append(f'<line x1="{x - 3:.2f}" y1="{low_y:.2f}" x2="{x + 3:.2f}" y2="{low_y:.2f}" stroke="{color}" stroke-width="1"/>')
            svg.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="1.8"{dash_attribute}/>')
            for report in reports:
                estimate = report[group][field]
                svg.append(_svg_marker(marker, x_position(int(report["rank"])), y_position(float(estimate["mean"])), color))

    exact_counts = sorted({int(report["n_exact_problems"]) for report in reports})
    count_text = str(exact_counts[0]) if len(exact_counts) == 1 else f'{min(exact_counts)}–{max(exact_counts)}'
    protocol = result["candidate_protocol"]
    svg.append(f'<text x="590" y="437" text-anchor="middle" font-size="10" fill="{_MID}">Exact-bin evaluation; n={count_text} problems per rank; candidate-level {protocol["fit_per_fold"]}-fit/{protocol["heldout_per_fold"]}-held-out {protocol["folds"]}-fold protocol.</text>')
    svg.extend(["</g>", "</svg>"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def _write_cross_dataset_curve_svg(
    path: Path,
    comparison: list[dict],
    *,
    fixed_rank: int,
    confidence_level: float,
    candidate_protocol: dict,
) -> None:
    if len(comparison) != 2:
        raise ValueError("cross-dataset rank figure requires exactly two datasets")
    ranks = [int(row["rank"]) for row in comparison[0]["reports"]]
    estimates = [
        report["heldout_ev"][field]
        for dataset in comparison
        for report in dataset["reports"]
        for field in ("target_minus_matched_common", "target_minus_wrong_context")
    ]
    observed_minimum = min(float(estimate["ci_low"]) for estimate in estimates)
    observed_maximum = max(float(estimate["ci_high"]) for estimate in estimates)
    axis_minimum, axis_maximum, y_ticks = _nice_axis(
        observed_maximum * 1.06,
        minimum=min(0.0, observed_minimum * 1.06),
    )

    width, height = 920, 430
    left, right, gap = 82, 30, 90
    top, bottom = 108, 88
    panel_width = (width - left - right - gap) / 2
    plot_height = height - top - bottom
    minimum_log = math.log2(min(ranks))
    maximum_log = math.log2(max(ranks))
    log_span = max(maximum_log - minimum_log, 1.0)
    y_span = max(axis_maximum - axis_minimum, 1e-6)
    protocol = candidate_protocol

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Target-context advantage across GSM8K and CommonsenseQA ranks</title>',
        '<desc id="desc">Side-by-side rank curves for target-context explained-variance advantage over matched-common and wrong-context controls, with shared axes and problem-bootstrap confidence intervals.</desc>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="DejaVu Sans, Arial, sans-serif" fill="#27313B">',
        '<text x="460" y="28" text-anchor="middle" font-size="20" font-weight="700">Target-context advantage across datasets</text>',
        f'<text x="460" y="53" text-anchor="middle" font-size="12" fill="{_MID}">Shared axes; means and problem-bootstrap {100 * confidence_level:g}% CIs; exact-bin controls</text>',
    ]

    for panel_index, dataset in enumerate(comparison):
        x0 = left + panel_index * (panel_width + gap)
        x1 = x0 + panel_width
        y0 = top
        y1 = top + plot_height

        def x_position(rank: int) -> float:
            return x0 + (math.log2(rank) - minimum_log) / log_span * panel_width

        def y_position(value: float) -> float:
            return y1 - (value - axis_minimum) / y_span * plot_height

        svg.append(f'<text x="{x0:.2f}" y="76" font-size="15" font-weight="700">{chr(65 + panel_index)}  {escape(dataset["dataset"])}</text>')
        model_label = dataset["model"].split("/")[-1]
        svg.append(f'<text x="{x0:.2f}" y="96" font-size="11" fill="{_MID}">{escape(model_label)}, block {int(dataset["layer"])}</text>')
        for tick in y_ticks:
            y = y_position(tick)
            svg.append(f'<line x1="{x0:.2f}" y1="{y:.2f}" x2="{x1:.2f}" y2="{y:.2f}" stroke="{_GRID}" stroke-width="1"/>')
            svg.append(f'<text x="{x0 - 9:.2f}" y="{y + 4:.2f}" text-anchor="end" font-size="11">{tick:.2f}</text>')
        svg.append(f'<line x1="{x0:.2f}" y1="{y1:.2f}" x2="{x1:.2f}" y2="{y1:.2f}" stroke="{_INK}" stroke-width="1"/>')
        svg.append(f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x0:.2f}" y2="{y1:.2f}" stroke="{_INK}" stroke-width="1"/>')
        if axis_minimum < 0 < axis_maximum:
            zero_y = y_position(0.0)
            svg.append(f'<line x1="{x0:.2f}" y1="{zero_y:.2f}" x2="{x1:.2f}" y2="{zero_y:.2f}" stroke="{_INK}" stroke-width="1"/>')
        if fixed_rank in ranks:
            fixed_x = x_position(fixed_rank)
            svg.append(f'<line x1="{fixed_x:.2f}" y1="{y0:.2f}" x2="{fixed_x:.2f}" y2="{y1:.2f}" stroke="{_MID}" stroke-width="1" stroke-dasharray="2 3"/>')
            if panel_index == 1:
                svg.append(f'<text x="{fixed_x - 5:.2f}" y="{y0 + 13:.2f}" text-anchor="end" font-size="10" fill="{_MID}">rank {fixed_rank}</text>')

        for rank in ranks:
            x = x_position(rank)
            svg.append(f'<line x1="{x:.2f}" y1="{y1:.2f}" x2="{x:.2f}" y2="{y1 + 4:.2f}" stroke="{_INK}" stroke-width="1"/>')
            svg.append(f'<text x="{x:.2f}" y="{y1 + 20:.2f}" text-anchor="middle" font-size="11">{rank}</text>')
        svg.append(f'<text x="{(x0 + x1) / 2:.2f}" y="{y1 + 45:.2f}" text-anchor="middle" font-size="12">Rank</text>')
        ylabel_x = x0 - 57
        ylabel_y = (y0 + y1) / 2
        svg.append(f'<text x="{ylabel_x:.2f}" y="{ylabel_y:.2f}" text-anchor="middle" font-size="12" transform="rotate(-90 {ylabel_x:.2f} {ylabel_y:.2f})">Target minus control EV</text>')

        legend_x = x0 + 8
        legend_y = y0 + 17
        for series_index, (label, field, color, dash, marker) in enumerate((
            ("Matched common", "target_minus_matched_common", _BLUE, "", "circle"),
            ("Wrong context", "target_minus_wrong_context", _ORANGE, "7 4", "square"),
        )):
            y_legend = legend_y + 20 * series_index
            dash_attribute = f' stroke-dasharray="{dash}"' if dash else ""
            svg.append(f'<line x1="{legend_x:.2f}" y1="{y_legend:.2f}" x2="{legend_x + 20:.2f}" y2="{y_legend:.2f}" stroke="{color}" stroke-width="1.8"{dash_attribute}/>')
            svg.append(_svg_marker(marker, legend_x + 10, y_legend, color))
            svg.append(f'<text x="{legend_x + 30:.2f}" y="{y_legend + 4:.2f}" font-size="11">{escape(label)}</text>')
            points = []
            for report in dataset["reports"]:
                estimate = report["heldout_ev"][field]
                x = x_position(int(report["rank"]))
                y = y_position(float(estimate["mean"]))
                low_y = y_position(float(estimate["ci_low"]))
                high_y = y_position(float(estimate["ci_high"]))
                points.append(f"{x:.2f},{y:.2f}")
                svg.append(f'<line x1="{x:.2f}" y1="{high_y:.2f}" x2="{x:.2f}" y2="{low_y:.2f}" stroke="{color}" stroke-width="1"/>')
                svg.append(f'<line x1="{x - 3:.2f}" y1="{high_y:.2f}" x2="{x + 3:.2f}" y2="{high_y:.2f}" stroke="{color}" stroke-width="1"/>')
                svg.append(f'<line x1="{x - 3:.2f}" y1="{low_y:.2f}" x2="{x + 3:.2f}" y2="{low_y:.2f}" stroke="{color}" stroke-width="1"/>')
            svg.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="1.8"{dash_attribute}/>')
            for report in dataset["reports"]:
                estimate = report["heldout_ev"][field]
                svg.append(_svg_marker(marker, x_position(int(report["rank"])), y_position(float(estimate["mean"])), color))

    sample_labels = []
    for dataset in comparison:
        counts = sorted({int(report["n_exact_problems"]) for report in dataset["reports"]})
        count_text = str(counts[0]) if len(counts) == 1 else f"{min(counts)}–{max(counts)}"
        sample_labels.append(f'{dataset["dataset"]} n={count_text}')
    svg.append(
        f'<text x="460" y="417" text-anchor="middle" font-size="10" fill="{_MID}">'
        f'Candidate-level {protocol["fit_per_fold"]}-fit/{protocol["heldout_per_fold"]}-held-out {protocol["folds"]}-fold protocol; {escape("; ".join(sample_labels))}.'
        '</text>'
    )
    svg.extend(["</g>", "</svg>"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def _write_cross_dataset_three_metric_curve_svg(
    path: Path,
    comparison: list[dict],
    *,
    fixed_rank: int,
    confidence_level: float,
    candidate_protocol: dict,
    group: str,
    panels: list[tuple[str, str]],
    svg_title: str,
    description: str,
    figure_title: str,
    ylabel: str,
    sample_count_field: str,
) -> None:
    """Compare three aligned rank curves across two datasets."""
    if len(comparison) != 2:
        raise ValueError("cross-dataset rank figure requires exactly two datasets")
    ranks = [int(row["rank"]) for row in comparison[0]["reports"]]
    estimates = [
        report[group][field]
        for dataset in comparison
        for report in dataset["reports"]
        for _, field in panels
    ]
    observed_maximum = max(float(estimate["ci_high"]) for estimate in estimates)
    axis_minimum, axis_maximum, y_ticks = _nice_axis(observed_maximum * 1.06, minimum=0.0)

    width, height = 1180, 450
    left, right, gap = 82, 30, 72
    top, bottom = 102, 90
    panel_width = (width - left - right - 2 * gap) / 3
    plot_height = height - top - bottom
    minimum_log = math.log2(min(ranks))
    maximum_log = math.log2(max(ranks))
    log_span = max(maximum_log - minimum_log, 1.0)
    y_span = max(axis_maximum - axis_minimum, 1e-6)
    protocol = candidate_protocol
    dataset_styles = [
        (_BLUE, "", "circle"),
        (_ORANGE, "7 4", "square"),
    ]

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{escape(svg_title)}</title>',
        f'<desc id="desc">{escape(description)}</desc>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="DejaVu Sans, Arial, sans-serif" fill="#27313B">',
        f'<text x="590" y="28" text-anchor="middle" font-size="20" font-weight="700">{escape(figure_title)}</text>',
        f'<text x="590" y="53" text-anchor="middle" font-size="12" fill="{_MID}">Shared axes; means and problem-bootstrap {100 * confidence_level:g}% CIs; exact-bin controls</text>',
    ]

    for panel_index, (panel_title, field) in enumerate(panels):
        x0 = left + panel_index * (panel_width + gap)
        x1 = x0 + panel_width
        y0 = top
        y1 = top + plot_height

        def x_position(rank: int) -> float:
            return x0 + (math.log2(rank) - minimum_log) / log_span * panel_width

        def y_position(value: float) -> float:
            return y1 - (value - axis_minimum) / y_span * plot_height

        svg.append(f'<text x="{x0:.2f}" y="78" font-size="14" font-weight="700">{chr(65 + panel_index)}  {escape(panel_title)}</text>')
        for tick in y_ticks:
            y = y_position(tick)
            svg.append(f'<line x1="{x0:.2f}" y1="{y:.2f}" x2="{x1:.2f}" y2="{y:.2f}" stroke="{_GRID}" stroke-width="1"/>')
            svg.append(f'<text x="{x0 - 9:.2f}" y="{y + 4:.2f}" text-anchor="end" font-size="11">{tick:.2f}</text>')
        svg.append(f'<line x1="{x0:.2f}" y1="{y1:.2f}" x2="{x1:.2f}" y2="{y1:.2f}" stroke="{_INK}" stroke-width="1"/>')
        svg.append(f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x0:.2f}" y2="{y1:.2f}" stroke="{_INK}" stroke-width="1"/>')
        if fixed_rank in ranks:
            fixed_x = x_position(fixed_rank)
            svg.append(f'<line x1="{fixed_x:.2f}" y1="{y0:.2f}" x2="{fixed_x:.2f}" y2="{y1:.2f}" stroke="{_MID}" stroke-width="1" stroke-dasharray="2 3"/>')
            if panel_index == 2:
                svg.append(f'<text x="{fixed_x - 5:.2f}" y="{y0 + 13:.2f}" text-anchor="end" font-size="10" fill="{_MID}">rank {fixed_rank}</text>')
        for rank in ranks:
            x = x_position(rank)
            svg.append(f'<line x1="{x:.2f}" y1="{y1:.2f}" x2="{x:.2f}" y2="{y1 + 4:.2f}" stroke="{_INK}" stroke-width="1"/>')
            svg.append(f'<text x="{x:.2f}" y="{y1 + 20:.2f}" text-anchor="middle" font-size="11">{rank}</text>')
        svg.append(f'<text x="{(x0 + x1) / 2:.2f}" y="{y1 + 45:.2f}" text-anchor="middle" font-size="12">Rank</text>')
        ylabel_x = x0 - 57
        ylabel_y = (y0 + y1) / 2
        svg.append(f'<text x="{ylabel_x:.2f}" y="{ylabel_y:.2f}" text-anchor="middle" font-size="12" transform="rotate(-90 {ylabel_x:.2f} {ylabel_y:.2f})">{escape(ylabel)}</text>')

        legend_x = x0 + 8
        legend_y = y0 + 17
        for dataset_index, dataset in enumerate(comparison):
            color, dash, marker = dataset_styles[dataset_index]
            y_legend = legend_y + 20 * dataset_index
            dash_attribute = f' stroke-dasharray="{dash}"' if dash else ""
            svg.append(f'<line x1="{legend_x:.2f}" y1="{y_legend:.2f}" x2="{legend_x + 20:.2f}" y2="{y_legend:.2f}" stroke="{color}" stroke-width="1.8"{dash_attribute}/>')
            svg.append(_svg_marker(marker, legend_x + 10, y_legend, color))
            svg.append(f'<text x="{legend_x + 30:.2f}" y="{y_legend + 4:.2f}" font-size="11">{escape(dataset["dataset"])}</text>')
            points = []
            for report in dataset["reports"]:
                estimate = report[group][field]
                x = x_position(int(report["rank"]))
                y = y_position(float(estimate["mean"]))
                low_y = y_position(float(estimate["ci_low"]))
                high_y = y_position(float(estimate["ci_high"]))
                points.append(f"{x:.2f},{y:.2f}")
                svg.append(f'<line x1="{x:.2f}" y1="{high_y:.2f}" x2="{x:.2f}" y2="{low_y:.2f}" stroke="{color}" stroke-width="1"/>')
                svg.append(f'<line x1="{x - 3:.2f}" y1="{high_y:.2f}" x2="{x + 3:.2f}" y2="{high_y:.2f}" stroke="{color}" stroke-width="1"/>')
                svg.append(f'<line x1="{x - 3:.2f}" y1="{low_y:.2f}" x2="{x + 3:.2f}" y2="{low_y:.2f}" stroke="{color}" stroke-width="1"/>')
            svg.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="1.8"{dash_attribute}/>')
            for report in dataset["reports"]:
                estimate = report[group][field]
                svg.append(_svg_marker(marker, x_position(int(report["rank"])), y_position(float(estimate["mean"])), color))

    sample_labels = []
    for dataset in comparison:
        counts = sorted({int(report[sample_count_field]) for report in dataset["reports"]})
        count_text = str(counts[0]) if len(counts) == 1 else f"{min(counts)}–{max(counts)}"
        sample_labels.append(f'{dataset["dataset"]} n={count_text}')
    svg.append(
        f'<text x="590" y="437" text-anchor="middle" font-size="10" fill="{_MID}">'
        f'{protocol["fit_per_fold"]}-fit/{protocol["heldout_per_fold"]}-held-out {protocol["folds"]}-fold candidate protocol; '
        f'{escape("; ".join(sample_labels))}. GSM8K rank selected on development data; CommonsenseQA rank 64 fixed for external replication.'
        '</text>'
    )
    svg.extend(["</g>", "</svg>"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def _write_cross_dataset_ev_curve_svg(
    path: Path,
    comparison: list[dict],
    *,
    fixed_rank: int,
    confidence_level: float,
    candidate_protocol: dict,
) -> None:
    _write_cross_dataset_three_metric_curve_svg(
        path,
        comparison,
        fixed_rank=fixed_rank,
        confidence_level=confidence_level,
        candidate_protocol=candidate_protocol,
        group="heldout_ev",
        panels=[
            ("Target context", "target_context"),
            ("Matched common", "matched_common"),
            ("Wrong context", "wrong_context"),
        ],
        svg_title="Held-out explained-variance rank curves across GSM8K and CommonsenseQA",
        description=(
            "Target-context, matched-common, and wrong-context held-out explained variance "
            "across ranks one through sixty-four for two datasets, with shared axes and "
            "problem-bootstrap confidence intervals."
        ),
        figure_title="Held-out EV rank curves across datasets",
        ylabel="Held-out EV",
        sample_count_field="n_exact_problems",
    )


def _write_cross_dataset_distance_curve_svg(
    path: Path,
    comparison: list[dict],
    *,
    fixed_rank: int,
    confidence_level: float,
    candidate_protocol: dict,
) -> None:
    _write_cross_dataset_three_metric_curve_svg(
        path,
        comparison,
        fixed_rank=fixed_rank,
        confidence_level=confidence_level,
        candidate_protocol=candidate_protocol,
        group="subspace_distance",
        panels=[
            ("Within context", "within_context"),
            ("Between context", "between_context"),
            ("Between minus within", "between_minus_within"),
        ],
        svg_title="Subspace-distance rank curves across GSM8K and CommonsenseQA",
        description=(
            "Within-context, between-context, and between-minus-within projection-rotation "
            "distance across ranks one through sixty-four for two datasets, with shared axes "
            "and problem-bootstrap confidence intervals."
        ),
        figure_title="Subspace-distance rank curves across datasets",
        ylabel="Projection-rotation distance",
        sample_count_field="n_distance_problems",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    output_path, result = summarize(args.config)
    print(f"model={result['model']} layer={result['layer']}")
    primary = next(row for row in result["ranks"] if row["rank"] == result["primary_rank"])
    primary_ev = primary["heldout_ev"]
    primary_distance = primary["subspace_distance"]
    print(f"{result['rank_annotation']} exact problems={primary['n_exact_problems']}")
    print(f"  Target-context EV: {_format_interval(primary_ev['target_context'])}")
    print(f"  Matched-common EV: {_format_interval(primary_ev['matched_common'])}")
    print(f"  Wrong-context EV: {_format_interval(primary_ev['wrong_context'])}")
    print(f"  Target - Matched-common: {_format_interval(primary_ev['target_minus_matched_common'])}")
    print(f"  Target - Wrong-context: {_format_interval(primary_ev['target_minus_wrong_context'])}")
    print(f"  Between - Within: {_format_interval(primary_distance['between_minus_within'])}")
    audit = result.get("design_audit")
    if audit is not None:
        print(f"GSM8K design/statistics match={audit['all_fields_match']} reference={audit['reference_config']}")
        for name, field in audit["fields"].items():
            if not field["matches"]:
                print(f"  MISMATCH {name}: current={field['current']} reference={field['reference']}")
    print("rank curve")
    print("rank  target EV  common EV  wrong EV  target-common [95% CI]       target-wrong [95% CI]        between-within [95% CI]")
    for row in result["ranks"]:
        ev = row["heldout_ev"]
        distance = row["subspace_distance"]
        print(
            f"{row['rank']:>4d}  {ev['target_context']['mean']:.4f}     "
            f"{ev['matched_common']['mean']:.4f}     "
            f"{ev['wrong_context']['mean']:.4f}    "
            f"{_format_interval(ev['target_minus_matched_common']):<29s} "
            f"{_format_interval(ev['target_minus_wrong_context']):<29s} "
            f"{_format_interval(distance['between_minus_within'])}"
        )
    print("same positive sign across ranks=" + json.dumps(result["all_ranks_same_positive_sign"], sort_keys=True))
    print(result["rank_curve_csv"])
    print(result["rank_curve_figure_svg"])
    if "cross_dataset_rank_curve_csv" in result:
        print(result["cross_dataset_rank_curve_csv"])
        print(result["cross_dataset_curve_agreement_json"])
        print(result["cross_dataset_rank_curve_figure_svg"])
        print(result["cross_dataset_distance_rank_curve_figure_svg"])
        print(result["cross_dataset_gap_rank_curve_figure_svg"])
    print(output_path)


if __name__ == "__main__":
    main()
