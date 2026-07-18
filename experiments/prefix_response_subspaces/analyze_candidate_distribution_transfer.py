from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .src.residualization import double_center
from .src.review_experiments import (
    auxiliary_token_statistics,
    center_context_block,
    evaluate_transfer,
    projection_cell_energies,
    review_roots,
)
from .src.statistics import problem_bootstrap, two_way_ratio_bootstrap
from .src.subspaces import top_svd
from .src.utils import atomic_json, ensure_layout, file_sha256, load_config, read_json, stable_hash, stage_is_complete


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _split_target(token_ids: list[int], seed: int) -> tuple[list[int], list[int]]:
    values = np.asarray(sorted(set(map(int, token_ids))), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(values)
    midpoint = len(values) // 2
    return sorted(values[:midpoint].tolist()), sorted(values[midpoint:].tolist())


def _deterministic_subsample(token_ids: list[int], count: int, seed: int) -> list[int]:
    values = np.asarray(sorted(set(map(int, token_ids))), dtype=np.int64)
    if int(count) >= len(values):
        return values.tolist()
    rng = np.random.default_rng(int(seed))
    rng.shuffle(values)
    return sorted(values[: int(count)].tolist())


def _transfer_pairs(groups: dict[str, list[int]]) -> list[tuple[str, str, str]]:
    pairs = [
        ("high_to_low", "high_probability", "low_probability"),
        ("low_to_high", "low_probability", "high_probability"),
        ("independent_A_to_B", "independent_A", "independent_B_exclusive"),
        ("independent_B_to_A", "independent_B", "independent_A_exclusive"),
    ]
    categories = [name for name in ("number", "operator", "word", "whitespace", "other") if name in groups]
    pairs.extend((f"category_{source}_to_{target}", source, target) for source in categories for target in categories if source != target)
    return pairs


def _fit_source_bases(
    z: np.ndarray,
    evaluation: np.ndarray,
    positions: list[int],
    statistics: tuple[np.ndarray, np.ndarray],
    rank: int,
) -> list[np.ndarray]:
    bases = []
    for full_index in evaluation:
        residual = center_context_block(z, int(full_index), positions, *statistics)
        bases.append(top_svd(residual, rank, allow_rank_reduction=True).astype(np.float32))
    return bases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    source_root, root = review_roots(config)
    paths = {
        "states": root / "manifests/candidate_transfer_states.json",
        "sets": root / "candidate_tokens/review_candidate_sets.json",
        "geometry": source_root / "metrics/paper_geometry_summary.json",
    }
    inputs = {f"{key}_sha256": file_sha256(path) for key, path in paths.items()}
    manifest_path = root / "manifests/candidate_distribution_transfer.json"
    if not args.force and stage_is_complete(manifest_path, config, inputs):
        print(manifest_path)
        return
    review = config.get("review_experiments", {})
    states = read_json(paths["states"])
    sets = read_json(paths["sets"])
    geometry = read_json(paths["geometry"])
    contexts = read_json(states["contexts"])
    selected_layer = int(review.get("candidate_transfer_layer", geometry["selected_layer"]))
    layer_entry = next(row for row in states["layers"] if int(row["layer"]) == selected_layer)
    z = np.load(layer_entry["path"], mmap_mode="r")
    token_axis = {int(token_id): index for index, token_id in enumerate(states["candidate_token_ids"])}
    auxiliary = np.asarray([index for index, row in enumerate(contexts) if row["problem_group"] == "auxiliary"], dtype=np.int64)
    evaluation_groups = set(review.get("candidate_transfer_evaluation_groups", ["analysis_test"]))
    evaluation = np.asarray([index for index, row in enumerate(contexts) if row["problem_group"] in evaluation_groups], dtype=np.int64)
    problem_ids = np.asarray([contexts[index]["problem_id"] for index in evaluation])
    rank_requested = int(review.get("candidate_transfer_rank", geometry["selected_rank"]))
    minimum_rank = int(review.get("candidate_transfer_minimum_rank", 2))
    bootstrap_replicates = int(review.get("two_way_bootstrap_replicates", config["statistics"]["bootstrap_replicates"]))
    equalize_fit_counts = bool(review.get("candidate_transfer_equalize_fit_counts", True))
    ci = float(config["statistics"]["ci"])
    seed = int(config["seed"])
    rows: list[dict] = []
    summaries: dict[str, dict] = {}
    skipped: dict[str, str] = {}
    statistics_cache: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]] = {}
    target_reference_cache: dict[tuple[tuple[int, ...], tuple[int, ...], int], dict] = {}
    active_source_key: tuple[int, ...] | None = None
    active_source_bases: list[np.ndarray] | None = None
    active_source_rank = 0
    for pair_index, (name, source_name, target_name) in enumerate(_transfer_pairs(sets["groups"])):
        source_ids = sorted(set(map(int, sets["groups"].get(source_name, []))))
        target_ids = sorted(set(map(int, sets["groups"].get(target_name, []))))
        target_ids = sorted(set(target_ids) - set(source_ids))
        target_split_seed = seed + int(stable_hash({"target_distribution": target_name})[:8], 16)
        target_fit_ids, target_eval_ids = _split_target(target_ids, target_split_seed) if len(target_ids) >= 2 else ([], [])
        source_fit_ids = source_ids
        if equalize_fit_counts and target_fit_ids:
            common_fit_count = min(len(source_ids), len(target_fit_ids))
            source_seed = seed + int(stable_hash({"pair": name, "side": "source_fit"})[:8], 16)
            target_seed = seed + int(stable_hash({"pair": name, "side": "target_fit"})[:8], 16)
            source_fit_ids = _deterministic_subsample(source_ids, common_fit_count, source_seed)
            target_fit_ids = _deterministic_subsample(target_fit_ids, common_fit_count, target_seed)
        rank = min(rank_requested, len(source_fit_ids) - 1, len(target_fit_ids) - 1)
        if rank < minimum_rank or not target_eval_ids:
            skipped[name] = (
                f"insufficient tokens: source_available={len(source_ids)} source_fit={len(source_fit_ids)} "
                f"target_unique={len(target_ids)} target_fit={len(target_fit_ids)} rank={rank}"
            )
            continue
        try:
            source_positions = [token_axis[token_id] for token_id in source_fit_ids]
            target_fit_positions = [token_axis[token_id] for token_id in target_fit_ids]
            target_eval_positions = [token_axis[token_id] for token_id in target_eval_ids]
        except KeyError as exc:
            skipped[name] = f"token missing from extracted union: {exc}"
            continue
        def statistics(positions: list[int]) -> tuple[np.ndarray, np.ndarray]:
            key = tuple(map(int, positions))
            if key not in statistics_cache:
                statistics_cache[key] = auxiliary_token_statistics(z, auxiliary, positions)
            return statistics_cache[key]

        source_key = tuple(source_positions)
        maximum_source_rank = min(rank_requested, len(source_fit_ids) - 1)
        if active_source_key != source_key or active_source_bases is None or active_source_rank < maximum_source_rank:
            active_source_bases = _fit_source_bases(z, evaluation, source_positions, statistics(source_positions), maximum_source_rank)
            active_source_key = source_key
            active_source_rank = maximum_source_rank
        reference_key = (tuple(target_fit_ids), tuple(target_eval_ids), rank)
        if reference_key not in target_reference_cache:
            target_fit_stats = statistics(target_fit_positions)
            target_eval_stats = statistics(target_eval_positions)
            reference_num = np.empty((len(evaluation), len(target_eval_ids)), dtype=np.float64)
            reference_den = np.empty_like(reference_num)
            reference_ev = np.empty(len(evaluation), dtype=np.float64)
            for context_index, full_index in enumerate(evaluation):
                target_fit_r = center_context_block(z, int(full_index), target_fit_positions, *target_fit_stats)
                target_eval_r = center_context_block(z, int(full_index), target_eval_positions, *target_eval_stats)
                target_basis = top_svd(target_fit_r, rank, allow_rank_reduction=True)
                target_cell, target_den = projection_cell_energies(target_eval_r, target_basis)
                reference_num[context_index], reference_den[context_index] = target_cell, target_den
                reference_ev[context_index] = float(target_cell.sum() / max(target_den.sum(), 1e-12))
            target_reference_cache[reference_key] = {
                "numerator": reference_num,
                "denominator": reference_den,
                "ev": reference_ev,
                "evaluation_positions": target_eval_positions,
                "evaluation_statistics": target_eval_stats,
            }
        reference = target_reference_cache[reference_key]
        source_num = np.empty((len(evaluation), len(target_eval_ids)), dtype=np.float64)
        target_num = reference["numerator"]
        denominator = reference["denominator"]
        for context_index, full_index in enumerate(evaluation):
            target_eval_r = center_context_block(z, int(full_index), reference["evaluation_positions"], *reference["evaluation_statistics"])
            source_basis = active_source_bases[context_index][:, :rank]
            source_cell, den_cell = projection_cell_energies(target_eval_r, source_basis)
            source_num[context_index] = source_cell
            source_ev = float(source_cell.sum() / max(den_cell.sum(), 1e-12))
            target_ev = float(reference["ev"][context_index])
            rows.append({
                "problem_id": problem_ids[context_index], "prefix_id": contexts[int(evaluation[context_index])]["context_id"],
                "problem_group": contexts[int(evaluation[context_index])]["problem_group"],
                "pair": name, "source_distribution": source_name, "target_distribution": target_name,
                "layer": selected_layer, "rank": rank,
                "source_available_token_count": len(source_ids), "source_fit_token_count": len(source_fit_ids),
                "target_fit_token_count": len(target_fit_ids), "target_evaluation_token_count": len(target_eval_ids),
                "ev_transfer": source_ev, "ev_target_reference": target_ev,
                "transfer_minus_reference": source_ev - target_ev,
            })
        bootstrap_args = {
            "context_ids": problem_ids, "candidate_ids": np.asarray(target_eval_ids),
            "replicates": bootstrap_replicates, "ci": ci,
        }
        transfer = two_way_ratio_bootstrap(source_num, denominator, seed=seed + 1000 + pair_index, **bootstrap_args)
        reference = two_way_ratio_bootstrap(target_num, denominator, seed=seed + 2000 + pair_index, **bootstrap_args)
        pair_rows = [row for row in rows if row["pair"] == name]
        difference = problem_bootstrap(
            np.asarray([row["transfer_minus_reference"] for row in pair_rows]),
            np.asarray([row["problem_id"] for row in pair_rows]),
            replicates=int(config["statistics"]["bootstrap_replicates"]), seed=seed + 3000 + pair_index, ci=ci,
        )
        summaries[name] = {
            "source_distribution": source_name, "target_distribution": target_name,
            "rank": rank, "source_available_token_count": len(source_ids),
            "source_fit_token_count": len(source_fit_ids), "target_fit_token_count": len(target_fit_ids),
            "target_evaluation_token_count": len(target_eval_ids), "transfer_ev_two_way_bootstrap": transfer,
            "target_reference_ev_two_way_bootstrap": reference, "transfer_minus_reference_problem_bootstrap": difference,
            "transfer_fraction_of_target_reference": float(transfer["mean"] / reference["mean"]) if reference["mean"] > 0 else float("nan"),
        }
    rows_path = root / "metrics/candidate_distribution_transfer_rows.csv"
    _write_csv(rows_path, rows)
    summary = {
        "selected_layer": selected_layer, "requested_rank": rank_requested,
        "evaluation_groups": sorted(evaluation_groups),
        "candidate_distribution_dependency": "U_i is estimated and evaluated on explicitly named candidate distributions",
        "fit_count_policy": "source and target-reference fitted bases use equal deterministic candidate counts" if equalize_fit_counts else "source uses all available candidates; target reference uses its target-fit split",
        "performance_implementation": "auxiliary means and target-reference fits cached; one float64 SVD fit per source/context with float32 cached bases",
        "two_way_bootstrap_definition": "problem/context IDs and held-out candidate token IDs are independently resampled; source fitted bases remain fixed",
        "independent_candidate_diagnostics": sets["diagnostics"], "pairs": summaries, "skipped_pairs": skipped,
    }
    summary_path = root / "metrics/candidate_distribution_transfer_summary.json"
    atomic_json(summary_path, summary)
    atomic_json(manifest_path, {
        "complete": True, "config_hash": stable_hash(config), **inputs,
        "rows": str(rows_path), "rows_sha256": file_sha256(rows_path),
        "summary": str(summary_path), "summary_sha256": file_sha256(summary_path),
    })
    print(manifest_path)


if __name__ == "__main__":
    main()
