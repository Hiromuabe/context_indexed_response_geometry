from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from .src.residualization import double_center
from .src.review_experiments import auxiliary_token_statistics, center_context_block, review_roots
from .src.statistics import problem_bootstrap, two_way_ratio_bootstrap
from .src.subspaces import explained_variance, normalized_projection_distance, principal_angle_cosines_squared, top_svd
from .src.utils import atomic_json, ensure_layout, file_sha256, load_config, read_json, read_jsonl, stable_hash, stage_is_complete


def _write_csv(path: Path, rows: list[dict]) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    source_root, root = review_roots(config)
    paths = {
        "states": root / "manifests/context_control_states.json",
        "candidates": source_root / "candidate_tokens/candidate_tokens.json",
        "geometry": source_root / "metrics/paper_geometry_summary.json",
    }
    inputs = {f"{key}_sha256": file_sha256(path) for key, path in paths.items()}
    manifest_path = root / "manifests/context_control_analysis.json"
    if not args.force and stage_is_complete(manifest_path, config, inputs):
        print(manifest_path)
        return
    states = read_json(paths["states"])
    candidates = read_json(paths["candidates"])
    geometry = read_json(paths["geometry"])
    contexts = read_jsonl(states["contexts"])
    review = config.get("review_experiments", {})
    layer = int(review.get("context_control_layer", geometry["selected_layer"]))
    rank_requested = int(review.get("context_control_rank", geometry["selected_rank"]))
    entry = next(row for row in states["layers"] if int(row["layer"]) == layer)
    z = np.load(entry["path"], mmap_mode="r")
    source_candidate_indices = list(map(int, states.get("candidate_indices", range(z.shape[1]))))
    local_candidate_axis = {source_index: local_index for local_index, source_index in enumerate(source_candidate_indices)}
    auxiliary = np.asarray([index for index, row in enumerate(contexts) if row["role"] == "auxiliary"], dtype=np.int64)
    nonaux = np.asarray([index for index, row in enumerate(contexts) if row["role"] != "auxiliary"], dtype=np.int64)
    local_position = {contexts[int(full)]["context_id"]: position for position, full in enumerate(nonaux)}
    target_context = {row["target_prefix_id"]: row for row in contexts if row["role"] == "target"}
    controls = [row for row in contexts if row["role"] == "control"]
    rows: list[dict] = []
    angle_rows: list[dict] = []
    cells: dict[str, dict[str, dict[int, dict]]] = defaultdict(lambda: defaultdict(dict))
    for fold in candidates["folds"]:
        train_source_indices = [int(index) for index in fold["train_indices"] if int(index) in local_candidate_axis]
        evaluation_source_indices = [int(index) for index in fold["evaluation_indices"] if int(index) in local_candidate_axis]
        train_indices = [local_candidate_axis[index] for index in train_source_indices]
        evaluation_indices = [local_candidate_axis[index] for index in evaluation_source_indices]
        if len(train_indices) < 2 or not evaluation_indices:
            continue
        train_stats = auxiliary_token_statistics(z, auxiliary, train_indices)
        eval_stats = auxiliary_token_statistics(z, auxiliary, evaluation_indices)
        target_cache: dict[str, dict] = {}
        for target_id in sorted({str(row["target_prefix_id"]) for row in controls if row["target_prefix_id"] in target_context}):
            target_position = local_position[target_id]
            target_train = center_context_block(z, int(nonaux[target_position]), train_indices, *train_stats)
            target_eval = center_context_block(z, int(nonaux[target_position]), evaluation_indices, *eval_stats)
            target_rank = min(rank_requested, target_train.shape[0] - 1)
            if target_rank <= 0:
                continue
            target_basis = top_svd(target_train, target_rank, allow_rank_reduction=True)
            target_cache[target_id] = {
                "basis": target_basis,
                "evaluation": target_eval,
                "ev": explained_variance(target_eval, target_basis),
                "projection": np.square(target_eval @ target_basis).sum(axis=1),
                "denominator": np.square(target_eval).sum(axis=1),
            }
        control_basis_cache: dict[str, np.ndarray] = {}
        for control in controls:
            target_id = str(control["target_prefix_id"])
            if target_id not in target_cache:
                continue
            control_id = str(control["context_id"])
            if control_id not in control_basis_cache:
                control_position = local_position[control_id]
                control_train = center_context_block(z, int(nonaux[control_position]), train_indices, *train_stats)
                control_rank = min(rank_requested, control_train.shape[0] - 1)
                if control_rank <= 0:
                    continue
                control_basis_cache[control_id] = top_svd(control_train, control_rank, allow_rank_reduction=True)
            target_values = target_cache[target_id]
            target_basis = target_values["basis"]
            target_eval = target_values["evaluation"]
            control_basis = control_basis_cache[control_id]
            rank = min(target_basis.shape[1], control_basis.shape[1])
            target_basis = target_basis[:, :rank]
            control_basis = control_basis[:, :rank]
            target_ev = explained_variance(target_eval, target_basis)
            control_ev = explained_variance(target_eval, control_basis)
            cos2 = principal_angle_cosines_squared(target_basis, control_basis)
            problem_id = str(control["target_problem_id"])
            row = {
                "problem_id": problem_id, "target_prefix_id": target_id, "control_context_id": control["context_id"],
                "control_type": control["control_type"], "source_prefix_id": control["source_prefix_id"],
                "layer": layer, "fold": int(fold["fold_id"]), "rank": rank,
                "exact_target_length": bool(control["exact_target_length"]), "same_last_token": bool(control["same_last_token"]),
                "ev_target": target_ev, "ev_control": control_ev, "delta_target_minus_control": target_ev - control_ev,
                "projection_distance": normalized_projection_distance(target_basis, control_basis),
            }
            if "timepoint_offset" in control:
                row["timepoint_offset"] = int(control["timepoint_offset"])
            rows.append(row)
            for angle_index, value in enumerate(cos2):
                angle_rows.append({**{key: row[key] for key in ("problem_id", "target_prefix_id", "control_context_id", "control_type", "fold", "rank")}, "angle_index": angle_index + 1, "cosine_squared": float(value), "angle_degrees": float(np.degrees(np.arccos(np.sqrt(value))))})
            target_projection = target_values["projection"] if rank == target_values["basis"].shape[1] else np.square(target_eval @ target_basis).sum(axis=1)
            control_projection = np.square(target_eval @ control_basis).sum(axis=1)
            denominator = target_values["denominator"]
            for token_axis, candidate_index in enumerate(evaluation_source_indices):
                token_id = int(candidates["candidate_token_ids"][candidate_index])
                cell = cells[str(control["control_type"])][target_id].setdefault(token_id, {"problem_id": problem_id, "target": [], "control": [], "denominator": []})
                cell["target"].append(float(target_projection[token_axis]))
                cell["control"].append(float(control_projection[token_axis]))
                cell["denominator"].append(float(denominator[token_axis]))
    bootstrap = {"replicates": int(config["statistics"]["bootstrap_replicates"]), "ci": float(config["statistics"]["ci"])}
    summary: dict[str, dict] = {}
    for type_index, control_type in enumerate(sorted({row["control_type"] for row in rows})):
        selected = [row for row in rows if row["control_type"] == control_type]
        problem_summary = problem_bootstrap(
            np.asarray([row["delta_target_minus_control"] for row in selected]), np.asarray([row["problem_id"] for row in selected]),
            seed=int(config["seed"]) + 4000 + type_index, **bootstrap,
        )
        distance_summary = problem_bootstrap(
            np.asarray([row["projection_distance"] for row in selected]), np.asarray([row["problem_id"] for row in selected]),
            seed=int(config["seed"]) + 5000 + type_index, **bootstrap,
        )
        target_ids = sorted(cells[control_type])
        candidate_ids = sorted(set.intersection(*(set(cells[control_type][target_id]) for target_id in target_ids))) if target_ids else []
        numerator = np.empty((len(target_ids), len(candidate_ids)), dtype=np.float64)
        denominator = np.empty_like(numerator)
        context_problem_ids = []
        for context_axis, target_id in enumerate(target_ids):
            context_problem_ids.append(next(iter(cells[control_type][target_id].values()))["problem_id"])
            for token_axis, token_id in enumerate(candidate_ids):
                cell = cells[control_type][target_id][token_id]
                numerator[context_axis, token_axis] = np.mean(cell["target"]) - np.mean(cell["control"])
                denominator[context_axis, token_axis] = np.mean(cell["denominator"])
        two_way = two_way_ratio_bootstrap(
            numerator, denominator, np.asarray(context_problem_ids), np.asarray(candidate_ids),
            replicates=int(review.get("two_way_bootstrap_replicates", bootstrap["replicates"])),
            seed=int(config["seed"]) + 6000 + type_index, ci=bootstrap["ci"],
        ) if target_ids and candidate_ids else {}
        summary[control_type] = {
            "rows": len(selected), "unique_target_problems": len(set(row["problem_id"] for row in selected)),
            "delta_target_minus_control_problem_bootstrap": problem_summary,
            "delta_target_minus_control_two_way_bootstrap": two_way,
            "projection_distance_problem_bootstrap": distance_summary,
            "exact_length_fraction": float(np.mean([row["exact_target_length"] for row in selected])),
            "same_last_token_fraction": float(np.mean([row["same_last_token"] for row in selected])),
        }
    rows_path = root / "metrics/context_control_rows.csv"
    angles_path = root / "metrics/context_control_principal_angles.csv"
    summary_path = root / "metrics/context_control_summary.json"
    _write_csv(rows_path, rows)
    _write_csv(angles_path, angle_rows)
    atomic_json(summary_path, {
        "selected_layer": layer, "requested_rank": rank_requested,
        "interpretation": "positive EV differences favor the exact target context over the named control on target held-out candidates",
        "controls": summary,
    })
    atomic_json(manifest_path, {
        "complete": True, "config_hash": stable_hash(config), **inputs,
        "rows": str(rows_path), "rows_sha256": file_sha256(rows_path),
        "principal_angles": str(angles_path), "principal_angles_sha256": file_sha256(angles_path),
        "summary": str(summary_path), "summary_sha256": file_sha256(summary_path),
    })
    print(manifest_path)


if __name__ == "__main__":
    main()
