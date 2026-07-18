from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .src.residualization import double_center
from .src.review_experiments import review_roots
from .src.statistics import problem_bootstrap
from .src.subspaces import explained_variance, normalized_projection_distance, principal_angle_cosines_squared, top_svd
from .src.utils import atomic_json, ensure_layout, file_sha256, load_config, read_json, read_jsonl, stable_hash, stage_is_complete


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def scaled_linearization_error(observed: np.ndarray, predicted: np.ndarray, eps: float = 1e-12) -> dict[str, float]:
    finite = np.asarray(observed, dtype=np.float64)
    linear = np.asarray(predicted, dtype=np.float64)
    if finite.shape != linear.shape:
        raise ValueError("observed and predicted response matrices must match")
    denominator = float(np.square(finite).sum())
    prediction_energy = float(np.square(linear).sum())
    scale = float(np.sum(finite * linear) / prediction_energy) if prediction_energy > eps else float("nan")
    error = float(np.square(finite - scale * linear).sum() / denominator) if denominator > eps and np.isfinite(scale) else float("nan")
    cosine = float(np.sum(finite * linear) / np.sqrt(denominator * prediction_energy)) if denominator > eps and prediction_energy > eps else float("nan")
    return {"optimal_scalar": scale, "relative_squared_error": error, "matrix_cosine": cosine}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    source_root, root = review_roots(config)
    paths = {
        "jacobian": root / "manifests/jacobian_responses.json",
        "hidden": source_root / "manifests/hidden_states.json",
        "candidates": source_root / "candidate_tokens/candidate_tokens.json",
        "geometry": source_root / "metrics/paper_geometry_summary.json",
    }
    inputs = {f"{key}_sha256": file_sha256(path) for key, path in paths.items()}
    manifest_path = root / "manifests/jacobian_alignment.json"
    if not args.force and stage_is_complete(manifest_path, config, inputs):
        print(manifest_path)
        return
    jacobian = read_json(paths["jacobian"])
    hidden = read_json(paths["hidden"])
    candidates = read_json(paths["candidates"])
    geometry = read_json(paths["geometry"])
    prefixes = read_jsonl(hidden["prefix_snapshot"])
    prefix_axis = {str(row["prefix_id"]): index for index, row in enumerate(prefixes)}
    layer = int(jacobian["layer"])
    rank_requested = int(config.get("review_experiments", {}).get("jacobian_alignment_rank", geometry["selected_rank"]))
    hidden_entry = next(row for row in hidden["layers"] if int(row["layer"]) == layer)
    z = np.load(hidden_entry["successor_path"], mmap_mode="r")
    candidate_indices = list(map(int, jacobian["candidate_indices"]))
    target_entries = [row for row in jacobian["contexts"] if row["role"] == "target"]
    auxiliary_entries = [row for row in jacobian["contexts"] if row["role"] == "auxiliary"]
    if not auxiliary_entries:
        raise RuntimeError("Jacobian interaction comparison requires sampled auxiliary contexts")
    target_axes = np.asarray([prefix_axis[row["prefix_id"]] for row in target_entries], dtype=np.int64)
    full_auxiliary_axes = np.asarray([index for index, row in enumerate(prefixes) if row["problem_group"] == "auxiliary"], dtype=np.int64)
    target_z = np.asarray(z[target_axes[:, None], np.asarray(candidate_indices)[None, :], :], dtype=np.float32)
    auxiliary_z = np.asarray(z[full_auxiliary_axes[:, None], np.asarray(candidate_indices)[None, :], :], dtype=np.float32)
    finite = double_center(target_z, auxiliary_z, np.arange(len(candidate_indices))).residuals
    embedding = np.load(jacobian["candidate_embedding_svd"])
    left = np.asarray(embedding["left_vectors"], dtype=np.float64)
    auxiliary_weighted = np.mean([
        np.asarray(np.load(row["path"])["weighted_responses"], dtype=np.float64) for row in auxiliary_entries
    ], axis=0)
    rows: list[dict] = []
    angle_rows: list[dict] = []
    for target_index, entry in enumerate(target_entries):
        target_weighted = np.asarray(np.load(entry["path"])["weighted_responses"], dtype=np.float64)
        interaction_weighted = target_weighted - auxiliary_weighted
        rank = min(rank_requested, finite[target_index].shape[0] - 1, interaction_weighted.shape[0])
        finite_basis = top_svd(finite[target_index], rank, allow_rank_reduction=True)
        jacobian_basis = top_svd(interaction_weighted, rank, allow_rank_reduction=True)
        cos2 = principal_angle_cosines_squared(finite_basis, jacobian_basis)
        predicted = left @ interaction_weighted
        linearization = scaled_linearization_error(finite[target_index], predicted)
        row = {
            "problem_id": entry["problem_id"], "prefix_id": entry["prefix_id"], "layer": layer, "rank": rank,
            "projection_distance": normalized_projection_distance(finite_basis, jacobian_basis),
            "mean_cosine_squared": float(cos2.mean()),
            "finite_response_ev_by_jacobian_subspace": explained_variance(finite[target_index], jacobian_basis),
            "jacobian_response_ev_by_finite_subspace": explained_variance(interaction_weighted, finite_basis),
            **linearization,
        }
        rows.append(row)
        for angle_index, value in enumerate(cos2):
            angle_rows.append({
                "problem_id": entry["problem_id"], "prefix_id": entry["prefix_id"], "angle_index": angle_index + 1,
                "cosine_squared": float(value), "angle_degrees": float(np.degrees(np.arccos(np.sqrt(value)))),
            })
    bootstrap_args = {
        "problem_ids": np.asarray([row["problem_id"] for row in rows]),
        "replicates": int(config["statistics"]["bootstrap_replicates"]),
        "ci": float(config["statistics"]["ci"]),
    }
    summary = {
        "layer": layer, "requested_rank": rank_requested,
        "candidate_embedding_components": int(jacobian["embedding_components"]),
        "candidate_embedding_energy_retained_fraction": float(jacobian["embedding_energy_retained_fraction"]),
        "auxiliary_jacobian_contexts": len(auxiliary_entries), "target_contexts": len(target_entries),
        "comparison_definition": "finite double-centered U_i versus top eigenspace of (J_i - mean_aux J_a) E at the mean candidate embedding",
        "projection_distance": problem_bootstrap(np.asarray([row["projection_distance"] for row in rows]), seed=int(config["seed"]) + 7101, **bootstrap_args),
        "finite_response_ev_by_jacobian_subspace": problem_bootstrap(np.asarray([row["finite_response_ev_by_jacobian_subspace"] for row in rows]), seed=int(config["seed"]) + 7102, **bootstrap_args),
        "scaled_linearization_relative_squared_error": problem_bootstrap(np.asarray([row["relative_squared_error"] for row in rows]), seed=int(config["seed"]) + 7103, **bootstrap_args),
        "interpretation": {
            "high_alignment": "the response space is largely a context-dependent local sensitivity image",
            "low_alignment_or_high_error": "finite candidate changes or nonlinear structure contribute beyond the local Jacobian",
        },
    }
    rows_path = root / "metrics/jacobian_alignment_rows.csv"
    angles_path = root / "metrics/jacobian_alignment_principal_angles.csv"
    summary_path = root / "metrics/jacobian_alignment_summary.json"
    _write_csv(rows_path, rows)
    _write_csv(angles_path, angle_rows)
    atomic_json(summary_path, summary)
    atomic_json(manifest_path, {
        "complete": True, "config_hash": stable_hash(config), **inputs,
        "rows": str(rows_path), "rows_sha256": file_sha256(rows_path),
        "principal_angles": str(angles_path), "principal_angles_sha256": file_sha256(angles_path),
        "summary": str(summary_path), "summary_sha256": file_sha256(summary_path),
    })
    print(manifest_path)


if __name__ == "__main__":
    main()
