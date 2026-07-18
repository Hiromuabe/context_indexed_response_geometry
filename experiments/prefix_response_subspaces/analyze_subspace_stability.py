from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .src.statistics import problem_bootstrap
from .src.review_experiments import review_roots
from .src.storage import load_residual_entry
from .src.subspaces import normalized_projection_distance, principal_angle_cosines_squared, top_svd
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


def reliability_corrected_distance(within_similarity: float, between_similarity: float) -> float:
    if not np.isfinite(within_similarity) or within_similarity <= 0 or not np.isfinite(between_similarity):
        return float("nan")
    return float(1.0 - np.clip(between_similarity / within_similarity, 0.0, 1.0))


def _corrected_bootstrap(rows: list[dict], replicates: int, seed: int, ci: float) -> dict[str, float]:
    problem_ids = np.asarray([row["problem_id"] for row in rows])
    unique = np.unique(problem_ids)
    grouped = []
    for problem_id in unique:
        selected = [row for row in rows if row["problem_id"] == problem_id]
        grouped.append((np.mean([row["within_similarity"] for row in selected]), np.mean([row["between_similarity"] for row in selected])))
    grouped = np.asarray(grouped, dtype=np.float64)
    if not len(grouped):
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_problems": 0}
    point = reliability_corrected_distance(float(grouped[:, 0].mean()), float(grouped[:, 1].mean())) if len(grouped) else float("nan")
    rng = np.random.default_rng(int(seed))
    samples = []
    for _ in range(int(replicates)):
        selected = grouped[rng.integers(0, len(grouped), size=len(grouped))]
        samples.append(reliability_corrected_distance(float(selected[:, 0].mean()), float(selected[:, 1].mean())))
    alpha = (1.0 - float(ci)) / 2.0
    low, high = np.quantile(np.asarray(samples), [alpha, 1.0 - alpha]) if samples else (float("nan"), float("nan"))
    return {"mean": point, "ci_low": float(low), "ci_high": float(high), "n_problems": int(len(unique))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    source_root, root = review_roots(config)
    paths = {
        "residuals": source_root / "manifests/residuals.json",
        "hidden": source_root / "manifests/hidden_states.json",
        "wrong": source_root / "controls/wrong_prefixes.jsonl",
        "geometry": source_root / "metrics/paper_geometry_summary.json",
    }
    inputs = {f"{key}_sha256": file_sha256(path) for key, path in paths.items()}
    manifest_path = root / "manifests/subspace_stability.json"
    if not args.force and stage_is_complete(manifest_path, config, inputs):
        print(manifest_path)
        return
    residuals = read_json(paths["residuals"])
    hidden = read_json(paths["hidden"])
    geometry = read_json(paths["geometry"])
    prefixes = read_jsonl(hidden["prefix_snapshot"])
    wrong_map = {row["prefix_id"]: row["wrong_prefix_ids"] for row in read_jsonl(paths["wrong"])}
    review = config.get("review_experiments", {})
    layer = int(review.get("stability_layer", geometry["selected_layer"]))
    rank_requested = int(review.get("stability_rank", geometry["selected_rank"]))
    candidate_bootstraps = int(review.get("subspace_candidate_bootstrap_replicates", 10))
    uncertainty_target_limit = int(review.get("subspace_candidate_bootstrap_targets", 32))
    maximum_targets = review.get("stability_max_targets")
    all_test_ids = sorted(str(row["prefix_id"]) for row in prefixes if row["problem_group"] == "analysis_test")
    uncertainty_rng = np.random.default_rng(int(config["seed"]) + 8051)
    uncertainty_target_ids = set(
        uncertainty_rng.choice(all_test_ids, size=min(uncertainty_target_limit, len(all_test_ids)), replace=False).tolist()
    )
    rows: list[dict] = []
    angles: list[dict] = []
    uncertainty_rows: list[dict] = []
    for entry in residuals["entries"]:
        if int(entry["layer"]) != layer:
            continue
        bundle = load_residual_entry(entry)
        train = bundle["train_residuals"]
        nonaux = bundle["nonauxiliary_prefix_indices"]
        position = {prefixes[int(full)]["prefix_id"]: index for index, full in enumerate(nonaux)}
        targets = [prefixes[int(full)] for full in nonaux if prefixes[int(full)]["problem_group"] == "analysis_test"]
        targets.sort(key=lambda row: str(row["prefix_id"]))
        if maximum_targets is not None:
            targets = targets[: int(maximum_targets)]
        for target_number, prefix in enumerate(targets):
            prefix_id = str(prefix["prefix_id"])
            samples = np.asarray(train[position[prefix_id]], dtype=np.float64)
            rng = np.random.default_rng(int(config["seed"]) + int(entry["fold"]) * 100003 + target_number)
            order = rng.permutation(len(samples))
            midpoint = len(order) // 2
            first, second = order[:midpoint], order[midpoint:]
            rank = min(rank_requested, len(first) - 1, len(second) - 1)
            if rank <= 0:
                continue
            first_basis = top_svd(samples[first], rank, allow_rank_reduction=True)
            second_basis = top_svd(samples[second], rank, allow_rank_reduction=True)
            within_cos2 = principal_angle_cosines_squared(first_basis, second_basis)
            wrong_cosines = []
            for wrong_id in wrong_map.get(prefix_id, []):
                if wrong_id not in position:
                    continue
                wrong_samples = np.asarray(train[position[wrong_id]], dtype=np.float64)
                wrong_basis = top_svd(wrong_samples[second], rank, allow_rank_reduction=True)
                wrong_cosines.append(principal_angle_cosines_squared(first_basis, wrong_basis))
            if not wrong_cosines:
                continue
            between_cos2 = np.mean(np.stack(wrong_cosines), axis=0)
            row = {
                "problem_id": str(prefix["problem_id"]), "prefix_id": prefix_id, "layer": layer,
                "fold": int(entry["fold"]), "rank": rank, "fit_candidates_per_half": len(first),
                "within_similarity": float(within_cos2.mean()), "between_similarity": float(between_cos2.mean()),
                "within_distance": float(1.0 - within_cos2.mean()), "between_distance": float(1.0 - between_cos2.mean()),
                "between_minus_within_distance": float(within_cos2.mean() - between_cos2.mean()),
                "wrong_context_count": len(wrong_cosines),
            }
            rows.append(row)
            for angle_index in range(rank):
                angles.append({
                    "problem_id": row["problem_id"], "prefix_id": prefix_id, "fold": row["fold"],
                    "angle_index": angle_index + 1, "within_cosine_squared": float(within_cos2[angle_index]),
                    "between_wrong_mean_cosine_squared": float(between_cos2[angle_index]),
                    "within_angle_degrees": float(np.degrees(np.arccos(np.sqrt(within_cos2[angle_index])))),
                    "between_angle_degrees": float(np.degrees(np.arccos(np.sqrt(between_cos2[angle_index])))),
                })
            if prefix_id in uncertainty_target_ids:
                full_basis = top_svd(samples, rank, allow_rank_reduction=True)
                distances = []
                for _ in range(candidate_bootstraps):
                    sampled = rng.integers(0, len(samples), size=len(samples))
                    bootstrap_basis = top_svd(samples[sampled], rank, allow_rank_reduction=True)
                    distances.append(normalized_projection_distance(full_basis, bootstrap_basis))
                uncertainty_rows.append({
                    "problem_id": row["problem_id"], "prefix_id": prefix_id, "fold": row["fold"],
                    "candidate_bootstrap_replicates": candidate_bootstraps,
                    "distance_to_full_fit_mean": float(np.mean(distances)),
                    "distance_to_full_fit_q025": float(np.quantile(distances, 0.025)),
                    "distance_to_full_fit_q975": float(np.quantile(distances, 0.975)),
                })
    bootstrap = {"replicates": int(config["statistics"]["bootstrap_replicates"]), "ci": float(config["statistics"]["ci"])}
    problem_ids = np.asarray([row["problem_id"] for row in rows])
    hidden_size = int(hidden["model"]["hidden_size"])
    effective_rank = int(np.median([row["rank"] for row in rows])) if rows else rank_requested
    random_similarity = effective_rank / hidden_size
    summary = {
        "layer": layer, "requested_rank": rank_requested, "rows": len(rows),
        "random_subspace_baseline": {
            "ambient_dimension": hidden_size, "rank": effective_rank,
            "expected_mean_cosine_squared": random_similarity,
            "expected_normalized_projection_distance": 1.0 - random_similarity,
            "definition": "analytic expectation for independent Haar-random equal-rank subspaces",
        },
        "within_distance": problem_bootstrap(np.asarray([row["within_distance"] for row in rows]), problem_ids, seed=int(config["seed"]) + 8101, **bootstrap),
        "between_distance": problem_bootstrap(np.asarray([row["between_distance"] for row in rows]), problem_ids, seed=int(config["seed"]) + 8102, **bootstrap),
        "between_minus_within_distance": problem_bootstrap(np.asarray([row["between_minus_within_distance"] for row in rows]), problem_ids, seed=int(config["seed"]) + 8103, **bootstrap),
        "reliability_corrected_between_distance": _corrected_bootstrap(rows, bootstrap["replicates"], int(config["seed"]) + 8104, bootstrap["ci"]),
        "reliability_correction_definition": "1 - mean_between_similarity / mean_split_half_within_similarity; equal-reliability attenuation model",
        "candidate_bootstrap_latent_subspace_uncertainty": problem_bootstrap(
            np.asarray([row["distance_to_full_fit_mean"] for row in uncertainty_rows]),
            np.asarray([row["problem_id"] for row in uncertainty_rows]), seed=int(config["seed"]) + 8105, **bootstrap,
        ),
        "candidate_bootstrap_sampling": {
            "configured_target_limit": uncertainty_target_limit,
            "sampled_target_prefixes": len(uncertainty_target_ids),
            "fold_rows": len(uncertainty_rows),
            "selection": "fixed seeded sample of analysis_test prefix IDs, shared across folds",
        },
        "uncertainty_method": "nonparametric candidate-ID bootstrap of the fitted latent subspace (a distribution-free alternative to probabilistic PCA)",
    }
    rows_path = root / "metrics/subspace_stability_rows.csv"
    angles_path = root / "metrics/subspace_principal_angle_spectrum.csv"
    uncertainty_path = root / "metrics/subspace_candidate_bootstrap_rows.csv"
    summary_path = root / "metrics/subspace_stability_summary.json"
    _write_csv(rows_path, rows)
    _write_csv(angles_path, angles)
    _write_csv(uncertainty_path, uncertainty_rows)
    atomic_json(summary_path, summary)
    atomic_json(manifest_path, {
        "complete": True, "config_hash": stable_hash(config), **inputs,
        "rows": str(rows_path), "rows_sha256": file_sha256(rows_path),
        "principal_angles": str(angles_path), "principal_angles_sha256": file_sha256(angles_path),
        "candidate_bootstrap": str(uncertainty_path), "candidate_bootstrap_sha256": file_sha256(uncertainty_path),
        "summary": str(summary_path), "summary_sha256": file_sha256(summary_path),
    })
    print(manifest_path)


if __name__ == "__main__":
    main()
