from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import numpy as np

from .src.residualization import double_center
from .src.statistics import problem_bootstrap
from .src.utils import atomic_json, ensure_layout, file_sha256, load_config, read_json, read_jsonl, stable_hash, stage_is_complete


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _split_indices(indices, train_size, rng):
    shuffled = np.asarray(indices, dtype=np.int64)[rng.permutation(len(indices))]
    return shuffled[:train_size], shuffled[train_size:]


def _prefix_r90(curve, ranks, reference_rank, fraction):
    reference = float(curve[reference_rank])
    achieved = [rank for rank in ranks if float(curve[rank]) >= fraction * reference]
    return min(achieved) if achieved else None


def _batched_rank_ev(train, heldout, ranks, reference_rank):
    """Exact batched held-out EV curve without materializing hidden-size bases."""
    train = np.asarray(train, dtype=np.float32)
    heldout = np.asarray(heldout, dtype=np.float32)
    gram = train @ np.swapaxes(train, 1, 2)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    eigenvalues = np.clip(eigenvalues[:, ::-1], 0.0, None)
    eigenvectors = eigenvectors[:, :, ::-1]
    singular_values = np.sqrt(eigenvalues[:, :reference_rank])
    tolerance = eigenvalues[:, :1] * max(train.shape[1:]) * np.finfo(np.float32).eps
    if bool(np.any(eigenvalues[:, reference_rank - 1] <= tolerance[:, 0])):
        raise RuntimeError("rank-saturation split has numerical rank below the reference rank")
    # heldout @ V = heldout @ train.T @ left_singular / singular_value
    cross = heldout @ np.swapaxes(train, 1, 2)
    coefficients = (cross @ eigenvectors[:, :, :reference_rank]) / singular_values[:, None, :]
    cumulative = np.cumsum(np.square(coefficients, dtype=np.float64), axis=2)
    denominator = np.square(heldout, dtype=np.float64).sum(axis=(1, 2))
    return {rank: cumulative[:, :, rank - 1].sum(axis=1) / denominator for rank in ranks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    root = ensure_layout(config)
    settings = config.get("rank_saturation", {})
    if not bool(settings.get("enabled", True)):
        print("Rank-saturation analysis disabled in configuration")
        return

    hidden_path = root / "manifests/hidden_states.json"
    geometry_path = root / "metrics/paper_geometry_summary.json"
    candidate_path = root / "candidate_tokens/candidate_tokens.json"
    inputs = {
        "hidden_states_sha256": file_sha256(hidden_path),
        "geometry_sha256": file_sha256(geometry_path),
        "candidate_tokens_sha256": file_sha256(candidate_path),
    }
    manifest_path = root / "manifests/rank_saturation.json"
    if not args.force and stage_is_complete(manifest_path, config, inputs):
        print(manifest_path)
        return

    hidden = read_json(hidden_path)
    geometry = read_json(geometry_path)
    candidates = read_json(candidate_path)
    prefixes = read_jsonl(hidden["prefix_snapshot"])
    layer = int(geometry["selected_layer"])
    layer_entry = next(entry for entry in hidden["layers"] if int(entry["layer"]) == layer)
    z = np.load(layer_entry["successor_path"], mmap_mode="r")
    analysis_indices = np.asarray(candidates["analysis_indices"], dtype=np.int64)

    train_size = int(settings.get("train_tokens", len(analysis_indices) // 2))
    evaluation_size = int(settings.get("evaluation_tokens", len(analysis_indices) - train_size))
    if train_size + evaluation_size != len(analysis_indices):
        raise ValueError("rank_saturation train/evaluation sizes must partition all analysis tokens")
    ranks = list(map(int, settings.get("ranks", [1, 2, 4, 8, 16, 32, 64, 96, 127])))
    reference_rank = int(settings.get("reference_rank", max(ranks)))
    if reference_rank not in ranks:
        raise ValueError("rank_saturation.reference_rank must be included in ranks")
    maximum_estimable = train_size - 1
    if max(ranks) > maximum_estimable:
        raise ValueError(f"rank {max(ranks)} exceeds centered training rank ceiling {maximum_estimable}")
    split_count = int(settings.get("random_splits", 8))
    fraction = float(settings.get("r90_fraction", 0.90))
    comparison_rank = int(settings.get("compact_rank", 64))
    gain_tolerance = float(settings.get("relative_gain_after_compact_tolerance", 0.05))

    auxiliary = np.asarray([i for i, row in enumerate(prefixes) if row["problem_group"] == "auxiliary"], dtype=np.int64)
    evaluation = np.asarray([i for i, row in enumerate(prefixes) if row["problem_group"] == "analysis_test"], dtype=np.int64)
    if not len(auxiliary) or not len(evaluation):
        raise ValueError("rank saturation requires auxiliary and analysis_test prefixes")

    rows = []
    r90_rows = []
    rng = np.random.default_rng(int(config["seed"]) + 1601)
    evaluation_z = np.asarray(z[evaluation], dtype=np.float32)
    auxiliary_z = np.asarray(z[auxiliary], dtype=np.float32)
    for split_id in range(split_count):
        train_indices, heldout_indices = _split_indices(analysis_indices, train_size, rng)
        train = double_center(evaluation_z, auxiliary_z, train_indices).residuals
        heldout = double_center(evaluation_z, auxiliary_z, heldout_indices).residuals
        curves = _batched_rank_ev(train, heldout, ranks, reference_rank)
        print(f"[rank_saturation] split={split_id + 1}/{split_count} train={len(train_indices)} heldout={len(heldout_indices)}", flush=True)
        for axis, prefix_index in enumerate(evaluation):
            curve = {}
            for rank in ranks:
                value = float(curves[rank][axis])
                curve[rank] = value
                rows.append({
                    "problem_id": prefixes[int(prefix_index)]["problem_id"],
                    "prefix_id": prefixes[int(prefix_index)]["prefix_id"],
                    "split_id": split_id,
                    "layer": layer,
                    "rank": rank,
                    "heldout_ev": value,
                })
            r90 = _prefix_r90(curve, ranks, reference_rank, fraction)
            relative_gain = (curve[reference_rank] - curve[comparison_rank]) / max(curve[reference_rank], 1e-12)
            r90_rows.append({
                "problem_id": prefixes[int(prefix_index)]["problem_id"],
                "prefix_id": prefixes[int(prefix_index)]["prefix_id"],
                "split_id": split_id,
                "r90": r90,
                f"ev_rank{comparison_rank}": curve[comparison_rank],
                f"ev_rank{reference_rank}": curve[reference_rank],
                "relative_gain_after_compact_rank": relative_gain,
            })

    by_prefix = defaultdict(list)
    for row in r90_rows:
        by_prefix[row["prefix_id"]].append(row)
    prefix_summary = []
    for prefix_id, group in by_prefix.items():
        prefix_summary.append({
            "problem_id": group[0]["problem_id"],
            "prefix_id": prefix_id,
            "median_r90": float(np.median([row["r90"] for row in group if row["r90"] is not None])),
            "mean_relative_gain_after_compact_rank": float(np.mean([row["relative_gain_after_compact_rank"] for row in group])),
        })

    bootstrap_args = {
        "replicates": int(config["statistics"]["bootstrap_replicates"]),
        "seed": int(config["seed"]) + 1609,
        "ci": float(config["statistics"]["ci"]),
    }
    gains = np.asarray([row["mean_relative_gain_after_compact_rank"] for row in prefix_summary])
    problem_ids = np.asarray([row["problem_id"] for row in prefix_summary])
    median_r90 = float(np.median([row["median_r90"] for row in prefix_summary]))
    summary = {
        "definition": {
            "analysis_token_partition": [train_size, evaluation_size],
            "random_splits": split_count,
            "ranks": ranks,
            "r90_fraction": fraction,
            "reference_rank": reference_rank,
            "compact_rank": comparison_rank,
            "relative_gain_tolerance": gain_tolerance,
        },
        "selected_layer": layer,
        "prefixes": len(prefix_summary),
        "median_r90": median_r90,
        "fraction_prefix_median_r90_le_compact_rank": float(np.mean([row["median_r90"] <= comparison_rank for row in prefix_summary])),
        "relative_gain_after_compact_rank": problem_bootstrap(gains, problem_ids, **bootstrap_args),
        "gate_median_r90_le_compact_rank": median_r90 <= comparison_rank,
        "gate_relative_gain_after_compact_small": float(np.mean(gains)) <= gain_tolerance,
        "claim_compact_supported": median_r90 <= comparison_rank and float(np.mean(gains)) <= gain_tolerance,
    }
    rows_path = root / "metrics/rank_saturation_rows.csv"
    r90_path = root / "metrics/rank_saturation_r90.csv"
    prefix_path = root / "metrics/rank_saturation_prefix_summary.csv"
    summary_path = root / "metrics/rank_saturation_summary.json"
    _write_csv(rows_path, rows)
    _write_csv(r90_path, r90_rows)
    _write_csv(prefix_path, prefix_summary)
    atomic_json(summary_path, summary)
    atomic_json(manifest_path, {
        "complete": True,
        "config_hash": stable_hash(config),
        **inputs,
        "rows": str(rows_path),
        "r90_rows": str(r90_path),
        "prefix_summary": str(prefix_path),
        "summary": str(summary_path),
    })
    print(manifest_path)


if __name__ == "__main__":
    main()
