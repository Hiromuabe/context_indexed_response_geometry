from __future__ import annotations

import argparse
import csv

import numpy as np

from .analyze_paper_geometry import _basis
from .src.statistics import problem_bootstrap
from .src.subspaces import explained_variance, randomized_mean_projection_eigensystem, remove_shared_subspace
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


def _specificity_delta(target, correct, wrong_bases):
    wrong_bases = [basis for basis in wrong_bases if basis is not None and basis.shape[1]]
    if correct is None or not correct.shape[1] or not wrong_bases:
        return float("nan"), 0, 0
    effective = min([correct.shape[1], *[basis.shape[1] for basis in wrong_bases]])
    local_ev = explained_variance(target, correct[:, :effective])
    wrong_ev = float(np.mean([explained_variance(target, basis[:, :effective]) for basis in wrong_bases]))
    return local_ev - wrong_ev, effective, len(wrong_bases)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    root = ensure_layout(config)
    settings = config.get("shared_backbone", {})
    if not bool(settings.get("enabled", True)):
        print("Shared-backbone analysis disabled in configuration")
        return

    residual_path = root / "manifests/residuals.json"
    geometry_path = root / "metrics/paper_geometry_summary.json"
    hidden_path = root / "manifests/hidden_states.json"
    wrong_path = root / "controls/wrong_prefixes.jsonl"
    inputs = {
        "residuals_sha256": file_sha256(residual_path),
        "geometry_sha256": file_sha256(geometry_path),
        "hidden_states_sha256": file_sha256(hidden_path),
        "wrong_prefixes_sha256": file_sha256(wrong_path),
    }
    manifest_path = root / "manifests/shared_backbone.json"
    if not args.force and stage_is_complete(manifest_path, config, inputs):
        print(manifest_path)
        return

    residual = read_json(residual_path)
    geometry = read_json(geometry_path)
    hidden = read_json(hidden_path)
    prefixes = read_jsonl(hidden["prefix_snapshot"])
    wrong_rows = read_jsonl(wrong_path)
    wrong_map = {row["prefix_id"]: row["wrong_prefix_ids"] for row in wrong_rows}
    layer = int(geometry["selected_layer"])
    rank = int(geometry["selected_rank"])
    threshold = float(settings.get("shared_eigenvalue_threshold", 0.50))
    # q=0 records the undecomposed reference; the threshold-selected q is the
    # confirmatory shared-removal result.  Larger exploratory q grids can be
    # requested explicitly but are not paid for by default.
    q_grid = sorted(set(map(int, settings.get("q_grid", [0]))))
    if any(q < 0 or q > rank for q in q_grid):
        raise ValueError("shared_backbone.q_grid must lie between zero and the selected rank")

    rows = []
    spectrum_rows = []
    selected_rows = []
    for entry in residual["entries"]:
        if int(entry["layer"]) != layer:
            continue
        bundle = np.load(entry["path"])
        train = bundle["train_residuals"]
        heldout = bundle["evaluation_residuals"]
        nonaux = bundle["nonauxiliary_prefix_indices"]
        local = {
            prefixes[int(full)]["prefix_id"]: _basis(train[axis], rank)
            for axis, full in enumerate(nonaux)
        }
        development_bases = [
            local[prefixes[int(full)]["prefix_id"]]
            for full in nonaux
            if prefixes[int(full)]["problem_group"] == "analysis_dev"
            and local.get(prefixes[int(full)]["prefix_id"]) is not None
        ]
        print(f"[shared_backbone] fold={int(entry['fold'])} fitting leading mean-projector eigensystem", flush=True)
        eigenvalues, eigenvectors = randomized_mean_projection_eigensystem(
            development_bases,
            rank,
            seed=int(config["seed"]) + 1901 + int(entry["fold"]),
            oversampling=int(settings.get("eigensolver_oversampling", 16)),
            power_iterations=int(settings.get("eigensolver_power_iterations", 2)),
        )
        selected_q = min(rank, int(np.sum(eigenvalues >= threshold)))
        fold = int(entry["fold"])
        for index, eigenvalue in enumerate(eigenvalues[:rank]):
            spectrum_rows.append({"fold": fold, "direction": index + 1, "eigenvalue": float(eigenvalue)})
        evaluated_qs = sorted(set(q_grid + [selected_q]))
        specific_cache = {}

        def specific(prefix_id, q):
            key = (prefix_id, q)
            if key not in specific_cache:
                basis = local.get(prefix_id)
                specific_cache[key] = remove_shared_subspace(basis, eigenvectors[:, :q]) if basis is not None else None
            return specific_cache[key]

        for axis, full in enumerate(nonaux):
            prefix = prefixes[int(full)]
            if prefix["problem_group"] != "analysis_test":
                continue
            prefix_id = prefix["prefix_id"]
            correct = local.get(prefix_id)
            for q in evaluated_qs:
                correct_specific = specific(prefix_id, q) if correct is not None else None
                wrong_specific = [specific(wrong_id, q) for wrong_id in wrong_map.get(prefix_id, [])]
                delta, effective, wrong_count = _specificity_delta(heldout[axis], correct_specific, wrong_specific)
                row = {
                    "problem_id": prefix["problem_id"],
                    "prefix_id": prefix_id,
                    "fold": fold,
                    "layer": layer,
                    "rank": rank,
                    "q": q,
                    "q_selection": "eigenvalue_threshold" if q == selected_q else "grid",
                    "shared_eigenvalue_threshold": threshold,
                    "delta_specific_local_wrong": delta,
                    "effective_specific_rank": effective,
                    "wrong_prefix_count": wrong_count,
                }
                rows.append(row)
                if q == selected_q:
                    selected_rows.append(row)
        print(f"[shared_backbone] fold={fold} dev_bases={len(development_bases)} selected_q={selected_q}", flush=True)

    if not selected_rows:
        raise RuntimeError("shared-backbone analysis produced no evaluation rows")
    finite_selected = [row for row in selected_rows if np.isfinite(row["delta_specific_local_wrong"])]
    bootstrap_args = {
        "replicates": int(config["statistics"]["bootstrap_replicates"]),
        "seed": int(config["seed"]) + 1907,
        "ci": float(config["statistics"]["ci"]),
    }
    selected_effect = problem_bootstrap(
        np.asarray([row["delta_specific_local_wrong"] for row in finite_selected]),
        np.asarray([row["problem_id"] for row in finite_selected]),
        **bootstrap_args,
    )
    by_direction = {}
    for direction in range(1, rank + 1):
        values = [row["eigenvalue"] for row in spectrum_rows if row["direction"] == direction]
        if values:
            by_direction[str(direction)] = {
                "mean": float(np.mean(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
    selected_qs = [int(np.sum(np.asarray([row["eigenvalue"] for row in spectrum_rows if row["fold"] == fold]) >= threshold)) for fold in sorted({row["fold"] for row in spectrum_rows})]
    summary = {
        "definition": {
            "mean_projector": "mean_i U_i U_i^T",
            "eigensolver": "deterministic randomized leading eigensystem without forming hidden^2",
            "shared_basis_fit_split": "analysis_dev only",
            "shared_direction_rule": f"eigenvalue >= {threshold}",
            "specific_component": "orth((I - S_q S_q^T) U_i)",
            "evaluation_split": "analysis_test",
        },
        "selected_layer": layer,
        "selected_rank": rank,
        "shared_q_by_fold": selected_qs,
        "shared_q_median": float(np.median(selected_qs)),
        "mean_projector_eigenvalues": by_direction,
        "delta_specific_local_wrong": selected_effect,
        "gate_prefix_specificity_remains_after_shared_removal": float(selected_effect["ci_low"]) > 0,
        "coverage": {
            "expected_prefix_fold_rows": int(config["data"]["evaluation_prefixes"]) * int(config["candidates"]["folds"]),
            "observed_selected_q_rows": len(selected_rows),
            "finite_selected_q_rows": len(finite_selected),
        },
    }
    rows_path = root / "metrics/shared_backbone_rows.csv"
    spectrum_path = root / "metrics/shared_backbone_spectrum.csv"
    summary_path = root / "metrics/shared_backbone_summary.json"
    _write_csv(rows_path, rows)
    _write_csv(spectrum_path, spectrum_rows)
    atomic_json(summary_path, summary)
    atomic_json(manifest_path, {
        "complete": True,
        "config_hash": stable_hash(config),
        **inputs,
        "rows": str(rows_path),
        "spectrum": str(spectrum_path),
        "summary": str(summary_path),
    })
    print(manifest_path)


if __name__ == "__main__":
    main()
