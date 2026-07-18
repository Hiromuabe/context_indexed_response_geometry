from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .analyze_paper_geometry import (
    _add_pooled_top_k_summary,
    _candidate_rank_map,
    _pooled_top_k_rows,
    _rotation_rows_for_ranks,
    _write_csv,
)
from .src.storage import load_residual_entry
from .src.utils import atomic_json, ensure_layout, file_sha256, load_config, read_json, read_jsonl


def _read_pooled_rows(path):
    bool_keys = {"conditional_global_exact_length_bin", "wrong_control_exact_length_bin"}
    int_keys = {"wrong_prefix_count", "folds_seen", "conditional_global_length_bin_distance"}
    with path.open(encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    rows = []
    for source in raw:
        row = {}
        for key, value in source.items():
            if key in bool_keys:
                row[key] = value.lower() == "true"
            elif key in int_keys or key.startswith("top") and key.endswith("_token_count"):
                row[key] = int(value)
            elif key.startswith("delta_"):
                row[key] = float(value)
            else:
                row[key] = value
        rows.append(row)
    return rows


def refresh_geometry(config, root):
    summary_path = root / "metrics/paper_geometry_summary.json"
    residual_path = root / "manifests/residuals.json"
    hidden_path = root / "manifests/hidden_states.json"
    wrong_path = root / "controls/wrong_prefixes.jsonl"
    for path in (summary_path, residual_path, hidden_path, wrong_path, root / "candidate_tokens/candidate_tokens.json"):
        if not path.is_file():
            raise FileNotFoundError(path)

    summary = read_json(summary_path)
    residual = read_json(residual_path)
    hidden = read_json(hidden_path)
    prefixes = read_jsonl(hidden["prefix_snapshot"])
    wrong_rows = read_jsonl(wrong_path)
    wrong_map = {row["prefix_id"]: row["wrong_prefix_ids"] for row in wrong_rows}
    relaxed_wrong_targets = {row["prefix_id"] for row in wrong_rows if int(row.get("relaxed_length_wrong_prefixes", 0)) > 0}
    top_ks = list(map(int, config["analysis"]["high_probability_top_ks"]))
    high_minimum = int(config["analysis"]["high_probability_min_tokens"])

    legacy = summary.setdefault("legacy_per_fold_top_k", {})
    for top_k in top_ks:
        for key in (f"delta_conditional_global_top{top_k}", f"delta_wrong_top{top_k}", f"delta_wrong_top{top_k}_exact_bin"):
            if key in summary and key not in legacy:
                legacy[key] = summary[key]
    if "top_k_coverage" in summary and "coverage" not in legacy:
        legacy["coverage"] = summary["top_k_coverage"]

    pooled_path = root / "metrics/paper_topk_pooled_rows.csv"
    current_residual_hash = file_sha256(residual_path)
    prior_refresh = summary.get("posthoc_refresh", {})
    if pooled_path.is_file() and prior_refresh.get("residuals_sha256") == current_residual_hash and prior_refresh.get("wrong_prefixes_sha256") == file_sha256(wrong_path):
        pooled = _read_pooled_rows(pooled_path)
    else:
        candidates = read_json(root / "candidate_tokens/candidate_tokens.json")
        candidate_ids = np.asarray(candidates["candidate_token_ids"], dtype=np.int64)
        source_root = Path(str(config.get("source_results_root", root)))
        rank_map = _candidate_rank_map(source_root, prefixes, candidate_ids)
        pooled = _pooled_top_k_rows(
            residual, int(summary["selected_layer"]), int(summary["selected_rank"]),
            prefixes, wrong_map, relaxed_wrong_targets, rank_map, top_ks,
        )
    _add_pooled_top_k_summary(summary, pooled, top_ks, high_minimum, config)

    expected_prefixes = int(config["data"]["evaluation_prefixes"])
    folds = int(config["candidates"]["folds"])
    exact_global = sum(bool(row["conditional_global_exact_length_bin"]) for row in pooled)
    complete_wrong = sum(int(row["wrong_prefix_count"]) == int(config["controls"]["wrong_prefixes_per_target"]) for row in pooled)
    summary["conditional_global_coverage"] = {
        "expected_prefixes": expected_prefixes,
        "observed_prefixes_with_fallback": len(pooled),
        "exact_prefixes": exact_global,
        "fallback_prefixes": len(pooled) - exact_global,
        "exact_prefix_fold_rows": exact_global * folds,
        "expected_prefix_fold_rows": expected_prefixes * folds,
        "exact_fraction": exact_global / max(1, expected_prefixes),
        "complete_with_fallback": len(pooled) == expected_prefixes,
        "fallback": "nearest length bin within the same reasoning-progress bin",
    }
    summary["wrong_basis_coverage"] = {
        "expected_prefixes": expected_prefixes,
        "complete_prefixes": complete_wrong,
        "wrong_prefixes_per_target": int(config["controls"]["wrong_prefixes_per_target"]),
        "complete": len(pooled) == expected_prefixes and complete_wrong == expected_prefixes,
    }
    summary["posthoc_refresh"] = {
        "reason": "pool Top-k evidence across held-out folds and audit the explicit conditional-Global fallback",
        "residuals_sha256": current_residual_hash,
        "wrong_prefixes_sha256": file_sha256(wrong_path),
    }
    _write_csv(pooled_path, pooled)
    atomic_json(summary_path, summary)

    manifest_path = root / "manifests/paper_geometry.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        manifest.update({
            "summary": str(summary_path), "pooled_top_k_rows": str(pooled_path),
            "pooled_top_k_rows_sha256": file_sha256(pooled_path), "posthoc_refreshed": True,
        })
        atomic_json(manifest_path, manifest)
    return summary_path


def refresh_multirank_rotation(config, root):
    """Build the rank-wise rotation table from saved residuals only.

    This deliberately leaves the completed geometry summary, permutation output,
    and held-out EV rows untouched.  It is intended for legacy runs that saved
    only the selected-rank rotation table.
    """
    summary_path = root / "metrics/paper_geometry_summary.json"
    residual_path = root / "manifests/residuals.json"
    hidden_path = root / "manifests/hidden_states.json"
    wrong_path = root / "controls/wrong_prefixes.jsonl"
    for path in (summary_path, residual_path, hidden_path, wrong_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    summary = read_json(summary_path)
    residual = read_json(residual_path)
    hidden = read_json(hidden_path)
    prefixes = read_jsonl(hidden["prefix_snapshot"])
    wrong_rows = read_jsonl(wrong_path)
    wrong_map = {row["prefix_id"]: row["wrong_prefix_ids"] for row in wrong_rows}
    relaxed_wrong_targets = {
        row["prefix_id"]
        for row in wrong_rows
        if int(row.get("relaxed_length_wrong_prefixes", 0)) > 0
    }
    selected_layer = int(summary["selected_layer"])
    ranks = sorted(set(map(int, config["analysis"]["ranks"])))
    rows = []
    selected_entries = [
        entry for entry in residual["entries"]
        if int(entry["layer"]) == selected_layer
    ]
    if not selected_entries:
        raise RuntimeError(f"no residual entries found for selected layer {selected_layer}")

    for entry_number, entry in enumerate(selected_entries, start=1):
        fold = int(entry["fold"])
        print(
            f"[rotation_rank_curve] fold={fold} entry={entry_number}/{len(selected_entries)} "
            f"layer={selected_layer} ranks={ranks}",
            flush=True,
        )
        bundle = load_residual_entry(entry)
        fold_rows = _rotation_rows_for_ranks(
            bundle["train_residuals"],
            prefixes,
            bundle["nonauxiliary_prefix_indices"],
            wrong_map,
            ranks,
            selected_layer,
            fold,
            int(config["seed"]),
        )
        for row in fold_rows:
            row["wrong_control_exact_length_bin"] = row["prefix_id"] not in relaxed_wrong_targets
        rows.extend(fold_rows)

    expected_rows = (
        int(config["data"]["evaluation_prefixes"])
        * int(config["candidates"]["folds"])
        * len(ranks)
    )
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"incomplete multirank rotation table: observed {len(rows)} rows, "
            f"expected {expected_rows}"
        )

    output_path = root / "metrics/paper_rotation_rank_rows.csv"
    _write_csv(output_path, rows)
    manifest_path = root / "manifests/paper_geometry.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        manifest.update({
            "rotation_rank_rows": str(output_path),
            "rotation_rank_rows_sha256": file_sha256(output_path),
            "multirank_rotation_refreshed_from_saved_residuals": True,
        })
        atomic_json(manifest_path, manifest)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh Top-k and control-coverage audits from completed geometry artifacts without rerunning rank curves or permutations"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--multirank-rotation-only",
        action="store_true",
        help="Generate paper_rotation_rank_rows.csv from saved residuals without rerunning EV or permutations",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.multirank_rotation_only:
        # A legacy completed run must retain its original resolved config.
        # This read-only-derived refresh does not create a new run layout, so it
        # should not reject the run merely because the working config gained an
        # unrelated field after completion.
        root = Path(config["results_root"])
        print(refresh_multirank_rotation(config, root))
    else:
        root = ensure_layout(config)
        print(refresh_geometry(config, root))


if __name__ == "__main__":
    main()
