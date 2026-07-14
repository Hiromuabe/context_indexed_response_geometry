from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .analyze_paper_geometry import (
    _add_pooled_top_k_summary,
    _candidate_rank_map,
    _pooled_top_k_rows,
    _write_csv,
)
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh Top-k and control-coverage audits from completed geometry artifacts without rerunning rank curves or permutations"
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    print(refresh_geometry(config, ensure_layout(config)))


if __name__ == "__main__":
    main()
