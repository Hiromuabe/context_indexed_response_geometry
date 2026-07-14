from __future__ import annotations

import argparse
import csv

import numpy as np

from .analyze_paper_geometry import _fit_controls, _interaction_energy, _resolve_conditional_basis
from .src.residualization import center_train_and_evaluation
from .src.statistics import problem_bootstrap
from .src.subspaces import explained_variance
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


def _paired_ev(target, left, right):
    if left is None or right is None:
        return float("nan"), float("nan"), float("nan"), 0
    effective = min(left.shape[1], right.shape[1])
    if effective <= 0:
        return float("nan"), float("nan"), float("nan"), 0
    left_ev = explained_variance(target, left[:, :effective])
    right_ev = explained_variance(target, right[:, :effective])
    return left_ev, right_ev, left_ev - right_ev, effective


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    root = ensure_layout(config)
    settings = config.get("first_layer_mechanism", {})
    if not bool(settings.get("enabled", True)):
        print("First-layer mechanism analysis disabled in configuration")
        return

    state_path = root / "manifests/first_layer_mechanism_states.json"
    hidden_path = root / "manifests/hidden_states.json"
    candidate_path = root / "candidate_tokens/candidate_tokens.json"
    wrong_path = root / "controls/wrong_prefixes.jsonl"
    inputs = {
        "mechanism_states_sha256": file_sha256(state_path),
        "hidden_states_sha256": file_sha256(hidden_path),
        "candidate_tokens_sha256": file_sha256(candidate_path),
        "wrong_prefixes_sha256": file_sha256(wrong_path),
    }
    manifest_path = root / "manifests/first_layer_mechanism.json"
    if not args.force and stage_is_complete(manifest_path, config, inputs):
        print(manifest_path)
        return

    state = read_json(state_path)
    hidden = read_json(hidden_path)
    candidates = read_json(candidate_path)
    prefixes = read_jsonl(hidden["prefix_snapshot"])
    wrong_map = {row["prefix_id"]: row["wrong_prefix_ids"] for row in read_jsonl(wrong_path)}
    site_arrays = {entry["site"]: np.load(entry["path"], mmap_mode="r") for entry in state["sites"]}
    value_paths = {int(row["prefix_index"]): row["path"] for row in state["value_output_bases"]}
    value_bases = {index: np.load(path) for index, path in value_paths.items()}
    rank = int(settings.get("value_space_rank", 64))
    auxiliary = np.asarray([i for i, row in enumerate(prefixes) if row["problem_group"] == "auxiliary"], dtype=np.int64)
    nonauxiliary = np.asarray([i for i, row in enumerate(prefixes) if row["problem_group"] != "auxiliary"], dtype=np.int64)
    evaluation = np.asarray([i for i, row in enumerate(prefixes) if row["problem_group"] == "analysis_test"], dtype=np.int64)

    rows = []
    energy = []
    for site, z in site_arrays.items():
        energy.append({"site": site, **_interaction_energy(z, auxiliary, evaluation, candidates["analysis_indices"])})
        for fold in candidates["folds"]:
            train, heldout = center_train_and_evaluation(
                z[nonauxiliary], z[auxiliary], fold["train_indices"], fold["evaluation_indices"]
            )
            local, conditional = _fit_controls(train.residuals, prefixes, nonauxiliary, rank)
            for local_axis, full_index in enumerate(nonauxiliary):
                prefix = prefixes[int(full_index)]
                if prefix["problem_group"] != "analysis_test":
                    continue
                target = heldout.residuals[local_axis]
                local_basis = local.get(prefix["prefix_id"])
                global_basis, global_exact, global_distance, _ = _resolve_conditional_basis(conditional, prefix)
                wrong_bases = [local.get(wrong_id) for wrong_id in wrong_map.get(prefix["prefix_id"], [])]
                wrong_bases = [basis for basis in wrong_bases if basis is not None]
                value_basis = value_bases.get(int(full_index))
                ev_local, ev_global, delta_global, effective_global = _paired_ev(target, local_basis, global_basis)
                wrong_deltas = []
                for wrong_basis in wrong_bases:
                    _local, _wrong, delta, _effective = _paired_ev(target, local_basis, wrong_basis)
                    if np.isfinite(delta):
                        wrong_deltas.append(delta)
                ev_local_value, ev_value, delta_value, effective_value = _paired_ev(target, local_basis, value_basis)
                rows.append({
                    "problem_id": prefix["problem_id"],
                    "prefix_id": prefix["prefix_id"],
                    "site": site,
                    "fold": int(fold["fold_id"]),
                    "rank": rank,
                    "ev_local": ev_local,
                    "ev_conditional_global": ev_global,
                    "delta_local_conditional_global": delta_global,
                    "conditional_global_exact_length_bin": bool(global_exact),
                    "conditional_global_length_bin_distance": global_distance,
                    "effective_global_rank": effective_global,
                    "delta_local_wrong": float(np.mean(wrong_deltas)) if wrong_deltas else float("nan"),
                    "wrong_prefix_count": len(wrong_deltas),
                    "ev_local_value_comparison": ev_local_value,
                    "ev_value_output_space": ev_value,
                    "delta_local_value_output": delta_value,
                    "effective_value_rank": effective_value,
                })
            print(f"[first_layer_mechanism_geometry] site={site} fold={int(fold['fold_id'])}", flush=True)

    bootstrap_args = {
        "replicates": int(config["statistics"]["bootstrap_replicates"]),
        "seed": int(config["seed"]) + 2203,
        "ci": float(config["statistics"]["ci"]),
    }
    summary = {
        "sites": {},
        "interaction_energy": {row["site"]: {key: value for key, value in row.items() if key != "site"} for row in energy},
        "value_output_definition": state["value_output_basis_definition"],
        "post_mlp_consistency_max_abs_difference": state["post_mlp_consistency_max_abs_difference"],
    }
    for site in site_arrays:
        selected = [row for row in rows if row["site"] == site]
        site_summary = {}
        for metric in ("ev_local", "delta_local_conditional_global", "delta_local_wrong", "delta_local_value_output"):
            finite = [row for row in selected if np.isfinite(row[metric])]
            site_summary[metric] = problem_bootstrap(
                np.asarray([row[metric] for row in finite]),
                np.asarray([row["problem_id"] for row in finite]),
                **bootstrap_args,
            )
        site_summary["coverage"] = {"rows": len(selected), "expected_rows": int(config["data"]["evaluation_prefixes"]) * int(config["candidates"]["folds"])}
        summary["sites"][site] = site_summary
    pre_fraction = float(summary["interaction_energy"]["pre_attention"]["interaction_fraction_eta"])
    post_attention_fraction = float(summary["interaction_energy"]["post_attention"]["interaction_fraction_eta"])
    post_mlp_fraction = float(summary["interaction_energy"]["post_mlp"]["interaction_fraction_eta"])
    summary["interaction_fraction_diagnostics"] = {
        "pre_attention": pre_fraction,
        "post_attention": post_attention_fraction,
        "post_mlp": post_mlp_fraction,
        "post_attention_minus_pre": post_attention_fraction - pre_fraction,
        "post_mlp_minus_post_attention": post_mlp_fraction - post_attention_fraction,
    }
    summary["mechanism_interpretation_inputs"] = {
        "value_matches_post_attention": abs(float(summary["sites"]["post_attention"]["delta_local_value_output"]["mean"])),
        "local_minus_value_post_mlp": float(summary["sites"]["post_mlp"]["delta_local_value_output"]["mean"]),
        "note": "No equivalence margin is imposed; use confidence intervals and the separately computed functional comparison.",
    }
    rows_path = root / "metrics/first_layer_mechanism_rows.csv"
    energy_path = root / "metrics/first_layer_mechanism_energy.csv"
    summary_path = root / "metrics/first_layer_mechanism_summary.json"
    _write_csv(rows_path, rows)
    _write_csv(energy_path, energy)
    atomic_json(summary_path, summary)
    atomic_json(manifest_path, {
        "complete": True,
        "config_hash": stable_hash(config),
        **inputs,
        "rows": str(rows_path),
        "energy": str(energy_path),
        "summary": str(summary_path),
    })
    print(manifest_path)


if __name__ == "__main__":
    main()
