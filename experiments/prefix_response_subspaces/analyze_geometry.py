from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import numpy as np

from .src.permutation import exchangeability_diagnostics, permutation_space_size, permute_prefix_labels_by_token, stratification_labels
from .src.statistics import permutation_pvalue, problem_bootstrap
from .src.subspaces import RankError, content_subspace, explained_variance, remove_directions, top_svd
from .src.utils import atomic_json, ensure_layout, file_sha256, load_config, read_json, read_jsonl, stable_hash, stage_is_complete


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: path.write_text("", encoding="utf-8"); return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def _safe_basis(samples, rank):
    effective = min(rank, samples.shape[0], samples.shape[1], int(np.linalg.matrix_rank(samples)))
    return (top_svd(samples, effective), effective) if effective > 0 else (None, 0)


def candidate_vocabulary_ranks(logits, candidate_ids, tokenizer_vocabulary_size):
    values = np.asarray(logits, dtype=np.float32)
    candidates = np.asarray(candidate_ids, dtype=np.int64)
    vocabulary_size = int(tokenizer_vocabulary_size)
    if values.ndim != 2 or not 0 < vocabulary_size <= values.shape[1]:
        raise ValueError("invalid logits shape or tokenizer vocabulary size")
    if candidates.ndim != 1 or np.any(candidates < 0) or np.any(candidates >= vocabulary_size):
        raise ValueError("candidate token ID lies outside tokenizer vocabulary")
    vocabulary_logits = values[:, :vocabulary_size]
    # Sort once per prefix rather than scanning the full vocabulary once per
    # candidate. Exact ties are deterministically ordered by token ID; model
    # logits do not normally tie outside masked/reserved rows, which are absent
    # from the tokenizer vocabulary slice.
    order = np.argsort(-vocabulary_logits, axis=1, kind="stable")
    inverse = np.empty(order.shape, dtype=np.int32)
    rank_values = np.arange(1, vocabulary_size + 1, dtype=np.int32)
    for row in range(len(values)):
        inverse[row, order[row]] = rank_values
    return inverse[:, candidates]


def _average_ranks(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def continuous_relationship(x, y):
    left, right = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    left, right = left[valid], right[valid]
    result = {"n_prefixes": int(len(left)), "pearson": float("nan"), "spearman": float("nan"), "ols_slope_delta_per_normalized_js": float("nan"), "ols_intercept": float("nan")}
    if len(left) < 2 or float(np.std(left)) <= 0 or float(np.std(right)) <= 0:
        return result
    result["pearson"] = float(np.corrcoef(left, right)[0, 1])
    result["spearman"] = float(np.corrcoef(_average_ranks(left), _average_ranks(right))[0, 1])
    slope = float(np.sum((left-left.mean())*(right-right.mean())) / np.sum((left-left.mean())**2))
    result["ols_slope_delta_per_normalized_js"] = slope
    result["ols_intercept"] = float(right.mean() - slope*left.mean())
    return result


def analyze_fold(train_r, eval_r, prefixes, nonaux_indices, matches, rank, content_paths, global_train_mask, *, excluded_content_positions=None, outlier=None, evaluation_candidate_indices=None, candidate_rank_by_prefix=None, high_top_ks=(), high_minimum=1):
    local_bases, local_ranks = {}, {}
    for local_index, full_index in enumerate(nonaux_indices):
        basis, effective = _safe_basis(train_r[local_index], rank)
        local_bases[prefixes[int(full_index)]["prefix_id"]] = basis; local_ranks[prefixes[int(full_index)]["prefix_id"]] = effective
    global_basis, global_rank = _safe_basis(train_r[global_train_mask].reshape(-1, train_r.shape[-1]), rank)
    outlier_local, outlier_global = {}, None
    if outlier is not None:
        transformed_train = remove_directions(train_r, outlier)
        outlier_global, _ = _safe_basis(transformed_train[global_train_mask].reshape(-1, transformed_train.shape[-1]), rank)
        for local_index, full_index in enumerate(nonaux_indices):
            outlier_local[prefixes[int(full_index)]["prefix_id"]], _ = _safe_basis(transformed_train[local_index], rank)
    rows = []
    match_by_id = {row["prefix_id"]: row for row in matches if row["split"] == "evaluation"}
    for local_index, full_index in enumerate(nonaux_indices):
        prefix = prefixes[int(full_index)]
        if prefix["problem_group"] != "analysis_test": continue
        prefix_id = prefix["prefix_id"]; local = local_bases[prefix_id]; match_info = match_by_id[prefix_id]; matched_id = match_info.get("matched_prefix_id")
        matched = local_bases.get(matched_id)
        if local is None or global_basis is None: continue
        content = np.load(content_paths[int(full_index)])
        try: content_basis, _ = content_subspace(content, rank, excluded_positions=set(excluded_content_positions or ()))
        except RankError: continue
        effective = min(local.shape[1], global_basis.shape[1], content_basis.shape[1], matched.shape[1] if matched is not None else rank)
        target = eval_r[local_index]
        values = {"local": explained_variance(target, local[:, :effective]), "global": explained_variance(target, global_basis[:, :effective]), "matched": explained_variance(target, matched[:, :effective]) if matched is not None else float("nan"), "content": explained_variance(target, content_basis[:, :effective])}
        row = {"problem_id": prefix["problem_id"], "prefix_id": prefix_id, "matched_prefix_id": matched_id, "match_available": bool(match_info.get("matched", False)), "match_js_distance": float(match_info["js_distance"]) if match_info.get("matched") else float("nan"), "match_normalized_js_distance": float(match_info["normalized_js_distance"]) if match_info.get("matched") else float("nan"), "top5_overlap": int(match_info.get("top5_overlap", 0)), "top20_overlap": int(match_info.get("top20_overlap", 0)), "good_match": match_info["good_match"], "effective_rank": effective, **{f"ev_{k}": v for k, v in values.items()}, "delta_global": values["local"]-values["global"], "delta_matched": values["local"]-values["matched"], "delta_content": values["local"]-values["content"]}
        if outlier is not None and outlier_global is not None and outlier_local.get(prefix_id) is not None and outlier_local.get(matched_id) is not None:
            transformed_target = remove_directions(target, outlier); transformed_content = remove_directions(content, outlier)
            transformed_content_basis, _ = content_subspace(transformed_content, rank, excluded_positions=set(excluded_content_positions or ()))
            effective_out = min(outlier_local[prefix_id].shape[1], transformed_content_basis.shape[1])
            row["delta_content_outlier_removed"] = explained_variance(transformed_target, outlier_local[prefix_id][:, :effective_out]) - explained_variance(transformed_target, transformed_content_basis[:, :effective_out])
        else: row["delta_content_outlier_removed"] = float("nan")
        ranks = (candidate_rank_by_prefix or {}).get(prefix_id)
        for top_k in map(int, high_top_ks):
            suffix = f"top{top_k}"
            mask = None if ranks is None or evaluation_candidate_indices is None else ranks[np.asarray(evaluation_candidate_indices)] <= top_k
            count = int(mask.sum()) if mask is not None else 0
            row[f"{suffix}_token_count"] = count
            if mask is not None and count >= high_minimum:
                high_target = target[mask]
                high_local = explained_variance(high_target, local[:, :effective])
                for name, basis in (("global", global_basis), ("matched", matched), ("content", content_basis)):
                    row[f"delta_{name}_{suffix}"] = high_local - explained_variance(high_target, basis[:, :effective]) if basis is not None else float("nan")
            else:
                for name in ("global", "matched", "content"):
                    row[f"delta_{name}_{suffix}"] = float("nan")
        rows.append(row)
    return rows, local_bases, global_basis


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); args = parser.parse_args()
    config = load_config(args.config); root = ensure_layout(config)
    residual_manifest_path = root / "manifests/residuals.json"; hidden_manifest_path = root / "manifests/hidden_states.json"; match_path = root / "matches/prefix_matches.jsonl"
    inputs = {"residuals_sha256": file_sha256(residual_manifest_path), "hidden_states_sha256": file_sha256(hidden_manifest_path), "matches_sha256": file_sha256(match_path)}
    manifest_path = root / "manifests/geometry.json"
    if stage_is_complete(manifest_path, config, inputs): print(manifest_path); return
    residual_manifest, hidden_manifest = read_json(residual_manifest_path), read_json(hidden_manifest_path)
    prefixes = read_jsonl(hidden_manifest["prefix_snapshot"]); matches = read_jsonl(match_path); rank = int(config["analysis"]["primary_rank"]); primary_layer = int(config["analysis"]["primary_layer"])
    candidate_manifest = read_json(root / "candidate_tokens/candidate_tokens.json"); candidate_ids = np.asarray(candidate_manifest["candidate_token_ids"], dtype=np.int64)
    candidate_stage_manifest = read_json(root / "manifests/candidate_tokens.json")
    full_prefixes = read_jsonl(root / "prefix_pool/prefixes.jsonl"); full_logits = np.load(root / "prefix_pool/next_token_logits.npy", mmap_mode="r")
    needed_prefix_ids = {row["prefix_id"] for row in prefixes}
    needed_full_indices = [i for i, row in enumerate(full_prefixes) if row["prefix_id"] in needed_prefix_ids]
    rank_by_prefix = {}
    for start in range(0, len(needed_full_indices), 8):
        indices = needed_full_indices[start:start+8]
        logits_chunk = np.asarray(full_logits[indices], dtype=np.float32)
        rank_chunk = candidate_vocabulary_ranks(
            logits_chunk,
            candidate_ids,
            int(candidate_stage_manifest["tokenizer_vocabulary_size"]),
        )
        for local_index, full_index in enumerate(indices):
            rank_by_prefix[full_prefixes[full_index]["prefix_id"]] = rank_chunk[local_index]
    high_top_ks = tuple(map(int, config["analysis"]["high_probability_top_ks"]))
    primary_top_k = int(config["analysis"]["high_probability_primary_top_k"])
    if primary_top_k not in high_top_ks:
        raise ValueError("analysis.high_probability_primary_top_k must occur in high_probability_top_ks")
    excluded_content_positions = set(map(int, config["content"].get("attention_sink_positions", [])))
    if bool(config["content"].get("exclude_bos", True)): excluded_content_positions.add(0)
    all_rows, null_deltas, basis_entries, exchangeability_entries = [], defaultdict(list), [], []; rng = np.random.default_rng(int(config["seed"]) + 991)
    for entry in residual_manifest["entries"]:
        bundle = np.load(entry["path"]); train_r = bundle["train_residuals"]; eval_r = bundle["evaluation_residuals"]; nonaux = bundle["nonauxiliary_prefix_indices"]
        global_mask = np.asarray([prefixes[int(i)]["problem_group"] == "analysis_train" for i in nonaux])
        content_paths = [root / f"hidden_states/content/layer_{entry['layer']}_prefix_{i:05d}.npy" for i in range(len(prefixes))]
        hidden_entry = next(item for item in hidden_manifest["layers"] if int(item["layer"]) == int(entry["layer"])); raw_z = np.asarray(np.load(hidden_entry["successor_path"], mmap_mode="r"), dtype=np.float32)
        auxiliary_indices = np.asarray([i for i, row in enumerate(prefixes) if row["problem_group"] == "auxiliary"]); auxiliary_flat = raw_z[auxiliary_indices].reshape(-1, raw_z.shape[-1]); auxiliary_flat -= auxiliary_flat.mean(axis=0, keepdims=True)
        outlier_count = min(int(config["analysis"]["outlier_directions"]), int(np.linalg.matrix_rank(auxiliary_flat))); outlier = top_svd(auxiliary_flat, outlier_count) if outlier_count else None
        local_bases = {}; global_basis = None
        for descriptive_rank in config["analysis"]["ranks"]:
            try:
                rows, fitted_local, fitted_global = analyze_fold(train_r, eval_r, prefixes, nonaux, matches, int(descriptive_rank), content_paths, global_mask, excluded_content_positions=excluded_content_positions, outlier=outlier if int(descriptive_rank) == rank else None, evaluation_candidate_indices=bundle["evaluation_candidate_indices"] if int(descriptive_rank) == rank else None, candidate_rank_by_prefix=rank_by_prefix if int(descriptive_rank) == rank else None, high_top_ks=high_top_ks, high_minimum=int(config["analysis"]["high_probability_min_tokens"]))
            except RankError:
                continue
            for row in rows: row.update({"layer": entry["layer"], "fold": entry["fold"], "rank": int(descriptive_rank)})
            all_rows.extend(rows)
            if int(descriptive_rank) == rank: local_bases, global_basis = fitted_local, fitted_global
        basis_dir = root / f"subspaces/layer_{entry['layer']}_fold_{entry['fold']}"; basis_dir.mkdir(parents=True, exist_ok=True)
        if global_basis is not None: np.save(basis_dir / "global.npy", global_basis.astype(np.float32))
        local_index_manifest = {}
        for basis_index, (prefix_id, basis) in enumerate(sorted(local_bases.items())):
            if basis is None: continue
            filename = f"local_{basis_index:05d}.npy"; np.save(basis_dir / filename, basis.astype(np.float32)); local_index_manifest[prefix_id] = filename
        atomic_json(basis_dir / "index.json", {"layer": entry["layer"], "fold": entry["fold"], "rank": rank, "global": "global.npy" if global_basis is not None else None, "local": local_index_manifest})
        basis_entries.append({"layer": entry["layer"], "fold": entry["fold"], "directory": str(basis_dir), "index": str(basis_dir / "index.json")})
        # Full token-wise stratified permutation and pseudoprefix recentering.
        labels = stratification_labels(np.asarray([prefixes[int(i)]["prefix_length_bin"] for i in nonaux]), np.asarray([prefixes[int(i)]["reasoning_progress_bin"] for i in nonaux]))
        minimum_stratum_size = int(config["permutation"]["minimum_stratum_size"])
        exchangeability_entry = {"layer": int(entry["layer"]), "fold": int(entry["fold"]), "train_token_count": int(train_r.shape[1]), "evaluation_token_count": int(eval_r.shape[1]), **exchangeability_diagnostics(labels, minimum_stratum_size), **permutation_space_size(labels, int(train_r.shape[1] + eval_r.shape[1]))}
        exchangeability_entries.append(exchangeability_entry)
        test_local = np.asarray([j for j, i in enumerate(nonaux) if prefixes[int(i)]["problem_group"] == "analysis_test"], dtype=np.int64)
        local_by_prefix = {prefixes[int(full)]["prefix_id"]: local for local, full in enumerate(nonaux)}
        evaluation_matches = {row["prefix_id"]: row["matched_prefix_id"] for row in matches if row["split"] == "evaluation"}
        train_moved_counts, eval_moved_counts, train_assignment_counts, eval_assignment_counts = [], [], [], []
        train_plan_hashes, eval_plan_hashes, joint_plan_hashes = set(), set(), set()
        for permutation_index in range(int(config["permutation"]["replicates"])):
            p_train, train_plan = permute_prefix_labels_by_token(train_r, labels, rng, return_diagnostics=True); p_eval, eval_plan = permute_prefix_labels_by_token(eval_r, labels, rng, return_diagnostics=True)
            train_moved_counts.append(train_plan["actual_moved_prefix_count"]); eval_moved_counts.append(eval_plan["actual_moved_prefix_count"])
            train_assignment_counts.append(train_plan["actual_moved_assignment_count"]); eval_assignment_counts.append(eval_plan["actual_moved_assignment_count"])
            train_plan_hashes.add(train_plan["plan_sha256"]); eval_plan_hashes.add(eval_plan["plan_sha256"]); joint_plan_hashes.add((train_plan["plan_sha256"], eval_plan["plan_sha256"]))
            for local_index in test_local:
                basis, _ = _safe_basis(p_train[local_index], rank)
                global_null, _ = _safe_basis(p_train[global_mask].reshape(-1, p_train.shape[-1]), rank)
                if basis is not None and global_null is not None:
                    target = p_eval[local_index]; prefix_id = prefixes[int(nonaux[local_index])]["prefix_id"]
                    local_ev = explained_variance(target, basis)
                    global_null_ev = explained_variance(target, global_null)
                    null_deltas[("delta_global", entry["layer"], entry["fold"], permutation_index)].append(local_ev - global_null_ev)
                    null_deltas[("global_ev", entry["layer"], entry["fold"], permutation_index)].append(global_null_ev)
                    null_deltas[("same_prefix_ev", entry["layer"], entry["fold"], permutation_index)].append(local_ev)
                    matched_local = local_by_prefix.get(evaluation_matches.get(prefix_id, ""))
                    if matched_local is not None:
                        matched_null, _ = _safe_basis(p_train[matched_local], rank)
                        if matched_null is not None: null_deltas[("delta_matched", entry["layer"], entry["fold"], permutation_index)].append(local_ev - explained_variance(target, matched_null))
                    try:
                        content_basis, _ = content_subspace(np.load(content_paths[int(nonaux[local_index])]), rank, excluded_positions=excluded_content_positions)
                        effective_null = min(basis.shape[1], content_basis.shape[1]); null_deltas[("delta_content", entry["layer"], entry["fold"], permutation_index)].append(explained_variance(target, basis[:, :effective_null]) - explained_variance(target, content_basis[:, :effective_null]))
                    except RankError:
                        pass
        exchangeability_entry.update({
            "sampled_permutation_replicates": int(config["permutation"]["replicates"]),
            "unique_sampled_train_plans": len(train_plan_hashes),
            "unique_sampled_evaluation_plans": len(eval_plan_hashes),
            "unique_sampled_joint_plans": len(joint_plan_hashes),
            "actual_moved_prefix_count_train": {"min": int(min(train_moved_counts)), "median": float(np.median(train_moved_counts)), "max": int(max(train_moved_counts))},
            "actual_moved_prefix_count_evaluation": {"min": int(min(eval_moved_counts)), "median": float(np.median(eval_moved_counts)), "max": int(max(eval_moved_counts))},
            "actual_moved_assignment_count_train": {"min": int(min(train_assignment_counts)), "median": float(np.median(train_assignment_counts)), "max": int(max(train_assignment_counts))},
            "actual_moved_assignment_count_evaluation": {"min": int(min(eval_assignment_counts)), "median": float(np.median(eval_assignment_counts)), "max": int(max(eval_assignment_counts))},
        })
    rows_path = root / "metrics/geometry_rows.csv"; _write_csv(rows_path, all_rows)
    summaries = {}
    primary_rows = [row for row in all_rows if int(row["rank"]) == rank and int(row["layer"]) == primary_layer]
    relationship_by_prefix = {}
    for row in primary_rows:
        group = relationship_by_prefix.setdefault(row["prefix_id"], {"problem_id": row["problem_id"], "prefix_id": row["prefix_id"], "matched_prefix_id": row["matched_prefix_id"], "match_js_distance": row["match_js_distance"], "match_normalized_js_distance": row["match_normalized_js_distance"], "good_match": row["good_match"], "delta_matched_by_fold": []})
        group["delta_matched_by_fold"].append(float(row["delta_matched"]))
    relationship_rows = []
    for group in relationship_by_prefix.values():
        deltas = group.pop("delta_matched_by_fold")
        relationship_rows.append({**group, "delta_matched_mean_across_folds": float(np.mean(deltas)), "delta_matched_std_across_folds": float(np.std(deltas)), "n_folds": len(deltas)})
    relationship_rows.sort(key=lambda row: row["prefix_id"])
    relationship_path = root / "metrics/matched_js_delta_relationship.csv"
    _write_csv(relationship_path, relationship_rows)
    for metric in ("delta_global", "delta_matched", "delta_content"):
        values = np.asarray([row[metric] for row in primary_rows]); ids = np.asarray([row["problem_id"] for row in primary_rows]); summaries[metric] = problem_bootstrap(values, ids, replicates=int(config["statistics"]["bootstrap_replicates"]), seed=int(config["seed"]), ci=float(config["statistics"]["ci"]))
    good = [row for row in primary_rows if row["good_match"]]
    summaries["delta_matched_good_matches"] = problem_bootstrap(np.asarray([row["delta_matched"] for row in good]), np.asarray([row["problem_id"] for row in good]), replicates=int(config["statistics"]["bootstrap_replicates"]), seed=int(config["seed"])+1, ci=float(config["statistics"]["ci"]))
    summaries["delta_content_outlier_removed"] = problem_bootstrap(np.asarray([row["delta_content_outlier_removed"] for row in primary_rows]), np.asarray([row["problem_id"] for row in primary_rows]), replicates=int(config["statistics"]["bootstrap_replicates"]), seed=int(config["seed"])+2, ci=float(config["statistics"]["ci"]))
    summaries["matched_js_delta_continuous_relationship"] = continuous_relationship([row["match_normalized_js_distance"] for row in relationship_rows], [row["delta_matched_mean_across_folds"] for row in relationship_rows]) | {"x": "D_NJS = D_JS / log(2)", "y": "mean Delta_matched across token folds", "row_artifact": str(relationship_path)}
    for top_k in high_top_ks:
        for metric in ("delta_global", "delta_matched", "delta_content"):
            key = f"{metric}_top{top_k}"
            summaries[key] = problem_bootstrap(np.asarray([row[key] for row in primary_rows]), np.asarray([row["problem_id"] for row in primary_rows]), replicates=int(config["statistics"]["bootstrap_replicates"]), seed=int(config["seed"])+3+top_k, ci=float(config["statistics"]["ci"]))
        summaries[f"top{top_k}_token_count"] = {
            "total": int(sum(row[f"top{top_k}_token_count"] for row in primary_rows)),
            "prefix_fold_rows_meeting_minimum": int(sum(row[f"top{top_k}_token_count"] >= int(config["analysis"]["high_probability_min_tokens"]) for row in primary_rows)),
        }
    summaries["high_probability_definition"] = {
        "type": "full_tokenizer_vocabulary_rank",
        "top_ks": list(high_top_ks),
        "primary_top_k": primary_top_k,
        "minimum_tokens_per_prefix_fold": int(config["analysis"]["high_probability_min_tokens"]),
    }
    null_by_metric = {}
    for metric in ("delta_global", "delta_matched", "delta_content", "same_prefix_ev", "global_ev"):
        replicate_means = []
        for permutation_index in range(int(config["permutation"]["replicates"])):
            values = [x for key, group in null_deltas.items() if key[0] == metric and int(key[1]) == primary_layer and int(key[3]) == permutation_index for x in group]
            if values: replicate_means.append(float(np.mean(values)))
        null_by_metric[metric] = np.asarray(replicate_means)
    primary_exchangeability = [row for row in exchangeability_entries if int(row["layer"]) == primary_layer]
    minimum_exchangeable_fraction = min((row["exchangeable_prefix_fraction"] for row in primary_exchangeability), default=0.0)
    required_exchangeable_fraction = float(config["permutation"]["minimum_exchangeable_prefix_fraction"])
    permutation_inference_valid = minimum_exchangeable_fraction >= required_exchangeable_fraction
    summaries["permutation_exchangeability"] = {"primary_layer": primary_layer, "folds": primary_exchangeability, "minimum_exchangeable_prefix_fraction": minimum_exchangeable_fraction, "required_exchangeable_prefix_fraction": required_exchangeable_fraction, "permutation_inference_valid": permutation_inference_valid}
    for metric in ("delta_global", "delta_matched", "delta_content"):
        raw_p = permutation_pvalue(summaries[metric]["mean"], null_by_metric[metric])
        summaries[f"{metric}_permutation_p_raw"] = raw_p
        summaries[f"{metric}_permutation_p"] = raw_p if permutation_inference_valid else float("nan")
    summaries["permutation_global_ev_diagnostic"] = {"observed_mean": float(np.mean([row["ev_global"] for row in primary_rows])), "null_mean": float(np.mean(null_by_metric["global_ev"])), "absolute_difference": float(abs(np.mean([row["ev_global"] for row in primary_rows])-np.mean(null_by_metric["global_ev"]))) }
    summary_path = root / "metrics/geometry_summary.json"; atomic_json(summary_path, summaries)
    atomic_json(root / "permutation/null_summary.json", {f"{metric}_null": values.tolist() for metric, values in null_by_metric.items()} | {"recentered": True, "replicates_per_layer_fold": int(config["permutation"]["replicates"]), "exchangeability": exchangeability_entries, "permutation_inference_valid": permutation_inference_valid})
    atomic_json(manifest_path, {"complete": True, "config_hash": stable_hash(config), **inputs, "rows": str(rows_path), "rows_sha256": file_sha256(rows_path), "matched_js_delta_relationship": str(relationship_path), "matched_js_delta_relationship_sha256": file_sha256(relationship_path), "summary": str(summary_path), "summary_sha256": file_sha256(summary_path), "primary_rank": rank, "primary_layer": primary_layer, "subspace_entries": basis_entries, "permutation_recentered": True})
    print(manifest_path)


if __name__ == "__main__": main()
