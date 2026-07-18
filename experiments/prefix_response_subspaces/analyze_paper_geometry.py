from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from .analyze_geometry import candidate_vocabulary_ranks
from .src.permutation import exchangeability_diagnostics, permutation_space_size, permute_prefix_labels_by_token, stratification_labels
from .src.residualization import double_center
from .src.statistics import permutation_pvalue, problem_bootstrap
from .src.storage import load_residual_entry
from .src.subspaces import RankError, content_subspace, explained_variance, top_svd
from .src.utils import atomic_json, ensure_layout, file_sha256, load_config, read_json, read_jsonl, stable_hash, stage_is_complete


_COMPACT_BASIS_STORAGE = False


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: path.write_text("", encoding="utf-8"); return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys: keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def _basis(samples, rank):
    matrix = np.asarray(samples, dtype=np.float64)
    try:
        basis = top_svd(matrix, int(rank), allow_rank_reduction=True)
        # The 3B replication keeps several thousand 3072x64 bases during the
        # split-half audit. SVD stays float64; only completed bases are compact.
        return basis.astype(np.float32) if _COMPACT_BASIS_STORAGE else basis
    except RankError:
        return None


def _equal_energy_global_basis(prefix_samples, rank, eps=1e-12):
    """Top eigenspace of mean_i K_i / tr(K_i), without forming hidden^2 K."""
    blocks = []
    for samples in np.asarray(prefix_samples, dtype=np.float64):
        energy = float(np.square(samples).sum())
        if energy > eps:
            blocks.append(samples / np.sqrt(energy))
    if not blocks:
        return None
    # stack.T @ stack is sum_i K_i / tr(K_i).  The omitted 1/N factor does
    # not change eigenvectors.
    return _basis(np.concatenate(blocks, axis=0), rank)


def _projection_rotation_distance(left, right):
    """Squared normalized projection distance: 1 - ||U'V||_F^2 / r."""
    if left is None or right is None:
        return float("nan")
    rank = min(int(left.shape[1]), int(right.shape[1]))
    if rank <= 0:
        return float("nan")
    overlap = float(np.square(left[:, :rank].T @ right[:, :rank]).sum())
    return float(np.clip(1.0 - overlap / rank, 0.0, 1.0))


def _stratum(row):
    return (int(row["prefix_length_bin"]), int(row["reasoning_progress_bin"]))


def _resolve_conditional_basis(conditional, prefix):
    """Resolve an exact conditioned Global, or a recorded same-progress fallback."""
    requested = _stratum(prefix)
    exact = conditional.get(requested)
    if exact is not None:
        return exact, True, 0, requested
    length_bin, progress_bin = requested
    candidates = [
        (abs(int(key[0]) - length_bin), int(key[0]), key, basis)
        for key, basis in conditional.items()
        if int(key[1]) == progress_bin and basis is not None
    ]
    if not candidates:
        return None, False, None, None
    distance, _length, key, basis = min(candidates, key=lambda item: (item[0], item[1]))
    return basis, False, int(distance), key


def _candidate_rank_map(root, prefixes, candidate_ids):
    logits = np.load(root / "prefix_pool/next_token_logits.npy", mmap_mode="r")
    candidate_stage = read_json(root / "manifests/candidate_tokens.json")
    axis_ids = candidate_stage.get("prefix_axis_ids")
    if axis_ids is None:
        axis_ids = [row["prefix_id"] for row in read_jsonl(root / "prefix_pool/prefixes.jsonl")]
    full_index = {prefix_id:index for index,prefix_id in enumerate(axis_ids)}
    result = {}
    needed = [(row["prefix_id"], full_index[row["prefix_id"]]) for row in prefixes if row["problem_group"] in {"analysis_dev","analysis_test"}]
    for start in range(0, len(needed), 8):
        chunk = needed[start:start+8]
        ranks = candidate_vocabulary_ranks(np.asarray(logits[[item[1] for item in chunk]], dtype=np.float32), candidate_ids, int(candidate_stage["tokenizer_vocabulary_size"]))
        for axis, (prefix_id, _index) in enumerate(chunk): result[prefix_id] = ranks[axis]
    return result


def _fit_controls(train_r, prefixes, nonaux, rank, required_local_ids=None):
    required = set(required_local_ids) if required_local_ids is not None else None
    local = {prefixes[int(full)]["prefix_id"]: _basis(train_r[index], rank) for index, full in enumerate(nonaux) if required is None or prefixes[int(full)]["prefix_id"] in required}
    train_positions = [index for index, full in enumerate(nonaux) if prefixes[int(full)]["problem_group"] == "analysis_train"]
    conditional = {}
    for position in train_positions:
        conditional.setdefault(_stratum(prefixes[int(nonaux[position])]), []).append(position)
    conditional = {key: _equal_energy_global_basis(train_r[positions], rank) for key, positions in conditional.items()}
    return local, conditional


def _pooled_top_k_rows(residual_manifest, selected_layer, selected_rank, prefixes, wrong_map, relaxed_wrong_targets, rank_map, top_ks):
    """Pool held-out projection energy across folds before applying Top-k eligibility.

    Every candidate token is still evaluated only in its held-out fold.  Pooling
    changes only the statistical unit for the minimum-token audit: it is one
    prefix across the complete cross-validated candidate set, not one quarter of
    that set.
    """
    accumulators = {}
    for entry in residual_manifest["entries"]:
        if int(entry["layer"]) != int(selected_layer):
            continue
        bundle = load_residual_entry(entry)
        train_r = bundle["train_residuals"]
        eval_r = bundle["evaluation_residuals"]
        nonaux = bundle["nonauxiliary_prefix_indices"]
        evaluation_indices = np.asarray(bundle["evaluation_candidate_indices"], dtype=np.int64)
        test_ids = {prefixes[int(index)]["prefix_id"] for index in nonaux if prefixes[int(index)]["problem_group"] == "analysis_test"}
        required = test_ids | {wrong_id for prefix_id in test_ids for wrong_id in wrong_map.get(prefix_id, [])}
        local, conditional = _fit_controls(train_r, prefixes, nonaux, selected_rank, required)
        for local_index, full_index in enumerate(nonaux):
            prefix = prefixes[int(full_index)]
            if prefix["problem_group"] != "analysis_test":
                continue
            prefix_id = prefix["prefix_id"]
            local_basis = local.get(prefix_id)
            global_basis, global_exact, length_distance, resolved_stratum = _resolve_conditional_basis(conditional, prefix)
            wrong_bases = [(wrong_id, local.get(wrong_id)) for wrong_id in wrong_map.get(prefix_id, [])]
            wrong_bases = [(wrong_id, basis) for wrong_id, basis in wrong_bases if basis is not None]
            if local_basis is None or global_basis is None:
                continue
            state = accumulators.setdefault(prefix_id, {
                "problem_id": prefix["problem_id"], "prefix_id": prefix_id,
                "conditional_global_exact_length_bin": bool(global_exact),
                "conditional_global_length_bin_distance": int(length_distance),
                "conditional_global_requested_stratum": f"{_stratum(prefix)[0]}:{_stratum(prefix)[1]}",
                "conditional_global_resolved_stratum": f"{resolved_stratum[0]}:{resolved_stratum[1]}",
                "wrong_control_exact_length_bin": prefix_id not in relaxed_wrong_targets,
                "wrong_prefix_count": len(wrong_bases), "folds_seen": 0, "top": {},
            })
            state["folds_seen"] += 1
            target = eval_r[local_index]
            ranks = rank_map[prefix_id][evaluation_indices]
            for top_k in map(int, top_ks):
                mask = ranks <= top_k
                top = state["top"].setdefault(top_k, {
                    "token_count": 0, "global_denominator": 0.0,
                    "local_global_numerator": 0.0, "global_numerator": 0.0,
                    "wrong": {},
                })
                top["token_count"] += int(mask.sum())
                if not np.any(mask):
                    continue
                high = np.asarray(target[mask], dtype=np.float64)
                effective = min(local_basis.shape[1], global_basis.shape[1])
                denominator = float(np.square(high).sum())
                top["global_denominator"] += denominator
                top["local_global_numerator"] += float(np.square(high @ local_basis[:, :effective]).sum())
                top["global_numerator"] += float(np.square(high @ global_basis[:, :effective]).sum())
                for wrong_id, wrong_basis in wrong_bases:
                    effective_wrong = min(local_basis.shape[1], wrong_basis.shape[1])
                    wrong = top["wrong"].setdefault(wrong_id, {"denominator": 0.0, "local_numerator": 0.0, "wrong_numerator": 0.0})
                    wrong["denominator"] += denominator
                    wrong["local_numerator"] += float(np.square(high @ local_basis[:, :effective_wrong]).sum())
                    wrong["wrong_numerator"] += float(np.square(high @ wrong_basis[:, :effective_wrong]).sum())
    rows = []
    for prefix_id in sorted(accumulators):
        state = accumulators[prefix_id]
        row = {key: value for key, value in state.items() if key != "top"}
        for top_k in map(int, top_ks):
            top = state["top"].get(top_k, {})
            count = int(top.get("token_count", 0)); denominator = float(top.get("global_denominator", 0.0))
            row[f"top{top_k}_token_count"] = count
            if denominator > 0:
                row[f"delta_conditional_global_top{top_k}"] = float((top["local_global_numerator"] - top["global_numerator"]) / denominator)
            else:
                row[f"delta_conditional_global_top{top_k}"] = float("nan")
            wrong_deltas = []
            for values in top.get("wrong", {}).values():
                if values["denominator"] > 0:
                    wrong_deltas.append((values["local_numerator"] - values["wrong_numerator"]) / values["denominator"])
            row[f"delta_wrong_top{top_k}"] = float(np.mean(wrong_deltas)) if wrong_deltas else float("nan")
        rows.append(row)
    return rows


def _add_pooled_top_k_summary(summary, pooled_rows, top_ks, high_minimum, config):
    bootstrap = {
        "replicates": int(config["statistics"]["bootstrap_replicates"]),
        "seed": int(config["seed"]), "ci": float(config["statistics"]["ci"]),
    }
    expected = int(config["data"]["evaluation_prefixes"])
    summary["top_k_definition"] = {
        "vocabulary_rank": "full tokenizer vocabulary",
        "aggregation": "projection energies pooled across all held-out folds per prefix before division",
        "minimum_tokens_unit": "prefix across complete cross-validated candidate set",
    }
    summary["top_k_coverage"] = {}
    for top_k in map(int, top_ks):
        counts = np.asarray([int(row[f"top{top_k}_token_count"]) for row in pooled_rows], dtype=np.int64)
        eligible = [row for row in pooled_rows if int(row[f"top{top_k}_token_count"]) >= int(high_minimum)]
        exact_global = [row for row in eligible if bool(row["conditional_global_exact_length_bin"])]
        exact_wrong = [row for row in eligible if bool(row["wrong_control_exact_length_bin"])]
        exact_both = [row for row in eligible if bool(row["conditional_global_exact_length_bin"]) and bool(row["wrong_control_exact_length_bin"])]
        def estimate(metric, rows, seed_offset=0):
            return problem_bootstrap(
                np.asarray([row[metric] for row in rows]), np.asarray([row["problem_id"] for row in rows]),
                replicates=bootstrap["replicates"], seed=bootstrap["seed"] + seed_offset, ci=bootstrap["ci"],
            )
        global_metric = f"delta_conditional_global_top{top_k}"
        wrong_metric = f"delta_wrong_top{top_k}"
        summary[global_metric] = estimate(global_metric, exact_global)
        summary[f"{global_metric}_with_fallback"] = estimate(global_metric, eligible, 11)
        summary[wrong_metric] = estimate(wrong_metric, eligible, 29)
        summary[f"{wrong_metric}_exact_bin"] = estimate(wrong_metric, exact_wrong, 43)
        summary["top_k_coverage"][str(top_k)] = {
            "minimum_tokens_per_prefix_across_heldout_folds": int(high_minimum),
            "expected_prefixes": expected, "observed_prefixes": len(pooled_rows),
            "eligible_prefixes": len(eligible), "eligible_fraction": len(eligible) / max(1, expected),
            "eligible_exact_conditional_global_prefixes": len(exact_global),
            "eligible_exact_wrong_prefixes": len(exact_wrong),
            "eligible_exact_both_prefixes": len(exact_both),
            "eligible_exact_both_fraction": len(exact_both) / max(1, expected),
            "complete": len(pooled_rows) == expected and len(eligible) == expected,
            "token_count_distribution": {
                "min": int(np.min(counts)) if len(counts) else 0,
                "q25": float(np.quantile(counts, .25)) if len(counts) else float("nan"),
                "median": float(np.median(counts)) if len(counts) else float("nan"),
                "q75": float(np.quantile(counts, .75)) if len(counts) else float("nan"),
                "max": int(np.max(counts)) if len(counts) else 0,
            },
        }
        sensitivity = {}
        for minimum in sorted({1, 4, 8, int(high_minimum)}):
            selected = [row for row in pooled_rows if int(row[f"top{top_k}_token_count"]) >= minimum and bool(row["conditional_global_exact_length_bin"]) and bool(row["wrong_control_exact_length_bin"])]
            sensitivity[str(minimum)] = {
                "minimum_tokens_per_prefix": minimum,
                "eligible_exact_prefixes": len(selected),
                "eligible_exact_fraction": len(selected) / max(1, expected),
                "delta_conditional_global": estimate(global_metric, selected, 101 + minimum),
                "delta_wrong": estimate(wrong_metric, selected, 151 + minimum),
            }
        summary["top_k_coverage"][str(top_k)]["minimum_token_sensitivity"] = sensitivity
    return summary


def _truncate_basis(basis, rank):
    if basis is None:
        return None
    return basis[:, :min(int(rank), int(basis.shape[1]))]


def _rotation_rows_for_ranks(train_r, prefixes, nonaux, wrong_map, ranks, layer, fold, seed):
    """Direct rotation curves, sharing one maximum-rank SVD across all ranks."""
    ranks = sorted(set(map(int, ranks)))
    maximum_rank = max(ranks)
    target_ids = {
        prefixes[int(full_index)]["prefix_id"]
        for full_index in nonaux
        if prefixes[int(full_index)]["problem_group"] == "analysis_test"
    }
    required_local_ids = set(target_ids)
    for prefix_id in target_ids:
        required_local_ids.update(wrong_map.get(prefix_id, []))
    local_full, conditional_full = _fit_controls(
        train_r, prefixes, nonaux, maximum_rank,
        required_local_ids=required_local_ids,
    )
    count = int(train_r.shape[1])
    order = np.random.default_rng(int(seed) + 1009 * (int(fold) + 1)).permutation(count)
    first, second = np.sort(order[::2]), np.sort(order[1::2])
    split_a_full = {}
    split_b_full = {}
    for index, full_index in enumerate(nonaux):
        prefix_id = prefixes[int(full_index)]["prefix_id"]
        if prefix_id not in required_local_ids:
            continue
        split_a_full[prefix_id] = _basis(train_r[index, first], maximum_rank)
        split_b_full[prefix_id] = _basis(train_r[index, second], maximum_rank)
    rows = []
    for rank in ranks:
        local = {key: _truncate_basis(value, rank) for key, value in local_full.items()}
        conditional = {key: _truncate_basis(value, rank) for key, value in conditional_full.items()}
        split_a = {key: _truncate_basis(value, rank) for key, value in split_a_full.items()}
        split_b = {key: _truncate_basis(value, rank) for key, value in split_b_full.items()}
        for full_index in nonaux:
            prefix = prefixes[int(full_index)]
            if prefix["problem_group"] != "analysis_test":
                continue
            prefix_id = prefix["prefix_id"]
            wrong_ids = [
                wrong_id for wrong_id in wrong_map.get(prefix_id, [])
                if local.get(wrong_id) is not None and split_b.get(wrong_id) is not None
            ]
            direct_wrong = [_projection_rotation_distance(local[prefix_id], local[wrong_id]) for wrong_id in wrong_ids]
            between = [_projection_rotation_distance(split_a[prefix_id], split_b[wrong_id]) for wrong_id in wrong_ids]
            within = _projection_rotation_distance(split_a[prefix_id], split_b[prefix_id])
            between_mean = float(np.mean(between)) if between else float("nan")
            global_basis, global_exact, length_distance, resolved_stratum = _resolve_conditional_basis(conditional, prefix)
            rows.append({
                "problem_id": prefix["problem_id"], "prefix_id": prefix_id,
                "layer": int(layer), "fold": int(fold), "rank": int(rank),
                "split_a_tokens": int(len(first)), "split_b_tokens": int(len(second)),
                "split_half_effective_rank": min(int(split_a[prefix_id].shape[1]), int(split_b[prefix_id].shape[1])),
                "local_global_effective_rank": min(int(local[prefix_id].shape[1]),int(global_basis.shape[1])) if global_basis is not None else 0,
                "local_wrong_min_effective_rank": min([min(int(local[prefix_id].shape[1]),int(local[wrong_id].shape[1])) for wrong_id in wrong_ids],default=0),
                "between_min_effective_rank": min([min(int(split_a[prefix_id].shape[1]),int(split_b[wrong_id].shape[1])) for wrong_id in wrong_ids],default=0),
                "wrong_prefix_count": len(wrong_ids),
                "conditional_global_exact_length_bin": bool(global_exact),
                "conditional_global_length_bin_distance": length_distance,
                "conditional_global_resolved_stratum": f"{resolved_stratum[0]}:{resolved_stratum[1]}" if resolved_stratum is not None else "",
                "d_rotation_local_conditional_global": _projection_rotation_distance(local[prefix_id], global_basis),
                "d_rotation_local_wrong_mean": float(np.mean(direct_wrong)) if direct_wrong else float("nan"),
                "R_within": within, "R_between": between_mean,
                "R_between_minus_within": between_mean - within,
            })
    return rows


def _rotation_rows(train_r, prefixes, nonaux, wrong_map, rank, layer, fold, seed):
    return _rotation_rows_for_ranks(
        train_r, prefixes, nonaux, wrong_map, [rank], layer, fold, seed,
    )


def _evaluate_rows(train_r, eval_r, prefixes, nonaux, wrong_map, rank, layer, fold, split, evaluation_candidate_indices, rank_map, top_ks, high_minimum, content_paths=None, excluded_content_positions=None, fitted_controls=None, content_bases=None):
    local_bases, conditional_bases = fitted_controls if fitted_controls is not None else _fit_controls(train_r, prefixes, nonaux, rank)
    rows = []
    for local_index, full_index in enumerate(nonaux):
        prefix = prefixes[int(full_index)]
        if prefix["problem_group"] != split: continue
        prefix_id = prefix["prefix_id"]; local_full = local_bases.get(prefix_id); conditional_full, conditional_exact, length_distance, resolved_stratum = _resolve_conditional_basis(conditional_bases, prefix); target = eval_r[local_index]
        if local_full is None or conditional_full is None: continue
        local=local_full[:,:min(int(rank),local_full.shape[1])]; conditional=conditional_full[:,:min(int(rank),conditional_full.shape[1])]
        effective_global = min(local.shape[1], conditional.shape[1]); local_global_ev = explained_variance(target, local[:, :effective_global]); global_ev = explained_variance(target, conditional[:, :effective_global])
        wrong_ids = wrong_map.get(prefix_id, []); wrong_evs, local_wrong_evs = [], []
        for wrong_id in wrong_ids:
            wrong_full = local_bases.get(wrong_id)
            if wrong_full is None: continue
            wrong=wrong_full[:,:min(int(rank),wrong_full.shape[1])]
            effective = min(local.shape[1], wrong.shape[1]); local_wrong_evs.append(explained_variance(target, local[:, :effective])); wrong_evs.append(explained_variance(target, wrong[:, :effective]))
        delta_wrong = float(np.mean(np.asarray(local_wrong_evs)-np.asarray(wrong_evs))) if wrong_evs else float("nan")
        row = {"problem_id": prefix["problem_id"], "prefix_id": prefix_id, "split": "development" if split == "analysis_dev" else "evaluation", "layer": int(layer), "fold": int(fold), "rank": int(rank), "effective_local_rank": int(local.shape[1]), "conditional_stratum": f"{_stratum(prefix)[0]}:{_stratum(prefix)[1]}", "conditional_global_exact_length_bin": bool(conditional_exact), "conditional_global_length_bin_distance": length_distance, "conditional_global_resolved_stratum": f"{resolved_stratum[0]}:{resolved_stratum[1]}", "wrong_prefix_count": len(wrong_evs), "ev_local": local_global_ev, "ev_conditional_global": global_ev, "ev_wrong_mean": float(np.mean(wrong_evs)) if wrong_evs else float("nan"), "delta_conditional_global": local_global_ev-global_ev, "delta_wrong": delta_wrong}
        ranks = rank_map[prefix_id][np.asarray(evaluation_candidate_indices)]
        for top_k in map(int,top_ks):
            mask = ranks <= top_k; row[f"top{top_k}_token_count"] = int(mask.sum())
            if int(mask.sum()) >= high_minimum:
                high = target[mask]; row[f"delta_conditional_global_top{top_k}"] = explained_variance(high, local[:, :effective_global])-explained_variance(high, conditional[:, :effective_global])
                wrong_high = []
                for wrong_id in wrong_ids:
                    wrong_full = local_bases.get(wrong_id)
                    if wrong_full is None: continue
                    wrong=wrong_full[:,:min(int(rank),wrong_full.shape[1])]
                    effective = min(local.shape[1], wrong.shape[1]); wrong_high.append(explained_variance(high, local[:, :effective])-explained_variance(high, wrong[:, :effective]))
                row[f"delta_wrong_top{top_k}"] = float(np.mean(wrong_high)) if wrong_high else float("nan")
            else:
                row[f"delta_conditional_global_top{top_k}"] = float("nan"); row[f"delta_wrong_top{top_k}"] = float("nan")
        if content_bases is not None:
            content_full=content_bases.get(int(full_index))
            if content_full is None: row["delta_content_appendix"]=float("nan")
            else:
                content=content_full[:,:min(int(rank),content_full.shape[1])]; effective=min(local.shape[1],content.shape[1]); row["delta_content_appendix"]=explained_variance(target,local[:,:effective])-explained_variance(target,content[:,:effective])
        elif content_paths is not None:
            try:
                content, _ = content_subspace(np.load(content_paths[int(full_index)]), rank, excluded_positions=set(excluded_content_positions or ())); effective = min(local.shape[1], content.shape[1]); row["delta_content_appendix"] = explained_variance(target, local[:, :effective])-explained_variance(target, content[:, :effective])
            except RankError: row["delta_content_appendix"] = float("nan")
        rows.append(row)
    return rows


def _select_primary(dev_rows, layers, ranks, selection_rank, fraction):
    layer_scores = {}
    for layer in layers:
        selected = [row for row in dev_rows if row["layer"] == layer and row["rank"] == selection_rank]
        values = [min(row["delta_conditional_global"], row["delta_wrong"]) for row in selected if np.isfinite(row["delta_wrong"])]
        layer_scores[int(layer)] = float(np.mean(values)) if values else float("-inf")
    selected_layer = max(layer_scores, key=lambda layer: (layer_scores[layer], -layer))
    if not np.isfinite(layer_scores[selected_layer]):
        raise RuntimeError(
            "No development layer has both a conditioned-Global and a "
            "Wrong-prefix comparison; enlarge the Wrong-prefix donor pool"
        )
    median_ev = {}
    for rank in ranks:
        values = [row["ev_local"] for row in dev_rows if row["layer"] == selected_layer and row["rank"] == rank]
        median_ev[int(rank)] = float(np.median(values)) if values else float("nan")
    reference = median_ev[max(map(int, ranks))]
    candidates = [int(rank) for rank in ranks if np.isfinite(median_ev[int(rank)]) and median_ev[int(rank)] >= fraction*reference]
    selected_rank = min(candidates) if candidates else int(selection_rank)
    return selected_layer, selected_rank, {"layer_scores": layer_scores, "median_dev_local_ev_by_rank": median_ev, "rank_reference": reference}


def _interaction_energy(z, auxiliary_indices, evaluation_indices, analysis_indices):
    centered = double_center(z[evaluation_indices], z[auxiliary_indices], analysis_indices).residuals.astype(np.float64)
    raw = np.asarray(z[np.ix_(evaluation_indices, analysis_indices)], dtype=np.float64); within = raw-raw.mean(axis=1, keepdims=True)
    denominator = float(np.square(within).sum()); numerator = float(np.square(centered).sum())
    return {"interaction_energy": numerator, "within_prefix_token_energy": denominator, "interaction_fraction_eta": numerator/denominator if denominator > 0 else float("nan")}


def _permutation_analysis(residual_manifest, selected_layer, selected_rank, prefixes, wrong_map, config, exact_wrong_target_ids=None):
    replicates = int(config["permutation"]["replicates"]); rng = np.random.default_rng(int(config["seed"])+8831)
    null_values = {"delta_conditional_global": [[] for _ in range(replicates)], "delta_wrong": [[] for _ in range(replicates)]}; diagnostics=[]
    for entry in residual_manifest["entries"]:
        if int(entry["layer"]) != int(selected_layer): continue
        bundle=load_residual_entry(entry); train_r=bundle["train_residuals"]; eval_r=bundle["evaluation_residuals"]; nonaux=bundle["nonauxiliary_prefix_indices"]
        test_ids={prefixes[int(i)]["prefix_id"] for i in nonaux if prefixes[int(i)]["problem_group"]=="analysis_test"}; required_local_ids=test_ids|{wrong_id for prefix_id in test_ids for wrong_id in wrong_map.get(prefix_id,[])}
        inference_positions=np.asarray([position for position,full in enumerate(nonaux) if prefixes[int(full)]["problem_group"] in {"analysis_train","analysis_test"} or prefixes[int(full)]["prefix_id"] in required_local_ids],dtype=np.int64)
        inference_nonaux=nonaux[inference_positions]; train_r=train_r[inference_positions]; eval_r=eval_r[inference_positions]
        labels=stratification_labels(np.asarray([prefixes[int(i)]["prefix_length_bin"] for i in inference_nonaux]),np.asarray([prefixes[int(i)]["reasoning_progress_bin"] for i in inference_nonaux]))
        diag={"layer":int(entry["layer"]),"fold":int(entry["fold"]),**exchangeability_diagnostics(labels,int(config["permutation"]["minimum_stratum_size"])),**permutation_space_size(labels,int(train_r.shape[1]+eval_r.shape[1]))}; moved=[]; plans=set()
        print(f"[permutation] layer={selected_layer} fold={int(entry['fold'])} prefixes={len(inference_nonaux)} replicates={replicates}",flush=True); report_every=max(1,replicates//10)
        for permutation_index in range(replicates):
            p_train,train_plan=permute_prefix_labels_by_token(train_r,labels,rng,return_diagnostics=True); p_eval,eval_plan=permute_prefix_labels_by_token(eval_r,labels,rng,return_diagnostics=True); moved.append((train_plan["actual_moved_prefix_count"],eval_plan["actual_moved_prefix_count"])); plans.add((train_plan["plan_sha256"],eval_plan["plan_sha256"]))
            local,conditional=_fit_controls(p_train,prefixes,inference_nonaux,selected_rank,required_local_ids)
            for local_index,full_index in enumerate(inference_nonaux):
                prefix=prefixes[int(full_index)]
                if prefix["problem_group"]!="analysis_test": continue
                local_basis=local.get(prefix["prefix_id"]); global_basis=conditional.get(_stratum(prefix)); target=p_eval[local_index]
                if local_basis is None or global_basis is None: continue
                effective=min(local_basis.shape[1],global_basis.shape[1]); null_values["delta_conditional_global"][permutation_index].append(explained_variance(target,local_basis[:,:effective])-explained_variance(target,global_basis[:,:effective]))
                wrong_deltas=[]
                for wrong_id in wrong_map.get(prefix["prefix_id"],[]):
                    wrong=local.get(wrong_id)
                    if wrong is None: continue
                    effective=min(local_basis.shape[1],wrong.shape[1]); wrong_deltas.append(explained_variance(target,local_basis[:,:effective])-explained_variance(target,wrong[:,:effective]))
                if wrong_deltas and (exact_wrong_target_ids is None or prefix["prefix_id"] in exact_wrong_target_ids): null_values["delta_wrong"][permutation_index].append(float(np.mean(wrong_deltas)))
            if (permutation_index+1)%report_every==0 or permutation_index+1==replicates: print(f"[permutation] fold={int(entry['fold'])} {permutation_index+1}/{replicates}",flush=True)
        diag.update({"sampled_replicates":replicates,"unique_sampled_joint_plans":len(plans),"actual_moved_prefix_count_train":{"min":min(x[0] for x in moved),"median":float(np.median([x[0] for x in moved])),"max":max(x[0] for x in moved)},"actual_moved_prefix_count_evaluation":{"min":min(x[1] for x in moved),"median":float(np.median([x[1] for x in moved])),"max":max(x[1] for x in moved)}}); diagnostics.append(diag)
    aggregated={metric:np.asarray([float(np.mean(values)) if values else float("nan") for values in per_rep]) for metric,per_rep in null_values.items()}
    valid_fraction=min((row["exchangeable_prefix_fraction"] for row in diagnostics),default=0.0); inference_valid=valid_fraction>=float(config["permutation"]["minimum_exchangeable_prefix_fraction"])
    return aggregated,{"folds":diagnostics,"minimum_exchangeable_prefix_fraction":valid_fraction,"permutation_inference_valid":inference_valid}


def _basis_label_permutation_analysis(residual_manifest, selected_layer, selected_rank, prefixes, wrong_map, config, exact_wrong_target_ids=None):
    """Permute fitted Local-space ownership within length/progress strata."""
    replicates=int(config["permutation"]["replicates"]); rng=np.random.default_rng(int(config["seed"])+8831)
    null_values={"delta_conditional_global":[[] for _ in range(replicates)],"delta_wrong":[[] for _ in range(replicates)]}; diagnostics=[]
    for entry in residual_manifest["entries"]:
        if int(entry["layer"])!=int(selected_layer): continue
        bundle=load_residual_entry(entry); train_r=bundle["train_residuals"]; eval_r=bundle["evaluation_residuals"]; nonaux=bundle["nonauxiliary_prefix_indices"]
        test_positions=[i for i,full in enumerate(nonaux) if prefixes[int(full)]["problem_group"]=="analysis_test"]
        test_ids=[prefixes[int(nonaux[i])]["prefix_id"] for i in test_positions]; required=set(test_ids)|{wrong_id for prefix_id in test_ids for wrong_id in wrong_map.get(prefix_id,[])}
        local,conditional=_fit_controls(train_r,prefixes,nonaux,selected_rank,required)
        labels=stratification_labels(np.asarray([prefixes[int(nonaux[i])]["prefix_length_bin"] for i in test_positions]),np.asarray([prefixes[int(nonaux[i])]["reasoning_progress_bin"] for i in test_positions]))
        global_null={}; wrong_null={}
        for target_axis,(local_index,target_id) in enumerate(zip(test_positions,test_ids)):
            prefix=prefixes[int(nonaux[local_index])]; target=eval_r[local_index]; global_basis=conditional.get(_stratum(prefix)); wrong_bases=[local.get(wrong_id) for wrong_id in wrong_map.get(target_id,[]) if local.get(wrong_id) is not None]
            candidate_axes=np.flatnonzero(labels==labels[target_axis])
            for source_axis in candidate_axes:
                source=local.get(test_ids[int(source_axis)])
                if source is None or global_basis is None: continue
                effective=min(source.shape[1],global_basis.shape[1]); source_global_ev=explained_variance(target,source[:,:effective]); global_ev=explained_variance(target,global_basis[:,:effective]); global_null[(target_axis,int(source_axis))]=source_global_ev-global_ev
                wrong_deltas=[]
                for wrong in wrong_bases:
                    effective=min(source.shape[1],wrong.shape[1]); wrong_deltas.append(explained_variance(target,source[:,:effective])-explained_variance(target,wrong[:,:effective]))
                if wrong_deltas: wrong_null[(target_axis,int(source_axis))]=float(np.mean(wrong_deltas))
        moved_counts=[]; plans=set(); identity=np.arange(len(test_ids),dtype=np.int64)
        for permutation_index in range(replicates):
            assignment=identity.copy()
            for label in np.unique(labels):
                indices=np.flatnonzero(labels==label); assignment[indices]=rng.permutation(indices)
            moved_counts.append(int(np.sum(assignment!=identity))); plans.add(stable_hash(assignment.tolist()))
            for target_axis,source_axis in enumerate(assignment):
                key=(target_axis,int(source_axis))
                if key in global_null: null_values["delta_conditional_global"][permutation_index].append(global_null[key])
                if key in wrong_null and (exact_wrong_target_ids is None or test_ids[target_axis] in exact_wrong_target_ids): null_values["delta_wrong"][permutation_index].append(wrong_null[key])
        exchange=exchangeability_diagnostics(labels,int(config["permutation"]["minimum_stratum_size"])); space=permutation_space_size(labels,1)
        diagnostics.append({"layer":int(entry["layer"]),"fold":int(entry["fold"]),"method":"within-stratum fitted-local-basis label permutation","n_prefixes":len(test_ids),**exchange,"stratum_label_permutations_log10":space["distinct_label_permutations_log10"],"sampled_replicates":replicates,"unique_sampled_plans":len(plans),"actual_moved_prefix_count":{"min":min(moved_counts),"median":float(np.median(moved_counts)),"max":max(moved_counts)}})
        print(f"[permutation] fold={int(entry['fold'])} completed {replicates} fitted-basis label permutations",flush=True)
    aggregated={metric:np.asarray([float(np.mean(values)) if values else float("nan") for values in per_rep]) for metric,per_rep in null_values.items()}
    valid_fraction=min((row["exchangeable_prefix_fraction"] for row in diagnostics),default=0.0); inference_valid=valid_fraction>=float(config["permutation"]["minimum_exchangeable_prefix_fraction"])
    return aggregated,{"method":"within-stratum fitted-local-basis label permutation","folds":diagnostics,"minimum_exchangeable_prefix_fraction":valid_fraction,"permutation_inference_valid":inference_valid}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--force",action="store_true"); parser.add_argument("--skip-content-appendix",action="store_true"); args = parser.parse_args(); config = load_config(args.config); root = ensure_layout(config)
    global _COMPACT_BASIS_STORAGE
    _COMPACT_BASIS_STORAGE = bool(config.get("replication_mode", False))
    residual_path, hidden_path, wrong_path = root/"manifests/residuals.json", root/"manifests/hidden_states.json", root/"controls/wrong_prefixes.jsonl"
    inputs = {"residuals_sha256": file_sha256(residual_path), "hidden_states_sha256": file_sha256(hidden_path), "wrong_prefixes_sha256": file_sha256(wrong_path), "content_appendix_mode":"skip" if args.skip_content_appendix else "compute"}; manifest_path = root/"manifests/paper_geometry.json"
    if not args.force and stage_is_complete(manifest_path, config, inputs): print(manifest_path); return
    residual_manifest, hidden_manifest = read_json(residual_path), read_json(hidden_path); prefixes = read_jsonl(hidden_manifest["prefix_snapshot"]); wrong_rows = read_jsonl(wrong_path); wrong_map = {row["prefix_id"]: row["wrong_prefix_ids"] for row in wrong_rows}; relaxed_wrong_targets={row["prefix_id"] for row in wrong_rows if int(row.get("relaxed_length_wrong_prefixes",0))>0}; wrong_diagnostics=read_json(root/"manifests/wrong_prefixes.json")["diagnostics"]
    candidates = read_json(root/"candidate_tokens/candidate_tokens.json"); candidate_ids = np.asarray(candidates["candidate_token_ids"], dtype=np.int64); source_root=Path(str(config.get("source_results_root",root))); rank_root=root if bool(config.get("replication_independent_tokenizer",False)) else source_root; rank_map = _candidate_rank_map(rank_root, prefixes, candidate_ids)
    ranks = list(map(int, config["analysis"]["ranks"])); top_ks=list(map(int,config["analysis"]["high_probability_top_ks"])); top_k = int(config["analysis"]["high_probability_primary_top_k"]); high_minimum = int(config["analysis"]["high_probability_min_tokens"])
    all_rows, energy_rows = [], []
    excluded_content_positions=set(map(int,config["content"].get("attention_sink_positions",[])))
    if bool(config["content"].get("exclude_bos",True)): excluded_content_positions.add(0)
    hidden_by_layer = {int(entry["layer"]): entry for entry in hidden_manifest["layers"]}
    prefix_aux = np.asarray([i for i,row in enumerate(prefixes) if row["problem_group"]=="auxiliary"], dtype=np.int64); prefix_test = np.asarray([i for i,row in enumerate(prefixes) if row["problem_group"]=="analysis_test"], dtype=np.int64)
    for layer, hidden_entry in hidden_by_layer.items():
        z = np.load(hidden_entry["successor_path"], mmap_mode="r"); energy_rows.append({"layer": layer, **_interaction_energy(z, prefix_aux, prefix_test, candidates["analysis_indices"])})
    maximum_rank=max(ranks); content_cache={}
    content_enabled=bool(config["content"].get("enabled_appendix")) and not args.skip_content_appendix
    if content_enabled:
        content_targets=[i for i,row in enumerate(prefixes) if row["problem_group"] in {"analysis_dev","analysis_test"}]
        for layer in hidden_by_layer:
            layer_cache={}
            for i in content_targets:
                try: layer_cache[i]=content_subspace(np.load(root/f"hidden_states/content/layer_{layer}_prefix_{i:05d}.npy"),maximum_rank,excluded_positions=set(excluded_content_positions))[0]
                except RankError: layer_cache[i]=None
            content_cache[layer]=layer_cache
    total_entries=len(residual_manifest["entries"])
    for entry_number,entry in enumerate(residual_manifest["entries"],start=1):
        bundle = load_residual_entry(entry); layer = int(entry["layer"]); nonaux = bundle["nonauxiliary_prefix_indices"]; fitted_controls=_fit_controls(bundle["train_residuals"],prefixes,nonaux,maximum_rank); print(f"[geometry_curves] entry={entry_number}/{total_entries} layer={layer} fold={int(entry['fold'])} fitted_rank={maximum_rank}",flush=True)
        for rank in ranks:
            for split in ("analysis_dev","analysis_test"):
                all_rows.extend(_evaluate_rows(bundle["train_residuals"], bundle["evaluation_residuals"], prefixes, nonaux, wrong_map, rank, layer, int(entry["fold"]), split, bundle["evaluation_candidate_indices"], rank_map, top_ks, high_minimum, None, excluded_content_positions, fitted_controls, content_cache.get(layer)))
    for row in all_rows: row["wrong_control_exact_length_bin"]=row["prefix_id"] not in relaxed_wrong_targets
    dev_rows = [row for row in all_rows if row["split"]=="development" and bool(row["conditional_global_exact_length_bin"])]; layers = sorted(hidden_by_layer)
    if bool(config.get("replication_mode",False)):
        source_geometry_path=source_root/"metrics/paper_geometry_summary.json"; source_hidden_path=source_root/"manifests/hidden_states.json"; source_artifacts_available=source_geometry_path.is_file() and source_hidden_path.is_file(); fixed_layer=config.get("replication_confirmatory_fixed_layer"); fixed_rank=config.get("replication_confirmatory_fixed_rank")
        if source_artifacts_available:
            source_geometry=read_json(source_geometry_path); source_hidden=read_json(source_hidden_path); source_layer=int(source_geometry["selected_layer"]); source_depth=0.0 if source_layer==0 else (source_layer+1)/int(source_hidden["model"]["num_decoder_layers"]); target_depth=int(hidden_manifest["model"]["num_decoder_layers"]); corresponding_layer=0 if source_depth==0 else min(layers,key=lambda layer:abs((layer+1)/target_depth-source_depth)); selected_layer=int(corresponding_layer if fixed_layer is None else fixed_layer); selected_rank=int(source_geometry["selected_rank"] if fixed_rank is None else fixed_rank)
        else:
            if fixed_layer is None or fixed_rank is None:
                raise FileNotFoundError(f"replication source geometry is unavailable ({source_geometry_path}, {source_hidden_path}); set replication_confirmatory_fixed_layer and replication_confirmatory_fixed_rank to run a self-contained fixed-condition replication")
            source_layer=None; source_depth=None; corresponding_layer=None; selected_layer=int(fixed_layer); selected_rank=int(fixed_rank)
        if selected_layer not in layers: raise ValueError(f"confirmatory replication layer {selected_layer} was not extracted")
        if selected_rank not in ranks: raise ValueError(f"confirmatory replication rank {selected_rank} was not configured")
        selection_diagnostics={"replication_fixed_from":str(source_root) if source_artifacts_available else "explicit_confirmatory_config","configured_source_results_root":str(source_root),"source_artifacts_available":source_artifacts_available,"confirmatory_no_reselection":True,"source_layer":source_layer,"source_normalized_depth":source_depth,"target_corresponding_layer":corresponding_layer,"target_confirmatory_layer":selected_layer,"fixed_rank":selected_rank}
    else:
        selected_layer, selected_rank, selection_diagnostics = _select_primary(dev_rows, layers, ranks, int(config["analysis"]["selection_rank"]), float(config["analysis"]["r90_fraction"]))
    test_primary = [row for row in all_rows if row["split"]=="evaluation" and row["layer"]==selected_layer and row["rank"]==selected_rank]
    pooled_top_k_rows = _pooled_top_k_rows(residual_manifest, selected_layer, selected_rank, prefixes, wrong_map, relaxed_wrong_targets, rank_map, top_ks)
    rotation_rows=[]; rotation_rank_rows=[]
    report_multirank_controls=bool(config["analysis"].get("report_multirank_controls",False))
    for entry in residual_manifest["entries"]:
        if int(entry["layer"]) != int(selected_layer): continue
        bundle=load_residual_entry(entry)
        ranks_to_report=ranks if report_multirank_controls else [selected_rank]
        rank_rows=_rotation_rows_for_ranks(bundle["train_residuals"],prefixes,bundle["nonauxiliary_prefix_indices"],wrong_map,ranks_to_report,selected_layer,int(entry["fold"]),int(config["seed"]))
        for row in rank_rows:
            row["wrong_control_exact_length_bin"]=row["prefix_id"] not in relaxed_wrong_targets
        rotation_rank_rows.extend(rank_rows)
        rotation_rows.extend(row for row in rank_rows if int(row["rank"])==int(selected_rank))
    expected_primary_rows=int(config["data"]["evaluation_prefixes"])*int(config["candidates"]["folds"])
    expected_wrong=int(config["controls"]["wrong_prefixes_per_target"])
    complete_wrong_rows=sum(int(row["wrong_prefix_count"])==expected_wrong for row in test_primary)
    complete_rotation_rows=sum(int(row["wrong_prefix_count"])==expected_wrong and int(row["split_half_effective_rank"])==selected_rank and int(row["between_min_effective_rank"])==selected_rank and np.isfinite(row["R_between_minus_within"]) for row in rotation_rows)
    exact_conditional_rows=[row for row in test_primary if bool(row["conditional_global_exact_length_bin"])]
    summary = {"selected_layer": selected_layer, "selected_rank": selected_rank, "selection_split": "analysis_dev_exact_bins", "selection_rules": config["selection"], "selection_diagnostics": selection_diagnostics, "content_appendix_computed":content_enabled, "conditional_global_definition":{"covariance":"mean_i K_i / trace(K_i)","conditioning":["prefix_length_bin","reasoning_progress_bin"],"prefix_weighting":"equal after total-response-energy normalization","primary_analysis":"exact bins","fallback":"nearest length bin within the same reasoning-progress bin, reported as sensitivity only"}, "rotation_distance_definition":"1 - ||U_i^T U_k||_F^2 / r", "conditional_global_coverage":{"expected_prefix_fold_rows":expected_primary_rows,"observed_prefix_fold_rows_with_fallback":len(test_primary),"exact_prefix_fold_rows":len(exact_conditional_rows),"fallback_prefix_fold_rows":len(test_primary)-len(exact_conditional_rows),"exact_fraction":len(exact_conditional_rows)/max(1,expected_primary_rows),"complete_with_fallback":len(test_primary)==expected_primary_rows}, "wrong_basis_coverage":{"expected_prefix_fold_rows":expected_primary_rows,"complete_prefix_fold_rows":complete_wrong_rows,"wrong_prefixes_per_target":expected_wrong,"complete":len(test_primary)==expected_primary_rows and complete_wrong_rows==expected_primary_rows}, "rotation_coverage":{"expected_prefix_fold_rows":expected_primary_rows,"observed_prefix_fold_rows":len(rotation_rows),"complete_primary_rank_rows":complete_rotation_rows,"complete":len(rotation_rows)==expected_primary_rows and complete_rotation_rows==expected_primary_rows}, "wrong_control_diagnostics":wrong_diagnostics, "interaction_energy_by_layer": energy_rows}
    summary["fitted_basis_solver_dtype"]="float64"; summary["fitted_basis_storage_dtype"]="float32" if _COMPACT_BASIS_STORAGE else "float64"
    bootstrap_args={"replicates":int(config["statistics"]["bootstrap_replicates"]),"seed":int(config["seed"]),"ci":float(config["statistics"]["ci"])}
    summary["delta_conditional_global"]=problem_bootstrap(np.asarray([row["delta_conditional_global"] for row in exact_conditional_rows]),np.asarray([row["problem_id"] for row in exact_conditional_rows]),**bootstrap_args)
    summary["delta_conditional_global_with_fallback"]=problem_bootstrap(np.asarray([row["delta_conditional_global"] for row in test_primary]),np.asarray([row["problem_id"] for row in test_primary]),replicates=bootstrap_args["replicates"],seed=bootstrap_args["seed"]+11,ci=bootstrap_args["ci"])
    summary["delta_wrong"]=problem_bootstrap(np.asarray([row["delta_wrong"] for row in test_primary]),np.asarray([row["problem_id"] for row in test_primary]),replicates=bootstrap_args["replicates"],seed=bootstrap_args["seed"]+29,ci=bootstrap_args["ci"])
    exact_test_primary=[row for row in test_primary if bool(row["wrong_control_exact_length_bin"])]
    summary["wrong_exact_bin_coverage"]={"expected_problems":int(config["data"]["evaluation_prefixes"]),"exact_bin_problems":len({row["problem_id"] for row in exact_test_primary}),"exact_bin_prefix_fold_rows":len(exact_test_primary),"fraction":len({row["problem_id"] for row in exact_test_primary})/max(1,int(config["data"]["evaluation_prefixes"]))}
    for metric in ("delta_wrong",*[f"delta_wrong_top{cutoff}" for cutoff in top_ks]):
        summary[f"{metric}_exact_bin"]=problem_bootstrap(np.asarray([row[metric] for row in exact_test_primary]),np.asarray([row["problem_id"] for row in exact_test_primary]),replicates=int(config["statistics"]["bootstrap_replicates"]),seed=int(config["seed"])+43,ci=float(config["statistics"]["ci"]))
    _add_pooled_top_k_summary(summary,pooled_top_k_rows,top_ks,high_minimum,config)
    for metric in ("d_rotation_local_conditional_global","d_rotation_local_wrong_mean","R_within","R_between","R_between_minus_within"):
        summary[metric]=problem_bootstrap(np.asarray([row[metric] for row in rotation_rows]),np.asarray([row["problem_id"] for row in rotation_rows]),replicates=int(config["statistics"]["bootstrap_replicates"]),seed=int(config["seed"])+271,ci=float(config["statistics"]["ci"]))
    r90_rows=[]
    for prefix_id in sorted({row["prefix_id"] for row in all_rows if row["split"]=="evaluation"}):
        curve={rank:np.mean([row["ev_local"] for row in all_rows if row["split"]=="evaluation" and row["layer"]==selected_layer and row["rank"]==rank and row["prefix_id"]==prefix_id]) for rank in ranks}; reference=curve[max(ranks)]; achieved=[rank for rank in ranks if curve[rank]>=float(config["analysis"]["r90_fraction"])*reference]; r90_rows.append({"prefix_id":prefix_id,"r90":min(achieved) if achieved else None,"ev_rank64":reference})
    summary["r90"]={"median":float(np.median([row["r90"] for row in r90_rows if row["r90"] is not None])),"fraction_le_32":float(np.mean([row["r90"]<=32 for row in r90_rows if row["r90"] is not None])),"rows":r90_rows}
    run_permutations=not bool(config.get("replication_mode",False)) or bool(config.get("replication",{}).get("run_permutations",False))
    permutation_path=root/"permutation/paper_null_summary.json"
    if run_permutations:
        exact_wrong_target_ids={row["prefix_id"] for row in wrong_rows if row["split"]=="evaluation" and int(row.get("relaxed_length_wrong_prefixes",0))==0}
        null_values,permutation_diagnostics=_basis_label_permutation_analysis(residual_manifest,selected_layer,selected_rank,prefixes,wrong_map,config,exact_wrong_target_ids); summary["permutation_diagnostics"]=permutation_diagnostics
        for metric in ("delta_conditional_global","delta_wrong"):
            observed=summary["delta_wrong_exact_bin"]["mean"] if metric=="delta_wrong" else summary[metric]["mean"]; raw_p=permutation_pvalue(observed,null_values[metric]); summary[f"{metric}_permutation_p_raw"]=raw_p; summary[f"{metric}_permutation_p"]=raw_p if permutation_diagnostics["permutation_inference_valid"] else float("nan")
        atomic_json(permutation_path,{"null":{key:value.tolist() for key,value in null_values.items()},"diagnostics":permutation_diagnostics,"recentered":False})
    else:
        summary["permutation_diagnostics"]={"skipped":True,"reason":"cheap checkpoint replication repeats Experiment 1 geometry without permutation inference"}; atomic_json(permutation_path,summary["permutation_diagnostics"])
    rows_path=root/"metrics/paper_geometry_rows.csv"; rotation_path=root/"metrics/paper_rotation_rows.csv"; rotation_rank_path=root/"metrics/paper_rotation_rank_rows.csv"; pooled_path=root/"metrics/paper_topk_pooled_rows.csv"; energy_path=root/"metrics/interaction_energy.csv"; summary_path=root/"metrics/paper_geometry_summary.json"; _write_csv(rows_path,all_rows); _write_csv(rotation_path,rotation_rows); _write_csv(rotation_rank_path,rotation_rank_rows); _write_csv(pooled_path,pooled_top_k_rows); _write_csv(energy_path,energy_rows); atomic_json(summary_path,summary)
    atomic_json(manifest_path,{"complete":True,"config_hash":stable_hash(config),**inputs,"rows":str(rows_path),"rows_sha256":file_sha256(rows_path),"rotation_rows":str(rotation_path),"rotation_rows_sha256":file_sha256(rotation_path),"rotation_rank_rows":str(rotation_rank_path),"rotation_rank_rows_sha256":file_sha256(rotation_rank_path),"pooled_top_k_rows":str(pooled_path),"pooled_top_k_rows_sha256":file_sha256(pooled_path),"energy":str(energy_path),"summary":str(summary_path),"permutation":str(permutation_path),"selected_layer":selected_layer,"selected_rank":selected_rank}); print(manifest_path)


if __name__=="__main__": main()
