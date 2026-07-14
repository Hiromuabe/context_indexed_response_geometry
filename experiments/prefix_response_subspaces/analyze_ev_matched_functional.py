from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from prefix_displacement.runtime import prepare_data_parallel
from experiments.prefix_successor_subspaces.src.model import _load_backbone_and_tokenizer

from .analyze_functional_recovery import FunctionalRecoveryForward
from .analyze_paper_geometry import _resolve_conditional_basis
from .src.data import pad_token_rows
from .src.statistics import problem_bootstrap, problem_ratio_bootstrap
from .src.storage import load_residual_entry
from .src.utils import (
    atomic_json,
    file_sha256,
    load_config,
    read_json,
    read_jsonl,
    result_root,
    stable_hash,
    stage_is_complete,
)


CHECKPOINT_VERSION = 2


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def cached_local_basis(cache: dict[str, np.ndarray], train: np.ndarray, position: int, rank: int) -> np.ndarray:
    """Reconstruct a hidden-space basis from the saved sample-space SVD."""
    positions = np.asarray(cache["local_positions"], dtype=np.int64)
    matches = np.flatnonzero(positions == int(position))
    if len(matches) != 1:
        raise RuntimeError(f"rank cache has {len(matches)} entries for local position {position}")
    axis = int(matches[0])
    effective = int(cache["local_effective_ranks"][axis])
    if int(rank) > effective:
        raise RuntimeError(f"cached local position {position} supports rank {effective}, requested {rank}")
    left = np.asarray(cache["local_left_singular_vectors"][axis, :, :rank], dtype=np.float64)
    singular = np.asarray(cache["local_singular_values"][axis, :rank], dtype=np.float64)
    raw = np.asarray(train[position], dtype=np.float64).T @ left
    raw /= singular[None, :]
    basis = np.linalg.qr(raw, mode="reduced")[0]
    if not np.allclose(basis.T @ basis, np.eye(rank), atol=2e-7):
        raise RuntimeError("reconstructed cached basis is not orthonormal")
    return basis.astype(np.float32)


def cached_common_bases(cache: dict[str, np.ndarray], rank: int) -> dict[tuple[int, int], np.ndarray | None]:
    result: dict[tuple[int, int], np.ndarray | None] = {}
    for axis, values in enumerate(np.asarray(cache["common_strata"], dtype=np.int64)):
        key = (int(values[0]), int(values[1]))
        effective = int(cache["common_effective_ranks"][axis])
        if effective <= 0:
            result[key] = None
        elif int(rank) > effective:
            raise RuntimeError(f"cached common stratum {key} supports rank {effective}, requested {rank}")
        else:
            result[key] = np.asarray(cache["common_bases"][axis, :, :rank], dtype=np.float32)
    return result


def load_reference_cells(path: Path, target_rank: int) -> dict[tuple[str, int, int], dict[str, dict[str, Any]]]:
    """Load Oracle/Rank-0/target-rank results from the existing main run."""
    cells: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            condition = row.get("condition")
            if condition not in {"Oracle", "Rank-0", "Local"}:
                continue
            if condition == "Local":
                if int(row.get("rank", target_rank)) != int(target_rank):
                    raise RuntimeError("saved Local functional rows do not use the EV-matching target rank")
                if int(row.get("effective_rank", target_rank)) != int(target_rank):
                    raise RuntimeError("saved Local functional rows were rank-reduced; cannot label them target rank 64")
            key = (row["prefix_id"], int(row["fold"]), int(row["candidate_index"]))
            cells[key][condition] = {
                **row,
                "problem_id": row["problem_id"],
                "js": float(row["js"]),
            }
    incomplete = [key for key, conditions in cells.items() if set(conditions) != {"Oracle", "Rank-0", "Local"}]
    if incomplete:
        raise RuntimeError(f"existing paper functional output has {len(incomplete)} incomplete reference cells")
    if not cells:
        raise RuntimeError("existing paper functional output contains no reusable evaluation cells")
    return dict(cells)


def summarize_ev_matched_cells(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    problem_ids = np.asarray([row["problem_id"] for row in rows])
    bootstrap = {
        "replicates": int(config["statistics"]["bootstrap_replicates"]),
        "ci": float(config["statistics"]["ci"]),
    }
    result: dict[str, Any] = {
        "definitions": {
            "distance": "Jensen-Shannon distance from the unmodified next-token distribution",
            "gain": "D_rank0 - D_condition",
            "target_advantage": "D_control - D_target = G_target - G_control; positive values favor the target context",
            "inference": "the algebraically identical JSD and gain differences are one paired contrast, with one problem-bootstrap CI",
            "wrong_context_aggregation": "average wrong contexts within each candidate cell before problem-level bootstrap",
            "bootstrap_unit": "problem_id",
        }
    }
    metric_keys = ("D_rank0", "D_target", "D_matched_common", "D_wrong_mean", "G_target", "G_matched_common", "G_wrong_mean")
    for index, key in enumerate(metric_keys):
        result[key] = problem_bootstrap(
            np.asarray([float(row[key]) for row in rows]),
            problem_ids,
            seed=int(config["seed"]) + 3100 + index,
            **bootstrap,
        )
    for index, (label, key) in enumerate((
        ("matched_common", "target_advantage_vs_matched_common"),
        ("wrong_context", "target_advantage_vs_wrong_context"),
    )):
        values = np.asarray([
            float(row["D_matched_common"] - row["D_target"])
            if label == "matched_common"
            else float(row["D_wrong_mean"] - row["D_target"])
            for row in rows
        ])
        result[key] = problem_bootstrap(values, problem_ids, seed=int(config["seed"]) + 3120 + index, **bootstrap)
    d0 = np.asarray([float(row["D_rank0"]) for row in rows])
    for index, (label, key) in enumerate((
        ("target", "D_target"),
        ("matched_common", "D_matched_common"),
        ("wrong_mean", "D_wrong_mean"),
    )):
        distance = np.asarray([float(row[key]) for row in rows])
        result[f"recovery_fraction_{label}"] = problem_ratio_bootstrap(
            d0 - distance,
            d0,
            problem_ids,
            seed=int(config["seed"]) + 3140 + index,
            **bootstrap,
        )
    result["n_cells"] = len(rows)
    result["n_problems"] = len(set(problem_ids.tolist()))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--results-root")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(args.results_root) if args.results_root else result_root(config)

    geometry_path = root / "metrics/paper_geometry_summary.json"
    selection_path = root / "metrics/ev_matched_rank_selection.json"
    hidden_path = root / "manifests/hidden_states.json"
    residual_path = root / "manifests/residuals.json"
    wrong_path = root / "controls/wrong_prefixes.jsonl"
    candidate_path = root / "candidate_tokens/candidate_tokens.json"
    paper_rows_path = root / "functional/paper_distribution_rows.csv"
    paper_manifest_path = root / "manifests/paper_functional.json"
    control_rank_manifest_path = root / "manifests/control_rank_sensitivity.json"
    inputs = {
        "geometry_sha256": file_sha256(geometry_path),
        "rank_selection_sha256": file_sha256(selection_path),
        "hidden_states_sha256": file_sha256(hidden_path),
        "residuals_sha256": file_sha256(residual_path),
        "wrong_prefixes_sha256": file_sha256(wrong_path),
        "candidate_tokens_sha256": file_sha256(candidate_path),
        "paper_distribution_rows_sha256": file_sha256(paper_rows_path),
        "paper_functional_manifest_sha256": file_sha256(paper_manifest_path),
        "control_rank_manifest_sha256": file_sha256(control_rank_manifest_path),
    }
    manifest_path = root / "manifests/ev_matched_functional.json"
    if not args.force and stage_is_complete(manifest_path, config, inputs):
        print(manifest_path)
        return

    geometry = read_json(geometry_path)
    selection = read_json(selection_path)
    hidden = read_json(hidden_path)
    residual_manifest = read_json(residual_path)
    candidates = read_json(candidate_path)
    prefixes = read_jsonl(hidden["prefix_snapshot"])
    wrong_rows = read_jsonl(wrong_path)
    control_rank_manifest = read_json(control_rank_manifest_path)
    cache_by_fold = {int(row["fold"]): row for row in control_rank_manifest["rank_basis_cache"]}
    wrong_map = {row["prefix_id"]: list(row["wrong_prefix_ids"]) for row in wrong_rows}
    relaxed_wrong = {row["prefix_id"] for row in wrong_rows if int(row.get("relaxed_length_wrong_prefixes", 0)) > 0}
    target_rank = int(selection["definition"]["target_rank"])
    common_rank = int(selection["matched_common"]["selected_rank"])
    wrong_rank = int(selection["wrong_context"]["selected_rank"])
    layer = int(geometry["selected_layer"])
    if int(geometry["selected_rank"]) != target_rank:
        raise RuntimeError(f"existing paper functional target rank is {geometry['selected_rank']}, expected {target_rank}")
    if bool(config.get("replication_mode", False)):
        raise RuntimeError("EV-matched reinjection is intentionally restricted to the main model")
    references = load_reference_cells(paper_rows_path, target_rank)
    oracle_max = max(float(conditions["Oracle"]["js"]) for conditions in references.values())
    if oracle_max > float(config["functional"]["oracle_tolerance"]):
        raise RuntimeError(f"reused paper functional Oracle check failed: maximum JS={oracle_max}")

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for EV-matched functional recovery")
    layer_entry = next(item for item in hidden["layers"] if int(item["layer"]) == layer)
    z = np.load(layer_entry["successor_path"], mmap_mode="r")
    backbone, tokenizer, source, precision_name, _dtype = _load_backbone_and_tokenizer(config, args.model_path)
    forward, device, device_ids = prepare_data_parallel(FunctionalRecoveryForward.build(backbone, layer, js_only=True))
    forward.eval()
    expected_wrong = int(config["controls"]["wrong_prefixes_per_target"])
    auxiliary = np.asarray([i for i, row in enumerate(prefixes) if row["problem_group"] == "auxiliary"], dtype=np.int64)
    calibration = np.asarray(candidates["calibration_indices"], dtype=np.int64)
    aux_token = np.asarray(z[auxiliary], dtype=np.float32).mean(axis=0)
    aux_calibration = np.asarray(z[np.ix_(auxiliary, calibration)], dtype=np.float32).mean(axis=(0, 1))

    all_rows: list[dict[str, Any]] = []
    checkpoint_root = root / "functional/ev_matched_checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    entries = [entry for entry in residual_manifest["entries"] if int(entry["layer"]) == layer]
    for entry_number, entry in enumerate(entries, start=1):
        bundle = load_residual_entry(entry)
        train = bundle["train_residuals"]
        nonaux = np.asarray(bundle["nonauxiliary_prefix_indices"], dtype=np.int64)
        fold = int(entry["fold"])
        evaluation_tokens = np.asarray(bundle["evaluation_candidate_indices"], dtype=np.int64)
        position_by_id = {prefixes[int(full)]["prefix_id"]: index for index, full in enumerate(nonaux)}
        test_ids = {prefixes[int(full)]["prefix_id"] for full in nonaux if prefixes[int(full)]["problem_group"] == "analysis_test"}
        wrong_ids = {wrong_id for prefix_id in test_ids for wrong_id in wrong_map.get(prefix_id, []) if wrong_id in position_by_id}
        cache_entry = cache_by_fold.get(fold)
        if cache_entry is None:
            raise RuntimeError(f"rank-basis cache is missing fold {fold}")
        cache_path = Path(cache_entry["path"])
        if file_sha256(cache_path) != cache_entry["sha256"]:
            raise RuntimeError(f"rank-basis cache checksum mismatch: {cache_path}")
        with np.load(cache_path, allow_pickle=False) as loaded_cache:
            rank_cache = {key: loaded_cache[key] for key in loaded_cache.files}
        if int(rank_cache["maximum_rank"]) < max(common_rank, wrong_rank):
            raise RuntimeError("rank-basis cache does not reach the selected control rank")
        wrong_bases = {
            wrong_id: cached_local_basis(rank_cache, train, position_by_id[wrong_id], wrong_rank)
            for wrong_id in sorted(wrong_ids)
        }
        conditional = cached_common_bases(rank_cache, common_rank)
        records: list[dict[str, Any]] = []
        for prefix_index, prefix in enumerate(prefixes):
            if prefix["problem_group"] != "analysis_test":
                continue
            common_basis, common_exact, length_distance, _resolved = _resolve_conditional_basis(conditional, prefix)
            if common_basis is None:
                raise RuntimeError(f"no matched-common basis for {prefix['prefix_id']}")
            selected_wrong = [(wrong_id, wrong_bases.get(wrong_id)) for wrong_id in wrong_map.get(prefix["prefix_id"], [])]
            selected_wrong = [(wrong_id, basis) for wrong_id, basis in selected_wrong if basis is not None]
            if len(selected_wrong) != expected_wrong:
                raise RuntimeError(f"{prefix['prefix_id']} has {len(selected_wrong)} wrong bases, expected {expected_wrong}")
            prefix_calibration = np.asarray(z[prefix_index, calibration], dtype=np.float32).mean(axis=0)
            for token_index in evaluation_tokens:
                oracle = np.asarray(z[prefix_index, token_index], dtype=np.float32)
                baseline = prefix_calibration + aux_token[token_index] - aux_calibration
                interaction = oracle - baseline
                conditions = [("MatchedCommon", common_basis, common_rank)] + [
                    (f"Wrong::{wrong_id}", basis, wrong_rank) for wrong_id, basis in selected_wrong
                ]
                for condition, basis, nominal_rank in conditions:
                    replacement = baseline + basis @ (basis.T @ interaction)
                    records.append({
                        "prefix_index": prefix_index, "problem_id": prefix["problem_id"], "prefix_id": prefix["prefix_id"],
                        "candidate_index": int(token_index), "candidate_token_id": int(candidates["candidate_token_ids"][int(token_index)]),
                        "condition": condition, "nominal_rank": int(nominal_rank), "effective_rank": int(basis.shape[1]),
                        "matched_common_exact_bin": bool(common_exact), "matched_common_length_bin_distance": length_distance,
                        "wrong_context_exact_bin": prefix["prefix_id"] not in relaxed_wrong, "replacement": replacement,
                    })
        records.sort(key=lambda row: (int(row["prefix_index"]), int(row["candidate_index"]), str(row["condition"])))
        batch_size = int(config["functional"]["per_device_batch_size"]) * max(1, len(device_ids))
        rows_path = checkpoint_root / f"fold_{fold}_rows.jsonl"
        meta_path = checkpoint_root / f"fold_{fold}.json"
        checkpoint_key = stable_hash({
            "version": CHECKPOINT_VERSION, "inputs": inputs, "config_hash": stable_hash(config), "model_source": source,
            "precision": precision_name, "layer": layer, "target_rank": target_rank, "common_rank": common_rank,
            "wrong_rank": wrong_rank, "fold": fold, "record_count": len(records), "batch_size": batch_size,
        })
        meta = read_json(meta_path) if meta_path.is_file() else {}
        if args.force or meta.get("checkpoint_key") != checkpoint_key or not rows_path.is_file():
            rows_path.write_text("", encoding="utf-8")
            atomic_json(meta_path, {"checkpoint_key": checkpoint_key, "complete": False, "completed_records": 0})
            fold_rows: list[dict[str, Any]] = []
        else:
            fold_rows = read_jsonl(rows_path)
        completed = len(fold_rows)
        if completed > len(records):
            raise RuntimeError("EV-matched checkpoint contains more rows than the regenerated design")
        for checkpoint_index in ({0, completed - 1} if completed else set()):
            expected = records[checkpoint_index]
            observed = fold_rows[checkpoint_index]
            if (observed.get("prefix_id"), int(observed.get("candidate_index", -1)), observed.get("condition")) != (
                expected["prefix_id"], int(expected["candidate_index"]), expected["condition"]
            ):
                raise RuntimeError("EV-matched checkpoint order does not match the regenerated design")
        total_batches = (len(records) + batch_size - 1) // batch_size
        report_every = max(1, total_batches // 20)
        started = time.monotonic()
        print(f"[ev_matched_functional] fold={fold} entry={entry_number}/{len(entries)} records={len(records)} resume={completed}", flush=True)
        with rows_path.open("a", encoding="utf-8") as handle:
            for start in range(completed, len(records), batch_size):
                batch = records[start : start + batch_size]
                sequences = [prefixes[row["prefix_index"]]["prefix_token_ids"] + [row["candidate_token_id"]] for row in batch]
                ids, mask, positions = pad_token_rows(sequences, tokenizer.pad_token_id)
                replacements = torch.from_numpy(np.stack([row["replacement"] for row in batch])).float()
                oracle_mask = torch.zeros(len(batch), dtype=torch.bool)
                sample = torch.arange(start, start + len(batch))
                cell = torch.tensor([
                    int(row["prefix_index"]) * len(candidates["candidate_token_ids"]) + int(row["candidate_index"])
                    for row in batch
                ], dtype=torch.long)
                js, _kl, _top1, _overlap, _logit_difference, observed = forward(
                    ids.to(device), mask.to(device), positions.to(device), replacements.to(device),
                    oracle_mask.to(device), sample.to(device), cell.to(device),
                )
                if not torch.equal(observed.cpu(), sample):
                    raise RuntimeError("DataParallel changed EV-matched functional batch order")
                batch_rows = []
                for axis, row in enumerate(batch):
                    clean = {key: value for key, value in row.items() if key != "replacement"}
                    clean.update({"layer": layer, "fold": fold, "js": float(js[axis].cpu())})
                    batch_rows.append(clean)
                    handle.write(json.dumps(clean, sort_keys=True, ensure_ascii=False) + "\n")
                handle.flush()
                fold_rows.extend(batch_rows)
                if args.preflight_only:
                    atomic_json(meta_path, {"checkpoint_key": checkpoint_key, "complete": False, "completed_records": len(fold_rows), "preflight_pass": True})
                    print(f"preflight checkpoint saved at {rows_path}")
                    return
                batch_number = start // batch_size + 1
                if batch_number % report_every == 0 or batch_number == total_batches:
                    rate = (batch_number - completed // batch_size) / max(time.monotonic() - started, 1e-9)
                    eta = (total_batches - batch_number) / max(rate, 1e-9) / 60
                    print(f"[ev_matched_functional] fold={fold} {batch_number}/{total_batches} eta={eta:.1f}m", flush=True)
        atomic_json(meta_path, {"checkpoint_key": checkpoint_key, "complete": True, "completed_records": len(fold_rows), "rows": str(rows_path)})
        all_rows.extend(fold_rows)

    new_cells: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in all_rows:
        key = (row["prefix_id"], int(row["fold"]), int(row["candidate_index"]))
        new_cells[key][row["condition"]] = row
    aggregate: list[dict[str, Any]] = []
    for key, reference in references.items():
        controls = new_cells.get(key, {})
        common = controls.get("MatchedCommon")
        wrong = [row for condition, row in controls.items() if condition.startswith("Wrong::")]
        if common is None or len(wrong) != expected_wrong:
            continue
        d0 = float(reference["Rank-0"]["js"])
        target = float(reference["Local"]["js"])
        common_distance = float(common["js"])
        wrong_distance = float(np.mean([float(row["js"]) for row in wrong]))
        aggregate.append({
            "problem_id": reference["Local"]["problem_id"], "prefix_id": key[0], "fold": key[1], "candidate_index": key[2],
            "target_rank": target_rank, "matched_common_rank": common_rank, "wrong_context_rank": wrong_rank,
            "wrong_context_count": len(wrong), "matched_common_exact_bin": bool(common["matched_common_exact_bin"]),
            "wrong_context_exact_bin": bool(common["wrong_context_exact_bin"]),
            "D_oracle": float(reference["Oracle"]["js"]), "D_rank0": d0, "D_target": target,
            "D_matched_common": common_distance, "D_wrong_mean": wrong_distance,
            "G_target": d0 - target, "G_matched_common": d0 - common_distance, "G_wrong_mean": d0 - wrong_distance,
            "target_advantage_vs_matched_common": common_distance - target,
            "target_advantage_vs_wrong_context": wrong_distance - target,
        })
    expected_cells = int(config["data"]["evaluation_prefixes"]) * sum(len(fold["evaluation_indices"]) for fold in candidates["folds"])
    if len(aggregate) != expected_cells:
        raise RuntimeError(f"EV-matched functional coverage is {len(aggregate)}/{expected_cells} cells")
    summary = summarize_ev_matched_cells(aggregate, config)
    summary.update({
        "selected_layer": layer,
        "ranks": {"target_context": target_rank, "matched_common": common_rank, "wrong_context": wrong_rank},
        "development_rank_selection": selection,
        "evaluation_achieved_ev_match": selection["evaluation_achieved_match"],
        "rank_basis_cache_reused": True,
        "reused_main_functional_reference": True,
        "reused_oracle_max_js": oracle_max,
        "control_coverage": {"expected_cells": expected_cells, "observed_cells": len(aggregate), "complete": len(aggregate) == expected_cells},
    })
    distribution_path = root / "functional/ev_matched_distribution_rows.csv"
    cells_path = root / "functional/ev_matched_cell_summary.csv"
    summary_path = root / "functional/ev_matched_summary.json"
    _write_csv(distribution_path, all_rows)
    _write_csv(cells_path, aggregate)
    atomic_json(summary_path, summary)
    atomic_json(manifest_path, {
        "complete": True,
        "config_hash": stable_hash(config),
        **inputs,
        "distribution_rows": str(distribution_path),
        "cell_summary": str(cells_path),
        "summary": str(summary_path),
        "model_source": source,
        "precision": precision_name,
    })
    print(summary_path)


if __name__ == "__main__":
    main()
