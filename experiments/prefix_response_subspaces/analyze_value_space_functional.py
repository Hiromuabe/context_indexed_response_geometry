from __future__ import annotations

import argparse
import csv
import json
import time

import numpy as np

from prefix_displacement.runtime import prepare_data_parallel
from experiments.prefix_successor_subspaces.src.model import _load_backbone_and_tokenizer

from .analyze_functional_recovery import FunctionalRecoveryForward
from .src.data import pad_token_rows
from .src.statistics import problem_bootstrap
from .src.utils import atomic_json, ensure_layout, file_sha256, load_config, read_json, read_jsonl, stable_hash, stage_is_complete


CHECKPOINT_VERSION = 1


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    root = ensure_layout(config)
    settings = config.get("first_layer_mechanism", {})
    if not bool(settings.get("run_value_functional", True)):
        print("Value-space functional analysis disabled in configuration")
        return

    mechanism_path = root / "manifests/first_layer_mechanism_states.json"
    functional_path = root / "functional/paper_cell_summary.csv"
    functional_summary_path = root / "functional/paper_summary.json"
    hidden_path = root / "manifests/hidden_states.json"
    candidate_path = root / "candidate_tokens/candidate_tokens.json"
    inputs = {
        "mechanism_states_sha256": file_sha256(mechanism_path),
        "paper_functional_cells_sha256": file_sha256(functional_path),
        "paper_functional_summary_sha256": file_sha256(functional_summary_path),
        "hidden_states_sha256": file_sha256(hidden_path),
        "candidate_tokens_sha256": file_sha256(candidate_path),
    }
    manifest_path = root / "manifests/value_space_functional.json"
    if not args.force and stage_is_complete(manifest_path, config, inputs):
        print(manifest_path)
        return

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for value-space functional recovery")
    paper_functional_summary = read_json(functional_summary_path)
    if not bool(paper_functional_summary["oracle_pass"]):
        raise RuntimeError("value-space functional comparison requires a passed paper Oracle")
    layer = int(paper_functional_summary["selected_layer"])
    if layer != 0:
        raise RuntimeError("first-layer value-space functional comparison requires the primary Functional site to be layer 0")

    mechanism = read_json(mechanism_path)
    hidden = read_json(hidden_path)
    prefixes = read_jsonl(hidden["prefix_snapshot"])
    candidates = read_json(candidate_path)
    basis_paths = {int(row["prefix_index"]): row["path"] for row in mechanism["value_output_bases"]}
    cell_rows = _read_csv(functional_path)
    cell_map = {
        (row["prefix_id"], int(row["fold"]), int(row["candidate_index"])): row
        for row in cell_rows
    }
    layer_entry = next(entry for entry in hidden["layers"] if int(entry["layer"]) == 0)
    z = np.asarray(np.load(layer_entry["successor_path"], mmap_mode="r"), dtype=np.float32)
    auxiliary = np.asarray([i for i, row in enumerate(prefixes) if row["problem_group"] == "auxiliary"], dtype=np.int64)
    calibration = np.asarray(candidates["calibration_indices"], dtype=np.int64)
    aux_token = z[auxiliary].mean(axis=0)
    aux_calibration = z[np.ix_(auxiliary, calibration)].mean(axis=(0, 1))

    backbone, tokenizer, source, precision_name, _dtype = _load_backbone_and_tokenizer(config, args.model_path)
    forward, device, device_ids = prepare_data_parallel(FunctionalRecoveryForward.build(backbone, 0, js_only=True))
    forward.eval()
    records = []
    for prefix_index, prefix in enumerate(prefixes):
        if prefix["problem_group"] != "analysis_test":
            continue
        basis_path = basis_paths.get(prefix_index)
        if basis_path is None:
            continue
        basis = np.asarray(np.load(basis_path), dtype=np.float32)
        prefix_calibration = z[prefix_index, calibration].mean(axis=0)
        for fold in candidates["folds"]:
            fold_id = int(fold["fold_id"])
            for token_index in map(int, fold["evaluation_indices"]):
                key = (prefix["prefix_id"], fold_id, token_index)
                paper_cell = cell_map.get(key)
                if paper_cell is None:
                    continue
                oracle = z[prefix_index, token_index]
                baseline = prefix_calibration + aux_token[token_index] - aux_calibration
                interaction = oracle - baseline
                replacement = baseline + basis @ (basis.T @ interaction)
                records.append({
                    "problem_id": prefix["problem_id"],
                    "prefix_id": prefix["prefix_id"],
                    "prefix_index": prefix_index,
                    "fold": fold_id,
                    "candidate_index": token_index,
                    "candidate_token_id": int(candidates["candidate_token_ids"][token_index]),
                    "D_rank0": float(paper_cell["D_rank0"]),
                    "G_local": float(paper_cell["G_local"]),
                    "effective_value_rank": int(basis.shape[1]),
                    "replacement": replacement,
                })
    records.sort(key=lambda row: (row["prefix_index"], row["fold"], row["candidate_index"]))
    expected = int(config["data"]["evaluation_prefixes"]) * len(candidates["analysis_indices"])
    if len(records) != expected:
        raise RuntimeError(f"value-space functional coverage is incomplete before forward: {len(records)}/{expected}")

    checkpoint_root = root / "functional/value_space_checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    checkpoint_rows_path = checkpoint_root / "rows.jsonl"
    checkpoint_meta_path = checkpoint_root / "manifest.json"
    batch_size = int(config["functional"]["per_device_batch_size"]) * max(1, len(device_ids))
    checkpoint_key = stable_hash({
        "version": CHECKPOINT_VERSION,
        "inputs": inputs,
        "config_hash": stable_hash(config),
        "model_source": source,
        "precision": precision_name,
        "record_count": len(records),
        "batch_size": batch_size,
    })
    checkpoint_meta = read_json(checkpoint_meta_path) if checkpoint_meta_path.is_file() else {}
    if args.force or checkpoint_meta.get("checkpoint_key") != checkpoint_key or not checkpoint_rows_path.is_file():
        checkpoint_rows_path.write_text("", encoding="utf-8")
        atomic_json(checkpoint_meta_path, {"checkpoint_key": checkpoint_key, "complete": False, "completed_records": 0})
        rows = []
    else:
        rows = read_jsonl(checkpoint_rows_path)
    if len(rows) > len(records):
        raise RuntimeError("value-space functional checkpoint contains too many rows")

    total_batches = (len(records) + batch_size - 1) // batch_size
    report_every = max(1, total_batches // 20)
    started = time.monotonic()
    with torch.no_grad(), checkpoint_rows_path.open("a", encoding="utf-8") as handle:
        for start in range(len(rows), len(records), batch_size):
            batch = records[start:start + batch_size]
            sequences = [prefixes[row["prefix_index"]]["prefix_token_ids"] + [row["candidate_token_id"]] for row in batch]
            ids, mask, positions = pad_token_rows(sequences, tokenizer.pad_token_id)
            replacement = torch.from_numpy(np.stack([row["replacement"] for row in batch])).float()
            oracle_mask = torch.zeros(len(batch), dtype=torch.bool)
            sample = torch.arange(start, start + len(batch), dtype=torch.long)
            cell = torch.tensor([row["prefix_index"] * len(candidates["candidate_token_ids"]) + row["candidate_index"] for row in batch], dtype=torch.long)
            js, _kl, _top1, _overlap, _logit_difference, observed = forward(
                ids.to(device), mask.to(device), positions.to(device), replacement.to(device), oracle_mask.to(device), sample.to(device), cell.to(device)
            )
            if not torch.equal(observed.cpu(), sample):
                raise RuntimeError("DataParallel changed value-space functional batch order")
            for axis, row in enumerate(batch):
                clean = {key: value for key, value in row.items() if key != "replacement"}
                clean["D_value_output"] = float(js[axis].cpu())
                clean["G_value_output"] = clean["D_rank0"] - clean["D_value_output"]
                clean["G_local_minus_value_output"] = clean["G_local"] - clean["G_value_output"]
                handle.write(json.dumps(clean, sort_keys=True, ensure_ascii=False) + "\n")
                rows.append(clean)
            handle.flush()
            batch_number = start // batch_size + 1
            if batch_number % report_every == 0 or batch_number == total_batches:
                rate = batch_number / max(time.monotonic() - started, 1e-9)
                print(f"[value_functional] {batch_number}/{total_batches} batches rate={rate:.3f}/s eta={(total_batches-batch_number)/max(rate,1e-9)/60:.1f}m", flush=True)
    atomic_json(checkpoint_meta_path, {"checkpoint_key": checkpoint_key, "complete": True, "completed_records": len(rows), "rows": str(checkpoint_rows_path)})

    bootstrap_args = {
        "replicates": int(config["statistics"]["bootstrap_replicates"]),
        "seed": int(config["seed"]) + 2309,
        "ci": float(config["statistics"]["ci"]),
    }
    problem_ids = np.asarray([row["problem_id"] for row in rows])
    summary = {
        "site": "first decoder block output",
        "basis": mechanism["value_output_basis_definition"],
        "coverage": {"expected_cells": expected, "observed_cells": len(rows), "complete": len(rows) == expected},
    }
    for metric in ("G_value_output", "G_local_minus_value_output"):
        summary[metric] = problem_bootstrap(np.asarray([row[metric] for row in rows]), problem_ids, **bootstrap_args)
    summary["diagnostic_local_functionally_above_value_output"] = float(summary["G_local_minus_value_output"]["ci_low"]) > 0
    rows_path = root / "functional/value_space_rows.csv"
    summary_path = root / "functional/value_space_summary.json"
    _write_csv(rows_path, rows)
    atomic_json(summary_path, summary)
    atomic_json(manifest_path, {
        "complete": True,
        "config_hash": stable_hash(config),
        **inputs,
        "rows": str(rows_path),
        "summary": str(summary_path),
        "model_source": source,
        "precision": precision_name,
    })
    print(manifest_path)


if __name__ == "__main__":
    main()
