from __future__ import annotations

import argparse
import time

import numpy as np

from .src.data import pad_token_rows
from .src.mechanism import SITE_NAMES, first_layer_value_output_basis, load_first_layer_mechanism_model
from .src.model import assert_output_order
from .src.utils import atomic_json, ensure_layout, file_sha256, load_config, read_json, read_jsonl, stable_hash, stage_is_complete


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    root = ensure_layout(config)
    settings = config.get("first_layer_mechanism", {})
    if not bool(settings.get("enabled", True)):
        print("First-layer mechanism extraction disabled in configuration")
        return

    hidden_path = root / "manifests/hidden_states.json"
    candidate_path = root / "candidate_tokens/candidate_tokens.json"
    inputs = {
        "hidden_states_sha256": file_sha256(hidden_path),
        "candidate_tokens_sha256": file_sha256(candidate_path),
    }
    manifest_path = root / "manifests/first_layer_mechanism_states.json"
    if not args.force and stage_is_complete(manifest_path, config, inputs):
        print(manifest_path)
        return

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for first-layer mechanism extraction")
    hidden = read_json(hidden_path)
    prefixes = read_jsonl(hidden["prefix_snapshot"])
    candidates = read_json(candidate_path)
    token_ids = list(map(int, candidates["candidate_token_ids"]))
    post_mlp_entry = next((entry for entry in hidden["layers"] if int(entry["layer"]) == 0), None)
    if post_mlp_entry is None:
        raise RuntimeError("first-layer mechanism analysis requires saved decoder layer 0 states")
    post_mlp_reference = np.load(post_mlp_entry["successor_path"], mmap_mode="r")
    loaded = load_first_layer_mechanism_model(config, args.model_path)
    if post_mlp_reference.shape != (len(prefixes), len(token_ids), loaded.hidden_size):
        raise RuntimeError("saved layer-0 successor axis does not match mechanism extraction")

    mechanism_root = root / "hidden_states/first_layer_mechanism"
    mechanism_root.mkdir(parents=True, exist_ok=True)
    dtype = np.float16 if config["extraction"]["storage_dtype"] == "float16" else np.float32
    checkpoint_path = mechanism_root / "checkpoint.json"
    checkpoint_key = stable_hash({"version": 1, "config_hash": stable_hash(config), "inputs": inputs, "model": loaded.metadata})
    checkpoint = read_json(checkpoint_path) if checkpoint_path.is_file() else {}
    array_paths = {"pre_attention": mechanism_root / "pre_attention.npy", "post_attention": mechanism_root / "post_attention.npy"}
    resume_arrays = (
        not args.force
        and checkpoint.get("checkpoint_key") == checkpoint_key
        and all(path.is_file() for path in array_paths.values())
    )
    array_mode = "r+" if resume_arrays else "w+"
    arrays = {
        "pre_attention": np.lib.format.open_memmap(
            array_paths["pre_attention"], mode=array_mode, dtype=dtype,
            shape=(len(prefixes), len(token_ids), loaded.hidden_size),
        ),
        "post_attention": np.lib.format.open_memmap(
            array_paths["post_attention"], mode=array_mode, dtype=dtype,
            shape=(len(prefixes), len(token_ids), loaded.hidden_size),
        ),
    }
    expected_shape = (len(prefixes), len(token_ids), loaded.hidden_size)
    if any(tuple(array.shape) != expected_shape or array.dtype != np.dtype(dtype) for array in arrays.values()):
        raise RuntimeError("first-layer mechanism checkpoint array shape/dtype mismatch")
    batch_per_device = int(config["extraction"]["per_device_batch_size"])
    global_batch = max(1, batch_per_device * max(1, len(loaded.device_ids)))
    completed_prefixes = int(checkpoint.get("completed_prefixes", 0)) if resume_arrays else 0
    consistency_max = float(checkpoint.get("post_mlp_consistency_max_abs_difference", 0.0)) if resume_arrays else 0.0
    if completed_prefixes > len(prefixes):
        raise RuntimeError("first-layer mechanism checkpoint contains too many prefixes")
    started = time.monotonic()
    report_every = max(1, len(prefixes) // 100)
    print(f"[first_layer_mechanism] prefixes={len(prefixes)} candidates={len(token_ids)} global_batch={global_batch}", flush=True)
    with torch.no_grad():
        for prefix_index in range(completed_prefixes, len(prefixes)):
            row = prefixes[prefix_index]
            prefix = list(map(int, row["prefix_token_ids"]))
            for start in range(0, len(token_ids), global_batch):
                batch_tokens = token_ids[start:start + global_batch]
                sequences = [prefix + [token] for token in batch_tokens]
                ids, mask, positions = pad_token_rows(sequences, loaded.tokenizer.pad_token_id)
                sample = torch.arange(start, start + len(batch_tokens), dtype=torch.long)
                sites, observed = loaded.model(
                    ids.to(loaded.device), mask.to(loaded.device), positions.to(loaded.device), sample.to(loaded.device)
                )
                assert_output_order(sample, observed)
                values = sites.float().cpu().numpy()
                arrays["pre_attention"][prefix_index, start:start + len(batch_tokens)] = values[:, 0]
                arrays["post_attention"][prefix_index, start:start + len(batch_tokens)] = values[:, 1]
                reference = np.asarray(post_mlp_reference[prefix_index, start:start + len(batch_tokens)])
                observed_stored_precision = values[:, 2].astype(reference.dtype)
                consistency_max = max(consistency_max, float(np.max(np.abs(observed_stored_precision.astype(np.float32) - reference.astype(np.float32)))))
            completed = prefix_index + 1
            for array in arrays.values():
                array.flush()
            atomic_json(checkpoint_path, {
                "checkpoint_key": checkpoint_key,
                "complete": False,
                "completed_prefixes": completed,
                "post_mlp_consistency_max_abs_difference": consistency_max,
                "value_basis_rows": checkpoint.get("value_basis_rows", []) if resume_arrays else [],
            })
            if completed % report_every == 0 or completed == len(prefixes):
                rate = completed / max(time.monotonic() - started, 1e-9)
                print(f"[first_layer_mechanism] {completed}/{len(prefixes)} prefixes rate={rate:.3f}/s eta={(len(prefixes)-completed)/max(rate,1e-9)/60:.1f}m", flush=True)
            if args.preflight_only:
                target_index = next(i for i, item in enumerate(prefixes) if item["problem_group"] in {"analysis_dev", "analysis_test"})
                value_rank = int(settings.get("value_space_rank", 64))
                excluded = set(map(int, settings.get("value_space_excluded_positions", config.get("content", {}).get("attention_sink_positions", [0]))))
                if bool(settings.get("value_space_exclude_bos", True)):
                    excluded.add(0)
                torch.manual_seed(int(config["seed"]) + 2111 + target_index)
                torch.cuda.manual_seed_all(int(config["seed"]) + 2111 + target_index)
                basis, diagnostics = first_layer_value_output_basis(loaded, list(map(int, prefixes[target_index]["prefix_token_ids"])), value_rank, excluded)
                print(f"[first_layer_mechanism] PREFLIGHT PASS sites={list(SITE_NAMES)} value_basis_shape={basis.shape} diagnostics={diagnostics}; rerun without --preflight-only to resume", flush=True)
                return
    for array in arrays.values():
        array.flush()

    tolerance = float(settings.get("post_mlp_consistency_tolerance", 2e-3))
    if consistency_max > tolerance:
        raise RuntimeError(f"re-extracted first-layer block output differs from saved layer-0 states: {consistency_max} > {tolerance}")

    value_rank = int(settings.get("value_space_rank", 64))
    excluded = set(map(int, settings.get("value_space_excluded_positions", config.get("content", {}).get("attention_sink_positions", [0]))))
    if bool(settings.get("value_space_exclude_bos", True)):
        excluded.add(0)
    basis_root = mechanism_root / "value_output_bases"
    basis_root.mkdir(parents=True, exist_ok=True)
    checkpoint = read_json(checkpoint_path)
    basis_rows = list(checkpoint.get("value_basis_rows", []))
    basis_targets = [i for i, row in enumerate(prefixes) if row["problem_group"] in {"analysis_dev", "analysis_test"}]
    if len(basis_rows) > len(basis_targets):
        raise RuntimeError("first-layer value-space checkpoint contains too many prefixes")
    if [int(row["prefix_index"]) for row in basis_rows] != basis_targets[:len(basis_rows)]:
        raise RuntimeError("first-layer value-space checkpoint prefix order mismatch")
    reduction_seed = int(config["seed"]) + 2111
    torch.manual_seed(reduction_seed)
    torch.cuda.manual_seed_all(reduction_seed)
    with torch.no_grad():
        for number in range(len(basis_rows), len(basis_targets)):
            prefix_index = basis_targets[number]
            torch.manual_seed(reduction_seed + prefix_index)
            torch.cuda.manual_seed_all(reduction_seed + prefix_index)
            basis, diagnostics = first_layer_value_output_basis(
                loaded, list(map(int, prefixes[prefix_index]["prefix_token_ids"])), value_rank, excluded
            )
            path = basis_root / f"prefix_{prefix_index:05d}.npy"
            np.save(path, basis.astype(np.float32))
            basis_rows.append({"prefix_index": prefix_index, "prefix_id": prefixes[prefix_index]["prefix_id"], "path": str(path), "sha256": file_sha256(path), **diagnostics})
            atomic_json(checkpoint_path, {
                "checkpoint_key": checkpoint_key,
                "complete": False,
                "completed_prefixes": len(prefixes),
                "post_mlp_consistency_max_abs_difference": consistency_max,
                "value_basis_rows": basis_rows,
            })
            completed_bases = number + 1
            if completed_bases % max(1, len(basis_targets) // 20) == 0 or completed_bases == len(basis_targets):
                print(f"[first_layer_value_space] {completed_bases}/{len(basis_targets)} prefixes", flush=True)

    site_entries = [
        {"site": "pre_attention", "path": str(mechanism_root / "pre_attention.npy"), "sha256": file_sha256(mechanism_root / "pre_attention.npy"), "shape": list(arrays["pre_attention"].shape), "dtype": str(arrays["pre_attention"].dtype)},
        {"site": "post_attention", "path": str(mechanism_root / "post_attention.npy"), "sha256": file_sha256(mechanism_root / "post_attention.npy"), "shape": list(arrays["post_attention"].shape), "dtype": str(arrays["post_attention"].dtype)},
        {"site": "post_mlp", "path": post_mlp_entry["successor_path"], "sha256": post_mlp_entry["successor_sha256"], "shape": post_mlp_entry["shape"], "dtype": post_mlp_entry["dtype"], "reused_from_hidden_states_manifest": True},
    ]
    atomic_json(manifest_path, {
        "complete": True,
        "config_hash": stable_hash(config),
        **inputs,
        "sites": site_entries,
        "site_order": list(SITE_NAMES),
        "value_output_basis_definition": "span{W_O E_h W_V^(h) input_layernorm(h_i,t)} over prefix positions and attention heads",
        "value_output_rank": value_rank,
        "value_output_reduction_seed": reduction_seed,
        "value_output_bases": basis_rows,
        "post_mlp_consistency_max_abs_difference": consistency_max,
        "post_mlp_consistency_tolerance": tolerance,
        "model": loaded.metadata,
    })
    atomic_json(checkpoint_path, {
        "checkpoint_key": checkpoint_key,
        "complete": True,
        "completed_prefixes": len(prefixes),
        "post_mlp_consistency_max_abs_difference": consistency_max,
        "value_basis_rows": basis_rows,
    })
    print(manifest_path)


if __name__ == "__main__":
    main()
