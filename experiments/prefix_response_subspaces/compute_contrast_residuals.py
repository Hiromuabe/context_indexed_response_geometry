from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .src.residualization import center_train_and_evaluation
from .src.storage import save_layer_array
from .src.utils import atomic_json, ensure_layout, file_sha256, load_config, read_json, read_jsonl, stable_hash, stage_is_complete


def _chunked_double_center_to_npy(
    z: np.ndarray,
    evaluation_prefixes: np.ndarray,
    auxiliary_prefixes: np.ndarray,
    token_indices: list[int] | np.ndarray,
    destination: Path,
    prefix_chunk_size: int,
) -> tuple[float, list[int]]:
    """Double-center into a float32 NPY without materializing the full tensor."""
    tokens = np.asarray(token_indices, dtype=np.int64)
    if tokens.ndim != 1 or not len(tokens) or len(np.unique(tokens)) != len(tokens):
        raise ValueError("token_indices must be a non-empty unique vector")
    if tokens.min() < 0 or tokens.max() >= z.shape[1]:
        raise ValueError("token_indices contain an out-of-range index")
    hidden = int(z.shape[2])
    token_sum = np.zeros((len(tokens), hidden), dtype=np.float64)
    chunk_size = max(1, int(prefix_chunk_size))
    for start in range(0, len(auxiliary_prefixes), chunk_size):
        indices = auxiliary_prefixes[start:start + chunk_size]
        block = np.asarray(z[indices[:, None], tokens[None, :], :], dtype=np.float32)
        if not np.isfinite(block).all():
            raise ValueError("auxiliary hidden states contain NaN or Inf")
        token_sum += block.sum(axis=0, dtype=np.float64)
    token_mean = (token_sum / float(len(auxiliary_prefixes))).astype(np.float32)
    grand_mean = token_mean.mean(axis=0, dtype=np.float64).astype(np.float32)
    shape = (len(evaluation_prefixes), len(tokens), hidden)
    output = np.lib.format.open_memmap(destination, mode="w+", dtype=np.float32, shape=shape)
    maximum_row_mean = 0.0
    for start in range(0, len(evaluation_prefixes), chunk_size):
        indices = evaluation_prefixes[start:start + chunk_size]
        block = np.asarray(z[indices[:, None], tokens[None, :], :], dtype=np.float32)
        if not np.isfinite(block).all():
            raise ValueError("evaluation hidden states contain NaN or Inf")
        prefix_mean = block.mean(axis=1, dtype=np.float64).astype(np.float32)
        block -= prefix_mean[:, None]
        block -= token_mean[None]
        block += grand_mean[None, None]
        correction = block.mean(axis=1, dtype=np.float64).astype(np.float32)
        block -= correction[:, None]
        row_mean = block.mean(axis=1, dtype=np.float64)
        maximum_row_mean = max(maximum_row_mean, float(np.abs(row_mean).max(initial=0.0)))
        output[start:start + len(indices)] = block
    output.flush()
    return maximum_row_mean, list(shape)


def _entry_files_exist(entry: dict) -> bool:
    if entry.get("storage_format") == "npy_bundle":
        return Path(entry["train_path"]).is_file() and Path(entry["evaluation_path"]).is_file()
    return Path(entry["path"]).is_file()


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--force", action="store_true"); args = parser.parse_args()
    config = load_config(args.config); root = ensure_layout(config)
    hidden_manifest_path = root / "manifests/hidden_states.json"; candidate_path = root / "candidate_tokens/candidate_tokens.json"
    inputs = {"hidden_manifest_sha256": file_sha256(hidden_manifest_path), "candidate_tokens_sha256": file_sha256(candidate_path)}
    manifest_path = root / "manifests/residuals.json"
    if not args.force and stage_is_complete(manifest_path, config, inputs): print(manifest_path); return
    hidden_manifest = read_json(hidden_manifest_path); candidates = read_json(candidate_path); prefixes = read_jsonl(hidden_manifest["prefix_snapshot"])
    auxiliary = np.asarray([i for i, row in enumerate(prefixes) if row["problem_group"] == "auxiliary"], dtype=np.int64)
    evaluation = np.asarray([i for i, row in enumerate(prefixes) if row["problem_group"] != "auxiliary"], dtype=np.int64)
    if not len(auxiliary) or not len(evaluation): raise ValueError("residualization requires auxiliary and non-auxiliary prefixes")
    # The 3B replication has twice the hidden width of the main model.  Store
    # fold arrays separately and memory-map them so neither residualization nor
    # the subsequent geometry stage needs a multi-gigabyte in-memory copy.
    mmap_bundle = bool(config.get("replication_mode", False))
    checkpoint_path = root / "residuals/checkpoint.json"
    checkpoint_key = stable_hash({"version": 2, "config_hash": stable_hash(config), "inputs": inputs, "storage_format": "npy_bundle" if mmap_bundle else "npz"})
    checkpoint = read_json(checkpoint_path) if checkpoint_path.is_file() and not args.force else {}
    entries = list(checkpoint.get("entries", [])) if checkpoint.get("checkpoint_key") == checkpoint_key else []
    if any(not _entry_files_exist(entry) for entry in entries):
        entries = []
    completed_keys = {(int(entry["layer"]), int(entry["fold"])) for entry in entries}
    maximum_row_mean = float(checkpoint.get("maximum_absolute_row_mean", 0.0)) if entries else 0.0
    total = len(hidden_manifest["layers"]) * len(candidates["folds"]); completed = len(entries); started = time.monotonic()
    print(f"[residuals] rebuilding {total} layer/fold files storage={'npy_bundle' if mmap_bundle else 'npz'} resume={completed}", flush=True)
    for layer_entry in hidden_manifest["layers"]:
        z = np.load(layer_entry["successor_path"], mmap_mode="r")
        for fold in candidates["folds"]:
            key = (int(layer_entry["layer"]), int(fold["fold_id"]))
            if key in completed_keys:
                print(f"[residuals] SKIP completed layer={key[0]} fold={key[1]}", flush=True); continue
            print(f"[residuals] START layer={key[0]} fold={key[1]}", flush=True)
            if mmap_bundle:
                train_path = root / f"residuals/layer_{key[0]}_fold_{key[1]}_train.npy"
                evaluation_path = root / f"residuals/layer_{key[0]}_fold_{key[1]}_evaluation.npy"
                chunk = int(config.get("extraction", {}).get("prefix_chunk_size", 8))
                train_mean, train_shape = _chunked_double_center_to_npy(z, evaluation, auxiliary, fold["train_indices"], train_path, chunk)
                heldout_mean, heldout_shape = _chunked_double_center_to_npy(z, evaluation, auxiliary, fold["evaluation_indices"], evaluation_path, chunk)
                maximum_row_mean = max(maximum_row_mean, train_mean, heldout_mean)
                entry = {"layer": key[0], "fold": key[1], "storage_format": "npy_bundle", "train_path": str(train_path), "train_sha256": file_sha256(train_path), "evaluation_path": str(evaluation_path), "evaluation_sha256": file_sha256(evaluation_path), "train_shape": train_shape, "evaluation_shape": heldout_shape, "nonauxiliary_prefix_indices": evaluation.tolist(), "train_candidate_indices": list(map(int, fold["train_indices"])), "evaluation_candidate_indices": list(map(int, fold["evaluation_indices"]))}
                size = train_path.stat().st_size + evaluation_path.stat().st_size
            else:
                z32 = np.asarray(z, dtype=np.float32)
                train, heldout = center_train_and_evaluation(z32[evaluation], z32[auxiliary], fold["train_indices"], fold["evaluation_indices"])
                train_mean = float(np.abs(train.residuals.mean(axis=1, dtype=np.float64)).max()); heldout_mean = float(np.abs(heldout.residuals.mean(axis=1, dtype=np.float64)).max()); maximum_row_mean = max(maximum_row_mean, train_mean, heldout_mean)
                path = root / f"residuals/layer_{key[0]}_fold_{key[1]}.npz"; save_layer_array(path, train_residuals=train.residuals.astype(np.float32), evaluation_residuals=heldout.residuals.astype(np.float32), nonauxiliary_prefix_indices=evaluation, train_candidate_indices=train.token_indices, evaluation_candidate_indices=heldout.token_indices)
                entry = {"layer": key[0], "fold": key[1], "path": str(path), "sha256": file_sha256(path), "train_shape": list(train.residuals.shape), "evaluation_shape": list(heldout.residuals.shape)}; size = path.stat().st_size
            entries.append(entry); completed += 1
            atomic_json(checkpoint_path, {"checkpoint_key": checkpoint_key, "complete": False, "entries": entries, "maximum_absolute_row_mean": maximum_row_mean})
            rate = (completed - len(completed_keys)) / max(time.monotonic() - started, 1e-9)
            print(f"[residuals] DONE {completed}/{total} layer={key[0]} fold={key[1]} size_gib={size/1024**3:.2f} eta={(total-completed)/max(rate,1e-9)/60:.1f}m", flush=True)
    atomic_json(manifest_path, {"complete": True, "config_hash": stable_hash(config), **inputs, "storage_format": "npy_bundle" if mmap_bundle else "npz", "entries": entries, "maximum_absolute_row_mean": maximum_row_mean, "gate_1_row_centering": maximum_row_mean < 2e-5})
    atomic_json(checkpoint_path, {"checkpoint_key": checkpoint_key, "complete": True, "entries": entries, "maximum_absolute_row_mean": maximum_row_mean}); print(manifest_path)


if __name__ == "__main__": main()
