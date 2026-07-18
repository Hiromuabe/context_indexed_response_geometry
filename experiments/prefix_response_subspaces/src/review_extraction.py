from __future__ import annotations

import time
import copy
from pathlib import Path
from typing import Any

import numpy as np

from .data import pad_token_rows
from .model import assert_output_order, load_endpoint_model
from .review_model import load_early_exit_endpoint_model
from .utils import atomic_json, file_sha256, read_json, stable_hash


def precomputed_cell_mask(precomputed: dict[str, Any] | None, context_count: int, token_count: int) -> np.ndarray:
    if precomputed is None:
        return np.zeros((int(context_count), int(token_count)), dtype=bool)
    context_indices = np.asarray(precomputed["context_indices"], dtype=np.int64)
    token_indices = np.asarray(precomputed["token_indices"], dtype=np.int64)
    if context_indices.shape != (int(context_count),) or token_indices.shape != (int(token_count),):
        raise ValueError("precomputed source axes do not match the destination grid")
    return (context_indices >= 0)[:, None] & (token_indices >= 0)[None, :]


def copy_precomputed_cells(
    arrays: dict[int, np.ndarray], precomputed: dict[str, Any] | None, *, context_chunk_size: int = 8
) -> int:
    """Copy an outer product of reusable context/token cells into output memmaps."""
    if precomputed is None:
        return 0
    context_indices = np.asarray(precomputed["context_indices"], dtype=np.int64)
    token_indices = np.asarray(precomputed["token_indices"], dtype=np.int64)
    destination_contexts = np.flatnonzero(context_indices >= 0)
    destination_tokens = np.flatnonzero(token_indices >= 0)
    if not len(destination_contexts) or not len(destination_tokens):
        return 0
    layer_paths = {int(layer): Path(path) for layer, path in precomputed["layers"].items()}
    for layer, output in arrays.items():
        if int(layer) not in layer_paths:
            raise ValueError(f"precomputed source is missing layer {layer}")
        source = np.load(layer_paths[int(layer)], mmap_mode="r")
        if int(source.shape[2]) != int(output.shape[2]):
            raise ValueError(f"precomputed layer {layer} hidden width does not match")
        for start in range(0, len(destination_contexts), max(1, int(context_chunk_size))):
            destination_chunk = destination_contexts[start : start + max(1, int(context_chunk_size))]
            source_chunk = context_indices[destination_chunk]
            block = np.asarray(source[source_chunk[:, None], token_indices[destination_tokens][None, :], :])
            output[destination_chunk[:, None], destination_tokens[None, :], :] = block.astype(output.dtype, copy=False)
        if hasattr(output, "flush"):
            output.flush()
    return int(len(destination_contexts) * len(destination_tokens))


def extract_endpoint_grid(
    config: dict[str, Any],
    *,
    contexts: list[dict[str, Any]],
    token_ids: list[int],
    output_root: Path,
    checkpoint_path: Path,
    model_path: str | None,
    force: bool,
    precomputed: dict[str, Any] | None = None,
    target_layers: list[int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Resumably extract layer endpoints for a context x candidate grid."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for review-experiment hidden-state extraction")
    if not contexts or not token_ids:
        raise ValueError("contexts and token_ids must be non-empty")
    model_config = copy.deepcopy(config)
    if target_layers is not None:
        selected = list(dict.fromkeys(map(int, target_layers)))
        if not selected:
            raise ValueError("target_layers must be non-empty when provided")
        model_config["model"]["target_layers"] = selected
        model_config["model"]["additional_target_layers"] = []
    early_exit = bool(config.get("review_experiments", {}).get("early_exit_extraction", True))
    loader = load_early_exit_endpoint_model if early_exit else load_endpoint_model
    loaded = loader(model_config, model_path)
    dtype = np.float16 if config["extraction"]["storage_dtype"] == "float16" else np.float32
    output_root.mkdir(parents=True, exist_ok=True)
    key = stable_hash({
        "version": 2,
        "config": config,
        "context_ids": [row["context_id"] for row in contexts],
        "token_ids": list(map(int, token_ids)),
        "model": loaded.metadata,
        "target_layers": loaded.layer_ids,
        "early_exit_extraction": early_exit,
        "precomputed": {
            "fingerprint": precomputed.get("fingerprint") if precomputed else None,
            "context_indices": precomputed.get("context_indices") if precomputed else None,
            "token_indices": precomputed.get("token_indices") if precomputed else None,
        },
    })
    checkpoint = read_json(checkpoint_path) if checkpoint_path.is_file() and not force else {}
    completed = int(checkpoint.get("completed_contexts", 0)) if checkpoint.get("checkpoint_key") == key else 0
    paths = {layer: output_root / f"layer_{layer}.npy" for layer in loaded.layer_ids}
    expected_shape = (len(contexts), len(token_ids), loaded.hidden_size)
    reusable_mask = precomputed_cell_mask(precomputed, len(contexts), len(token_ids))
    resumable = completed > 0 and all(path.is_file() for path in paths.values())
    if not resumable:
        completed = 0
        arrays = {
            layer: np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=expected_shape)
            for layer, path in paths.items()
        }
        copied_cells = copy_precomputed_cells(arrays, precomputed)
    else:
        arrays = {layer: np.lib.format.open_memmap(path, mode="r+") for layer, path in paths.items()}
        if any(tuple(array.shape) != expected_shape for array in arrays.values()):
            raise RuntimeError("review extraction checkpoint shape does not match regenerated design")
        copied_cells = int(reusable_mask.sum())
    global_batch = max(1, int(config["extraction"]["per_device_batch_size"]) * max(1, len(loaded.device_ids)))
    started = time.monotonic()
    total_cells = int(np.prod(reusable_mask.shape))
    forward_cells = total_cells - int(reusable_mask.sum())
    completed_forward_cells = int((~reusable_mask[:completed]).sum())
    report_every = max(1, len(contexts) // 100)
    print(
        f"[review_extraction] contexts={len(contexts)} candidates={len(token_ids)} layers={loaded.layer_ids} "
        f"reused_cells={int(reusable_mask.sum())}/{total_cells} forward_cells={forward_cells} resume={completed}",
        flush=True,
    )
    with torch.no_grad():
        for context_index in range(completed, len(contexts)):
            pending_positions = np.flatnonzero(~reusable_mask[context_index])
            prefix = list(map(int, contexts[context_index]["prefix_token_ids"])) if len(pending_positions) else []
            for start in range(0, len(pending_positions), global_batch):
                destination_positions = pending_positions[start : start + global_batch]
                candidate_batch = [token_ids[int(position)] for position in destination_positions]
                ids, mask, positions = pad_token_rows([prefix + [token_id] for token_id in candidate_batch], loaded.tokenizer.pad_token_id)
                sample = torch.tensor(destination_positions, dtype=torch.long)
                endpoints, observed = loaded.model(ids.to(loaded.device), mask.to(loaded.device), positions.to(loaded.device), sample.to(loaded.device))
                assert_output_order(sample, observed)
                values = endpoints.float().cpu().numpy()
                for layer_axis, layer in enumerate(loaded.layer_ids):
                    arrays[layer][context_index, destination_positions] = values[:, layer_axis]
            if len(pending_positions):
                for array in arrays.values():
                    array.flush()
            done = context_index + 1
            if len(pending_positions) or done == len(contexts):
                atomic_json(checkpoint_path, {"checkpoint_key": key, "complete": False, "completed_contexts": done})
            processed_forward_cells = int((~reusable_mask[completed:done]).sum())
            rate = processed_forward_cells / max(time.monotonic() - started, 1e-9)
            remaining = forward_cells - completed_forward_cells - processed_forward_cells
            if done % report_every == 0 or done == len(contexts):
                print(
                    f"[review_extraction] {done}/{len(contexts)} forward_cells={completed_forward_cells + processed_forward_cells}/{forward_cells} "
                    f"rate={rate:.1f} cells/s eta={remaining/max(rate,1e-9)/60:.1f}m",
                    flush=True,
                )
    entries = []
    for layer, path in paths.items():
        arrays[layer].flush()
        entries.append({
            "layer": int(layer), "path": str(path), "sha256": file_sha256(path),
            "shape": list(expected_shape), "dtype": str(arrays[layer].dtype),
        })
    atomic_json(checkpoint_path, {"checkpoint_key": key, "complete": True, "completed_contexts": len(contexts)})
    diagnostics = {
        "total_cells": total_cells,
        "precomputed_cells": int(reusable_mask.sum()),
        "model_forward_cells": forward_cells,
        "precomputed_fraction": float(reusable_mask.mean()),
        "copied_cells_on_this_run": copied_cells,
        "early_exit_after_layer": max(loaded.layer_ids) if early_exit else None,
    }
    return entries, loaded.metadata, diagnostics
