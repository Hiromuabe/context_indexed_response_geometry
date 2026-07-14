from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .utils import file_sha256


def save_layer_array(path: str | Path, **arrays: np.ndarray) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for name, value in arrays.items():
        array = np.asarray(value)
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN or Inf")
    np.savez(destination, **arrays)
    return destination


def manifest_entry(path: str | Path, **metadata: Any) -> dict[str, Any]:
    path = Path(path)
    return {"path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size, **metadata}


def load_npz_checked(entry: dict[str, Any], mmap_mode: str | None = None) -> Any:
    path = Path(entry["path"])
    if file_sha256(path) != entry["sha256"]:
        raise RuntimeError(f"artifact checksum mismatch: {path}")
    return np.load(path, mmap_mode=mmap_mode, allow_pickle=False)


def load_residual_entry(entry: dict[str, Any], mmap_mode: str | None = "r") -> Any:
    """Load legacy NPZ or the paper-replication memory-mapped NPY bundle."""
    if entry.get("storage_format") != "npy_bundle":
        return np.load(entry["path"], mmap_mode=mmap_mode, allow_pickle=False)
    required = ("train_path", "evaluation_path", "nonauxiliary_prefix_indices", "train_candidate_indices", "evaluation_candidate_indices")
    missing = [name for name in required if name not in entry]
    if missing:
        raise RuntimeError(f"residual NPY bundle is incomplete: {missing}")
    return {
        "train_residuals": np.load(entry["train_path"], mmap_mode=mmap_mode, allow_pickle=False),
        "evaluation_residuals": np.load(entry["evaluation_path"], mmap_mode=mmap_mode, allow_pickle=False),
        "nonauxiliary_prefix_indices": np.asarray(entry["nonauxiliary_prefix_indices"], dtype=np.int64),
        "train_candidate_indices": np.asarray(entry["train_candidate_indices"], dtype=np.int64),
        "evaluation_candidate_indices": np.asarray(entry["evaluation_candidate_indices"], dtype=np.int64),
    }
