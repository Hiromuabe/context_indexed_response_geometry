from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterator

from .schema import require_torch


def _torch_load_cpu(path: Path) -> dict[str, Any]:
    torch = require_torch()
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_cache_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def cache_hidden_dimension(manifest: dict[str, Any]) -> int:
    shapes = manifest["shards"][0]["tensor_shapes"]
    return int(shapes["delta"][-1])


def iter_cache_batches(
    manifest_path: str | Path,
    *,
    split: str,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Iterator[dict[str, Any]]:
    """Load one CPU shard at a time, never the full activation cache."""
    torch = require_torch()
    manifest_path = Path(manifest_path)
    manifest = load_cache_manifest(manifest_path)
    shard_order = list(range(len(manifest["shards"])))
    rng = random.Random(seed)
    if shuffle:
        rng.shuffle(shard_order)
    for shard_index in shard_order:
        shard = manifest["shards"][shard_index]
        tensors = _torch_load_cpu(manifest_path.parent / shard["tensor_file"])
        metadata = read_jsonl(manifest_path.parent / shard["metadata_file"])
        indices = [index for index, row in enumerate(metadata) if row["split"] == split]
        if shuffle:
            rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            if not selected:
                continue
            index_tensor = torch.tensor(selected, dtype=torch.long)
            yield {
                **{name: tensor.index_select(0, index_tensor) for name, tensor in tensors.items()},
                "current_token_id": torch.tensor(
                    [metadata[index]["current_token_id"] for index in selected],
                    dtype=torch.long,
                ),
                "next_token_id": torch.tensor(
                    [metadata[index]["next_token_id"] for index in selected],
                    dtype=torch.long,
                ),
                "position": torch.tensor(
                    [metadata[index]["absolute_position"] for index in selected],
                    dtype=torch.float32,
                ),
                "surprisal": torch.tensor(
                    [metadata[index]["surprisal"] for index in selected],
                    dtype=torch.float32,
                ),
                "metadata": [metadata[index] for index in selected],
            }
        del tensors


def count_split_rows(manifest_path: str | Path, split: str) -> int:
    manifest_path = Path(manifest_path)
    manifest = load_cache_manifest(manifest_path)
    count = 0
    for shard in manifest["shards"]:
        metadata = read_jsonl(manifest_path.parent / shard["metadata_file"])
        count += sum(row["split"] == split for row in metadata)
    return count
