from __future__ import annotations

import hashlib
import json
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .config import CacheConfig
from .schema import (
    METADATA_FIELDS,
    TENSOR_FIELDS,
    TransitionSchemaError,
    normalize_transition_record,
    record_tensor_nbytes,
    require_torch,
)
from .split_registry import (
    SplitLeakageError,
    assert_rows_respect_registry,
    registry_sha256,
    validate_split_registry,
)


class CacheValidationError(ValueError):
    """Raised when a shard or manifest violates the immutable cache contract."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _torch_load_cpu(path: Path):
    torch = require_torch()
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def runtime_environment() -> dict[str, Any]:
    torch = require_torch()
    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    return {
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "platform": platform.platform(),
        "cuda_available": cuda_available,
        "cuda_runtime": str(torch.version.cuda) if torch.version.cuda is not None else None,
        "gpu_count": device_count,
        "gpu_names": [torch.cuda.get_device_name(index) for index in range(device_count)],
    }


class TransitionCacheWriter:
    """Write bounded CPU shards; never retain activations on GPU between rows."""

    def __init__(
        self,
        *,
        cache_config: CacheConfig,
        split_registry: Mapping[str, Any],
        split_registry_path: str | Path,
        scientific_metadata: Mapping[str, Any],
        source_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        validate_split_registry(split_registry)
        self._torch = require_torch()
        self.cache_config = cache_config
        self.registry = dict(split_registry)
        self.registry_path = Path(split_registry_path)
        self.scientific_metadata = dict(scientific_metadata)
        self.source_metadata = dict(source_metadata or {})
        self.output_dir = cache_config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=False)

        self._metadata_buffer: list[dict[str, Any]] = []
        self._tensor_buffer: list[dict[str, Any]] = []
        self._buffer_bytes = 0
        self._shards: list[dict[str, Any]] = []
        self._num_records = 0
        self._problem_ids: set[str] = set()
        self._split_counts: Counter[str] = Counter()
        self._transition_keys: set[tuple[str, str, str]] = set()
        self._finalized = False

    def add(self, record: Mapping[str, Any]) -> None:
        if self._finalized:
            raise RuntimeError("Cannot add records after finalize()")
        metadata, tensors = normalize_transition_record(
            record,
            storage_dtype=self.cache_config.storage_dtype,
            delta_atol=self.cache_config.delta_atol,
            delta_rtol=self.cache_config.delta_rtol,
        )
        expected_split = self.registry["problem_to_split"].get(metadata["problem_id"])
        if expected_split is None:
            raise SplitLeakageError(f"Unknown problem_id: {metadata['problem_id']}")
        if metadata["split"] != expected_split:
            raise SplitLeakageError(
                f"problem_id {metadata['problem_id']!r} must be in {expected_split!r}, "
                f"not {metadata['split']!r}"
            )
        key = (
            metadata["problem_id"],
            metadata["trajectory_id"],
            metadata["transition_id"],
        )
        if key in self._transition_keys:
            raise TransitionSchemaError(f"Duplicate transition key: {key}")

        record_bytes = record_tensor_nbytes(tensors)
        if (
            self.cache_config.max_bytes_per_shard is not None
            and record_bytes > self.cache_config.max_bytes_per_shard
        ):
            raise TransitionSchemaError(
                f"One record requires {record_bytes} tensor bytes, exceeding "
                f"max_bytes_per_shard={self.cache_config.max_bytes_per_shard}"
            )
        would_exceed_count = (
            len(self._metadata_buffer) >= self.cache_config.max_records_per_shard
        )
        would_exceed_bytes = (
            self.cache_config.max_bytes_per_shard is not None
            and self._metadata_buffer
            and self._buffer_bytes + record_bytes > self.cache_config.max_bytes_per_shard
        )
        if would_exceed_count or would_exceed_bytes:
            self._flush_shard()

        self._metadata_buffer.append(metadata)
        self._tensor_buffer.append(tensors)
        self._buffer_bytes += record_bytes
        self._transition_keys.add(key)
        self._problem_ids.add(metadata["problem_id"])
        self._split_counts[metadata["split"]] += 1
        self._num_records += 1

        if (
            len(self._metadata_buffer) >= self.cache_config.max_records_per_shard
            or (
                self.cache_config.max_bytes_per_shard is not None
                and self._buffer_bytes >= self.cache_config.max_bytes_per_shard
            )
        ):
            self._flush_shard()

    def _flush_shard(self) -> None:
        if not self._metadata_buffer:
            return
        shard_index = len(self._shards)
        stem = f"shard-{shard_index:05d}"
        tensor_path = self.output_dir / f"{stem}.pt"
        metadata_path = self.output_dir / f"{stem}.metadata.jsonl"

        tensor_payload = {
            field: self._torch.stack([row[field] for row in self._tensor_buffer], dim=0)
            for field in TENSOR_FIELDS
        }
        self._torch.save(tensor_payload, tensor_path)
        with metadata_path.open("x", encoding="utf-8") as handle:
            for row in self._metadata_buffer:
                handle.write(json.dumps(row, sort_keys=True))
                handle.write("\n")

        self._shards.append(
            {
                "index": shard_index,
                "num_records": len(self._metadata_buffer),
                "tensor_file": tensor_path.name,
                "tensor_sha256": file_sha256(tensor_path),
                "metadata_file": metadata_path.name,
                "metadata_sha256": file_sha256(metadata_path),
                "tensor_shapes": {
                    field: list(value.shape) for field, value in tensor_payload.items()
                },
                "tensor_dtypes": {
                    field: str(value.dtype).removeprefix("torch.")
                    for field, value in tensor_payload.items()
                },
            }
        )
        self._metadata_buffer.clear()
        self._tensor_buffer.clear()
        self._buffer_bytes = 0

    def finalize(self) -> Path:
        if self._finalized:
            raise RuntimeError("finalize() may only be called once")
        self._flush_shard()
        if self._num_records == 0:
            raise CacheValidationError("Cannot finalize an empty transition cache")

        manifest = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "immutable": True,
            "source": self.source_metadata,
            "environment": runtime_environment(),
            "split_registry": {
                "path": str(self.registry_path),
                "sha256": registry_sha256(self.registry),
                "unit": self.registry["unit"],
                "seed": self.registry["seed"],
                "ratios": self.registry["ratios"],
            },
            **self.scientific_metadata,
            "cache_config": {
                "max_records_per_shard": self.cache_config.max_records_per_shard,
                "max_bytes_per_shard": self.cache_config.max_bytes_per_shard,
                "storage_dtype": self.cache_config.storage_dtype,
                "zero_norm_epsilon": self.cache_config.zero_norm_epsilon,
                "delta_atol": self.cache_config.delta_atol,
                "delta_rtol": self.cache_config.delta_rtol,
            },
            "schema": {
                "metadata_fields": list(METADATA_FIELDS),
                "tensor_fields": list(TENSOR_FIELDS),
                "tensor_semantics": {
                    "h_departure": "h_t^ell",
                    "h_arrival": "h_{t+1}^ell",
                    "delta": "h_arrival - h_departure",
                    "g_arrival": "gradient of answer margin at h_arrival",
                    "baseline_margin": "unmodified answer margin",
                },
            },
            "totals": {
                "num_records": self._num_records,
                "num_problems": len(self._problem_ids),
                "records_by_split": dict(sorted(self._split_counts.items())),
                "num_shards": len(self._shards),
            },
            "shards": self._shards,
        }
        manifest_path = self.output_dir / "manifest.json"
        with manifest_path.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        self._finalized = True
        return manifest_path


def _read_metadata(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise CacheValidationError(f"Metadata line {line_number} is not an object")
            rows.append(value)
    return rows


def validate_transition_cache(
    manifest_path: str | Path,
    *,
    split_registry: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate shards sequentially and return a compact CPU-side profile."""
    torch = require_torch()
    manifest_path = Path(manifest_path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("immutable") is not True:
        raise CacheValidationError("Manifest must declare immutable=true")
    validate_split_registry(split_registry)
    expected_registry_hash = registry_sha256(split_registry)
    if manifest.get("split_registry", {}).get("sha256") != expected_registry_hash:
        raise CacheValidationError("Split registry checksum mismatch")

    config = manifest.get("cache_config", {})
    delta_atol = float(config.get("delta_atol", 1e-5))
    delta_rtol = float(config.get("delta_rtol", 1e-4))
    zero_norm_epsilon = float(config.get("zero_norm_epsilon", 1e-12))
    all_metadata: list[dict[str, Any]] = []
    missing_counts: Counter[str] = Counter()
    boundary_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    dtype_by_field: dict[str, set[str]] = {field: set() for field in TENSOR_FIELDS}
    metadata_types: dict[str, set[str]] = {field: set() for field in METADATA_FIELDS}
    shape_suffix_by_field: dict[str, set[tuple[int, ...]]] = {
        field: set() for field in TENSOR_FIELDS
    }
    zero_norm_delta_count = 0
    total_records = 0

    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise CacheValidationError("Manifest contains no shards")
    for shard in shards:
        tensor_path = manifest_path.parent / shard["tensor_file"]
        metadata_path = manifest_path.parent / shard["metadata_file"]
        if file_sha256(tensor_path) != shard["tensor_sha256"]:
            raise CacheValidationError(f"Tensor checksum mismatch: {tensor_path}")
        if file_sha256(metadata_path) != shard["metadata_sha256"]:
            raise CacheValidationError(f"Metadata checksum mismatch: {metadata_path}")

        metadata_rows = _read_metadata(metadata_path)
        tensors = _torch_load_cpu(tensor_path)
        if set(tensors) != set(TENSOR_FIELDS):
            raise CacheValidationError(
                f"Tensor fields mismatch in {tensor_path}: {sorted(tensors)}"
            )
        expected_rows = int(shard["num_records"])
        if len(metadata_rows) != expected_rows:
            raise CacheValidationError(f"Metadata row count mismatch in {metadata_path}")

        for field in TENSOR_FIELDS:
            tensor = tensors[field]
            if tensor.device.type != "cpu":
                raise CacheValidationError(f"{field} was not loaded on CPU")
            if tensor.shape[0] != expected_rows:
                raise CacheValidationError(f"Leading dimension mismatch for {field}")
            if not bool(torch.isfinite(tensor).all()):
                raise CacheValidationError(f"NaN/Inf detected in {field}")
            dtype_by_field[field].add(str(tensor.dtype).removeprefix("torch."))
            shape_suffix_by_field[field].add(tuple(int(v) for v in tensor.shape[1:]))

        try:
            torch.testing.assert_close(
                tensors["delta"].float(),
                (tensors["h_arrival"] - tensors["h_departure"]).float(),
                atol=delta_atol,
                rtol=delta_rtol,
            )
        except AssertionError as exc:
            raise CacheValidationError(
                f"delta identity failed in shard {shard['index']}"
            ) from exc

        zero_norm_delta_count += int(
            (torch.linalg.vector_norm(tensors["delta"].float(), dim=-1) <= zero_norm_epsilon)
            .sum()
            .item()
        )
        for row in metadata_rows:
            for field in METADATA_FIELDS:
                value = row.get(field)
                if value is None or value == "":
                    missing_counts[field] += 1
                else:
                    metadata_types[field].add(type(value).__name__)
            boundary_counts[str(row.get("boundary_class"))] += 1
            split_counts[str(row.get("split"))] += 1
        all_metadata.extend(metadata_rows)
        total_records += expected_rows
        del tensors

    try:
        assert_rows_respect_registry(all_metadata, split_registry)
    except SplitLeakageError as exc:
        raise CacheValidationError(str(exc)) from exc
    if total_records != int(manifest.get("totals", {}).get("num_records", -1)):
        raise CacheValidationError("Manifest total record count mismatch")

    unique_problem_ids = {row["problem_id"] for row in all_metadata}
    return {
        "manifest_path": str(manifest_path),
        "source": manifest.get("source", {}),
        "model": manifest.get("model", {}),
        "margin": manifest.get("margin", {}),
        "num_records": total_records,
        "num_problems": len(unique_problem_ids),
        "num_trajectories": len({row["trajectory_id"] for row in all_metadata}),
        "num_shards": len(shards),
        "records_by_split": dict(sorted(split_counts.items())),
        "boundary_counts": dict(sorted(boundary_counts.items())),
        "tensor_dtypes": {
            field: sorted(values) for field, values in dtype_by_field.items()
        },
        "tensor_shape_suffixes": {
            field: [list(shape) for shape in sorted(values)]
            for field, values in shape_suffix_by_field.items()
        },
        "metadata_types": {
            field: sorted(values) for field, values in metadata_types.items()
        },
        "missing_counts": {
            field: int(missing_counts[field]) for field in METADATA_FIELDS
        },
        "missing_rates": {
            field: (missing_counts[field] / total_records if total_records else 0.0)
            for field in METADATA_FIELDS
        },
        "zero_norm_delta_count": zero_norm_delta_count,
        "split_leakage_check": "PASS",
        "delta_identity_check": "PASS",
        "finite_tensor_check": "PASS",
        "checksum_check": "PASS",
    }
