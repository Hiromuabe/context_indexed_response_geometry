from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .utils import atomic_write_json, file_sha256


# Version 2 requires immutable prefix/candidate snapshots plus candidate and
# prefix-token embedding controls. Version 1 caches must be regenerated rather
# than being silently interpreted under the stronger analysis contract.
BRANCH_SCHEMA_VERSION = 2


class StorageIntegrityError(RuntimeError):
    """Raised when a cache is incomplete, reordered, or corrupted."""


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "NumPy is required in the existing experiment environment; no package "
            "installation was attempted"
        ) from exc
    return np


def _write_npy_atomic(path: Path, array: Any) -> dict[str, Any]:
    np = _numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(array), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    loaded = np.load(path, mmap_mode="r", allow_pickle=False)
    metadata = {
        "path": str(path),
        "shape": list(loaded.shape),
        "dtype": str(loaded.dtype),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }
    del loaded
    return metadata


def _relative_file_metadata(base: Path, metadata: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(metadata)
    result["path"] = str(Path(str(metadata["path"])).relative_to(base))
    return result


class BranchStoreWriter:
    """Resumable immutable writer for chunked ``[prefix, candidate, hidden]`` cubes."""

    def __init__(self, output_dir: str | Path, header: Mapping[str, Any]) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.output_dir / "manifest.json"
        self.partial_path = self.output_dir / "manifest.partial.json"
        required = (
            "run_fingerprint",
            "n_prefixes",
            "n_candidates",
            "hidden_size",
            "layer_ids",
            "storage_dtype",
            "prefix_ids",
            "problem_ids",
            "problem_groups",
            "candidate_token_ids",
            "candidate_set_hash",
        )
        missing = [key for key in required if key not in header]
        if missing:
            raise StorageIntegrityError(f"Branch manifest header missing {missing}")
        if self.manifest_path.exists():
            existing = _read_json(self.manifest_path)
            self._assert_compatible(existing, header)
            if not existing.get("complete"):
                raise StorageIntegrityError("Final manifest exists but is not complete")
            self.manifest = existing
            return
        if self.partial_path.exists():
            existing = _read_json(self.partial_path)
            self._assert_compatible(existing, header)
            self._validate_recorded_files(existing)
            self.manifest = existing
        else:
            self.manifest = {
                "schema_version": BRANCH_SCHEMA_VERSION,
                "artifact_type": "prefix_successor_branch_cache",
                **dict(header),
                "layout": {
                    "branch_hidden": "[prefix, candidate, hidden]",
                    "prefix_hidden": "[prefix, hidden]",
                    "candidate_statistics": "[prefix, candidate]",
                    "candidate_axis": "common ordered candidate_token_ids",
                    "layer_convention": "zero_based_decoder_block_output_resid_post",
                    "probability_stratum_codes": {
                        "0": "low",
                        "1": "medium",
                        "2": "high",
                    },
                },
                "chunks": [],
                "complete": False,
            }
            atomic_write_json(self.partial_path, self.manifest)

    @staticmethod
    def _assert_compatible(existing: Mapping[str, Any], header: Mapping[str, Any]) -> None:
        for key in (
            "run_fingerprint",
            "n_prefixes",
            "n_candidates",
            "hidden_size",
            "layer_ids",
            "storage_dtype",
            "candidate_set_hash",
            "prefix_ids",
            "candidate_token_ids",
        ):
            if existing.get(key) != header.get(key):
                raise StorageIntegrityError(
                    f"Cannot resume branch cache: manifest {key} differs"
                )

    def _validate_recorded_files(self, manifest: Mapping[str, Any]) -> None:
        """Verify every file recorded by an interrupted writer before skipping it."""

        np = _numpy()
        metadata_entries = []
        for chunk in manifest.get("chunks", []):
            metadata_entries.extend(chunk.get("files", {}).values())
        metadata_entries.extend(manifest.get("global_files", {}).values())
        for metadata in metadata_entries:
            path = self.output_dir / str(metadata["path"])
            if not path.is_file():
                raise StorageIntegrityError(
                    f"Cannot resume branch cache; recorded file is missing: {path}"
                )
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            try:
                if list(array.shape) != list(metadata["shape"]):
                    raise StorageIntegrityError(
                        f"Cannot resume branch cache; shape mismatch: {path}"
                    )
                if str(array.dtype) != str(metadata["dtype"]):
                    raise StorageIntegrityError(
                        f"Cannot resume branch cache; dtype mismatch: {path}"
                    )
            finally:
                del array
            if file_sha256(path) != metadata.get("sha256"):
                raise StorageIntegrityError(
                    f"Cannot resume branch cache; checksum mismatch: {path}"
                )

    @property
    def complete(self) -> bool:
        return bool(self.manifest.get("complete"))

    @property
    def completed_ranges(self) -> set[tuple[int, int]]:
        return {
            (int(chunk["prefix_start"]), int(chunk["prefix_stop"]))
            for chunk in self.manifest.get("chunks", [])
        }

    def write_snapshots(
        self,
        *,
        prefix_records: Sequence[Mapping[str, Any]],
        candidate_payload: Mapping[str, Any],
    ) -> None:
        prefix_path = self.output_dir / "prefix_metadata.jsonl"
        candidate_path = self.output_dir / "candidate_metadata.json"
        if not self.complete:
            temporary = prefix_path.with_name(
                f".{prefix_path.name}.tmp-{os.getpid()}"
            )
            with temporary.open("w", encoding="utf-8") as handle:
                for row in prefix_records:
                    handle.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=False))
                    handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, prefix_path)
            atomic_write_json(candidate_path, candidate_payload)
        elif not prefix_path.exists() or not candidate_path.exists():
            raise StorageIntegrityError("Completed cache is missing metadata snapshots")
        self.manifest["metadata_files"] = {
            "prefix_metadata": {
                "path": prefix_path.name,
                "sha256": file_sha256(prefix_path),
            },
            "candidate_metadata": {
                "path": candidate_path.name,
                "sha256": file_sha256(candidate_path),
            },
        }
        if not self.complete:
            atomic_write_json(self.partial_path, self.manifest)

    def write_candidate_input_embeddings(self, embeddings: Any) -> None:
        """Save the common candidates' frozen input embeddings as a small control."""

        np = _numpy()
        if self.complete:
            return
        values = np.asarray(embeddings)
        expected = (
            int(self.manifest["n_candidates"]),
            int(self.manifest["hidden_size"]),
        )
        if values.shape != expected:
            raise StorageIntegrityError(
                f"Candidate embedding shape {values.shape}, expected {expected}"
            )
        if not np.isfinite(values).all():
            raise StorageIntegrityError("Candidate input embeddings contain NaN or infinity")
        dtype = np.dtype(str(self.manifest["storage_dtype"]))
        metadata = _write_npy_atomic(
            self.output_dir / "global" / "candidate_input_embeddings.npy",
            values.astype(dtype, copy=False),
        )
        self.manifest.setdefault("global_files", {})[
            "candidate_input_embeddings"
        ] = _relative_file_metadata(self.output_dir, metadata)
        atomic_write_json(self.partial_path, self.manifest)

    def write_prefix_last_token_embeddings(self, embeddings: Any) -> None:
        """Save each ordered prefix's last-token input embedding [P,H]."""

        np = _numpy()
        if self.complete:
            return
        values = np.asarray(embeddings)
        expected = (
            int(self.manifest["n_prefixes"]),
            int(self.manifest["hidden_size"]),
        )
        if values.shape != expected:
            raise StorageIntegrityError(
                f"Prefix last-token embedding shape {values.shape}, expected {expected}"
            )
        if not np.isfinite(values).all():
            raise StorageIntegrityError(
                "Prefix last-token input embeddings contain NaN or infinity"
            )
        dtype = np.dtype(str(self.manifest["storage_dtype"]))
        metadata = _write_npy_atomic(
            self.output_dir / "global" / "prefix_last_token_embeddings.npy",
            values.astype(dtype, copy=False),
        )
        self.manifest.setdefault("global_files", {})[
            "prefix_last_token_embeddings"
        ] = _relative_file_metadata(self.output_dir, metadata)
        atomic_write_json(self.partial_path, self.manifest)

    def write_chunk(
        self,
        *,
        prefix_start: int,
        prefix_stop: int,
        prefix_indices: Any,
        branch_hidden_by_layer: Mapping[int, Any],
        prefix_hidden_by_layer: Mapping[int, Any],
        candidate_logprob: Any,
        candidate_probability: Any,
        probability_stratum: Any,
        prefix_entropy: Any,
    ) -> dict[str, Any]:
        np = _numpy()
        key = (int(prefix_start), int(prefix_stop))
        if key in self.completed_ranges:
            return next(
                chunk
                for chunk in self.manifest["chunks"]
                if (chunk["prefix_start"], chunk["prefix_stop"]) == key
            )
        if self.complete:
            raise StorageIntegrityError("Cannot append to a completed branch cache")
        n_prefix = prefix_stop - prefix_start
        n_candidates = int(self.manifest["n_candidates"])
        hidden_size = int(self.manifest["hidden_size"])
        expected_layers = list(map(int, self.manifest["layer_ids"]))
        if prefix_stop <= prefix_start or prefix_start < 0:
            raise StorageIntegrityError("Invalid prefix chunk range")
        if list(map(int, branch_hidden_by_layer)) != expected_layers:
            raise StorageIntegrityError("Branch layer order differs from manifest")
        if list(map(int, prefix_hidden_by_layer)) != expected_layers:
            raise StorageIntegrityError("Prefix-hidden layer order differs from manifest")
        indices = np.asarray(prefix_indices)
        if indices.shape != (n_prefix,) or not np.array_equal(
            indices, np.arange(prefix_start, prefix_stop)
        ):
            raise StorageIntegrityError("Chunk prefix indices are not contiguous and ordered")
        expected_matrix = (n_prefix, n_candidates)
        for name, value in (
            ("candidate_logprob", candidate_logprob),
            ("candidate_probability", candidate_probability),
            ("probability_stratum", probability_stratum),
        ):
            array = np.asarray(value)
            if array.shape != expected_matrix:
                raise StorageIntegrityError(
                    f"{name} has shape {array.shape}, expected {expected_matrix}"
                )
            if not np.isfinite(array).all():
                raise StorageIntegrityError(f"{name} contains NaN or infinity")
        probabilities = np.asarray(candidate_probability)
        if bool((probabilities < 0).any()) or bool((probabilities > 1).any()):
            raise StorageIntegrityError("Candidate probabilities must lie in [0,1]")
        strata = np.asarray(probability_stratum)
        if not np.isin(strata, [0, 1, 2]).all():
            raise StorageIntegrityError("Probability stratum codes must be 0, 1, or 2")
        entropy_values = np.asarray(prefix_entropy)
        if entropy_values.shape != (n_prefix,):
            raise StorageIntegrityError("prefix_entropy has an unexpected shape")
        if not np.isfinite(entropy_values).all():
            raise StorageIntegrityError("prefix_entropy contains NaN or infinity")
        for layer_id in expected_layers:
            branch = np.asarray(branch_hidden_by_layer[layer_id])
            prefix = np.asarray(prefix_hidden_by_layer[layer_id])
            if branch.shape != (n_prefix, n_candidates, hidden_size):
                raise StorageIntegrityError(
                    f"Layer {layer_id} branch shape {branch.shape} is invalid"
                )
            if prefix.shape != (n_prefix, hidden_size):
                raise StorageIntegrityError(
                    f"Layer {layer_id} prefix shape {prefix.shape} is invalid"
                )
            if not np.isfinite(branch).all() or not np.isfinite(prefix).all():
                raise StorageIntegrityError(f"Layer {layer_id} contains NaN or infinity")
        chunk_index = len(self.manifest["chunks"])
        directory = self.output_dir / "chunks" / f"chunk_{chunk_index:05d}"
        directory.mkdir(parents=True, exist_ok=True)
        files: dict[str, Any] = {}

        def save(name: str, array: Any) -> None:
            metadata = _write_npy_atomic(directory / f"{name}.npy", array)
            files[name] = _relative_file_metadata(self.output_dir, metadata)

        save("prefix_indices", indices.astype(np.int64, copy=False))
        save("candidate_logprob", np.asarray(candidate_logprob, dtype=np.float32))
        save("candidate_probability", np.asarray(candidate_probability, dtype=np.float32))
        save("probability_stratum", np.asarray(probability_stratum, dtype=np.int8))
        save("prefix_entropy", np.asarray(prefix_entropy, dtype=np.float32))
        storage_dtype = np.dtype(str(self.manifest["storage_dtype"]))
        for layer_id in expected_layers:
            save(
                f"branch_hidden_layer_{layer_id}",
                np.asarray(branch_hidden_by_layer[layer_id], dtype=storage_dtype),
            )
            save(
                f"prefix_hidden_layer_{layer_id}",
                np.asarray(prefix_hidden_by_layer[layer_id], dtype=storage_dtype),
            )
        chunk = {
            "chunk_index": chunk_index,
            "prefix_start": int(prefix_start),
            "prefix_stop": int(prefix_stop),
            "n_prefixes": n_prefix,
            "files": files,
        }
        self.manifest["chunks"].append(chunk)
        self.manifest["chunks"].sort(key=lambda item: item["prefix_start"])
        atomic_write_json(self.partial_path, self.manifest)
        return chunk

    def finalize(self) -> Path:
        if self.complete:
            return self.manifest_path
        if not isinstance(self.manifest.get("metadata_files"), Mapping):
            raise StorageIntegrityError("Cannot finalize without metadata snapshots")
        expected_start = 0
        for chunk in sorted(self.manifest["chunks"], key=lambda item: item["prefix_start"]):
            if int(chunk["prefix_start"]) != expected_start:
                raise StorageIntegrityError(
                    f"Missing prefix chunk before index {chunk['prefix_start']}"
                )
            expected_start = int(chunk["prefix_stop"])
        if expected_start != int(self.manifest["n_prefixes"]):
            raise StorageIntegrityError(
                f"Cache ends at prefix {expected_start}, expected {self.manifest['n_prefixes']}"
            )
        self.manifest["complete"] = True
        self.manifest["n_chunks"] = len(self.manifest["chunks"])
        atomic_write_json(self.manifest_path, self.manifest)
        return self.manifest_path


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise StorageIntegrityError(f"Manifest must be a JSON object: {path}")
    return value


def load_manifest(path_or_directory: str | Path) -> dict[str, Any]:
    path = Path(path_or_directory)
    if path.is_dir():
        path = path / "manifest.json"
    manifest = _read_json(path)
    if manifest.get("schema_version") != BRANCH_SCHEMA_VERSION:
        raise StorageIntegrityError(
            f"Unsupported branch schema version: {manifest.get('schema_version')}"
        )
    if not manifest.get("complete"):
        raise StorageIntegrityError(f"Branch cache is incomplete: {path}")
    manifest["_manifest_path"] = str(path.resolve())
    return manifest


def _manifest_and_base(
    manifest_or_path: Mapping[str, Any] | str | Path,
) -> tuple[Mapping[str, Any], Path]:
    if isinstance(manifest_or_path, Mapping):
        manifest = manifest_or_path
        marker = manifest.get("_manifest_path")
        if marker is None:
            raise StorageIntegrityError(
                "A manifest mapping must come from load_manifest so relative paths resolve"
            )
        base = Path(str(marker)).parent
    else:
        loaded = load_manifest(manifest_or_path)
        manifest, base = loaded, Path(loaded["_manifest_path"]).parent
    return manifest, base


def _load_file(base: Path, metadata: Mapping[str, Any], mmap_mode: str | None) -> Any:
    np = _numpy()
    path = base / str(metadata["path"])
    array = np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
    if list(array.shape) != list(metadata["shape"]) or str(array.dtype) != metadata["dtype"]:
        raise StorageIntegrityError(f"Shape/dtype mismatch for {path}")
    return array


def iter_layer_chunks(
    manifest_or_path: Mapping[str, Any] | str | Path,
    layer_id: int,
    *,
    mmap_mode: str | None = "r",
) -> Iterator[dict[str, Any]]:
    """Iterate memory-mapped layer chunks without materializing the full cube."""

    manifest, base = _manifest_and_base(manifest_or_path)
    layer_id = int(layer_id)
    if layer_id not in list(map(int, manifest["layer_ids"])):
        raise KeyError(f"Layer {layer_id} is not in the branch cache")
    for chunk in sorted(manifest["chunks"], key=lambda item: item["prefix_start"]):
        files = chunk["files"]
        branch = _load_file(base, files[f"branch_hidden_layer_{layer_id}"], mmap_mode)
        prefix = _load_file(base, files[f"prefix_hidden_layer_{layer_id}"], mmap_mode)
        yield {
            "prefix_start": int(chunk["prefix_start"]),
            "prefix_stop": int(chunk["prefix_stop"]),
            "prefix_indices": _load_file(base, files["prefix_indices"], mmap_mode),
            "branches": branch,
            "z": branch,
            "prefix_hidden": prefix,
            "h_prefix": prefix,
            "candidate_logprob": _load_file(
                base, files["candidate_logprob"], mmap_mode
            ),
            "candidate_probability": _load_file(
                base, files["candidate_probability"], mmap_mode
            ),
            "probability_stratum": _load_file(
                base, files["probability_stratum"], mmap_mode
            ),
            "prefix_entropy": _load_file(base, files["prefix_entropy"], mmap_mode),
        }


def load_layer_arrays(
    manifest_or_path: Mapping[str, Any] | str | Path,
    layer_id: int,
    *,
    output_dtype: str | None = "float32",
) -> dict[str, Any]:
    """Load and concatenate one layer; intended for smoke/pilot analyses."""

    np = _numpy()
    chunks = list(iter_layer_chunks(manifest_or_path, layer_id, mmap_mode="r"))
    if not chunks:
        raise StorageIntegrityError("Branch cache has no chunks")
    result = {
        key: np.concatenate([np.asarray(chunk[key]) for chunk in chunks], axis=0)
        for key in (
            "prefix_indices",
            "branches",
            "prefix_hidden",
            "candidate_logprob",
            "candidate_probability",
            "probability_stratum",
            "prefix_entropy",
        )
    }
    if output_dtype is not None:
        result["branches"] = result["branches"].astype(output_dtype, copy=False)
        result["prefix_hidden"] = result["prefix_hidden"].astype(output_dtype, copy=False)
    result["z"] = result["branches"]
    result["h_prefix"] = result["prefix_hidden"]
    return result


def materialize_layer_memmap(
    manifest_or_path: Mapping[str, Any] | str | Path,
    layer_id: int,
    destination: str | Path,
    *,
    dtype: str = "float32",
) -> Path:
    """Construct a single ``.npy`` mmap cube [P,K,H] from verified chunks."""

    np = _numpy()
    manifest, _base = _manifest_and_base(manifest_or_path)
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    shape = (
        int(manifest["n_prefixes"]),
        int(manifest["n_candidates"]),
        int(manifest["hidden_size"]),
    )
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    cube = np.lib.format.open_memmap(temporary, mode="w+", dtype=dtype, shape=shape)
    for chunk in iter_layer_chunks(manifest, layer_id, mmap_mode="r"):
        cube[chunk["prefix_start"] : chunk["prefix_stop"]] = chunk["branches"]
    cube.flush()
    del cube
    os.replace(temporary, path)
    return path


def validate_manifest_files(
    manifest_or_path: Mapping[str, Any] | str | Path, *, verify_checksums: bool = True
) -> None:
    manifest, base = _manifest_and_base(manifest_or_path)
    metadata_files = manifest.get("metadata_files")
    if not isinstance(metadata_files, Mapping):
        raise StorageIntegrityError("Branch manifest is missing metadata snapshots")
    for required_name in ("prefix_metadata", "candidate_metadata"):
        metadata = metadata_files.get(required_name)
        if not isinstance(metadata, Mapping) or not metadata.get("path") or not metadata.get("sha256"):
            raise StorageIntegrityError(f"Missing {required_name} snapshot metadata")
        path = base / str(metadata["path"])
        if not path.is_file():
            raise StorageIntegrityError(f"Missing metadata snapshot: {path}")
        if file_sha256(path) != metadata["sha256"]:
            raise StorageIntegrityError(f"Metadata snapshot checksum mismatch: {path}")
    global_files = manifest.get("global_files", {})
    for required_name in manifest.get("required_global_files", []):
        if required_name not in global_files:
            raise StorageIntegrityError(f"Missing required global file: {required_name}")
    if "candidate_input_embeddings" in global_files:
        embedding_metadata = global_files["candidate_input_embeddings"]
        embedding_path = base / str(embedding_metadata["path"])
        if not embedding_path.is_file():
            raise StorageIntegrityError(f"Missing candidate input embeddings: {embedding_path}")
        if file_sha256(embedding_path) != embedding_metadata["sha256"]:
            raise StorageIntegrityError(
                f"Candidate input embedding checksum mismatch: {embedding_path}"
            )
        expected_embedding_shape = [
            int(manifest["n_candidates"]),
            int(manifest["hidden_size"]),
        ]
        if list(embedding_metadata.get("shape", [])) != expected_embedding_shape:
            raise StorageIntegrityError("Candidate input embedding shape is inconsistent")
        embedding_array = _load_file(base, embedding_metadata, mmap_mode="r")
        del embedding_array
    if "prefix_last_token_embeddings" in global_files:
        prefix_embedding_metadata = global_files["prefix_last_token_embeddings"]
        prefix_embedding_path = base / str(prefix_embedding_metadata["path"])
        if not prefix_embedding_path.is_file():
            raise StorageIntegrityError(
                f"Missing prefix last-token input embeddings: {prefix_embedding_path}"
            )
        if file_sha256(prefix_embedding_path) != prefix_embedding_metadata["sha256"]:
            raise StorageIntegrityError(
                "Prefix last-token input embedding checksum mismatch: "
                f"{prefix_embedding_path}"
            )
        expected_prefix_embedding_shape = [
            int(manifest["n_prefixes"]),
            int(manifest["hidden_size"]),
        ]
        if (
            list(prefix_embedding_metadata.get("shape", []))
            != expected_prefix_embedding_shape
        ):
            raise StorageIntegrityError(
                "Prefix last-token input embedding shape is inconsistent"
            )
        prefix_embedding_array = _load_file(
            base, prefix_embedding_metadata, mmap_mode="r"
        )
        del prefix_embedding_array

    expected_start = 0
    for chunk in sorted(manifest["chunks"], key=lambda item: item["prefix_start"]):
        if int(chunk["prefix_start"]) != expected_start:
            raise StorageIntegrityError("Branch chunks are not contiguous")
        expected_start = int(chunk["prefix_stop"])
        for metadata in chunk["files"].values():
            path = base / metadata["path"]
            if not path.is_file():
                raise StorageIntegrityError(f"Missing branch cache file: {path}")
            array = _load_file(base, metadata, mmap_mode="r")
            del array
            if verify_checksums and file_sha256(path) != metadata["sha256"]:
                raise StorageIntegrityError(f"Checksum mismatch: {path}")
    if expected_start != int(manifest["n_prefixes"]):
        raise StorageIntegrityError("Branch chunks do not cover all prefixes")


def load_candidate_input_embeddings(
    manifest_or_path: Mapping[str, Any] | str | Path,
    *,
    mmap_mode: str | None = "r",
    output_dtype: str | None = "float32",
) -> Any:
    manifest, base = _manifest_and_base(manifest_or_path)
    global_files = manifest.get("global_files", {})
    metadata = global_files.get("candidate_input_embeddings")
    if not isinstance(metadata, Mapping):
        raise StorageIntegrityError("Branch cache has no candidate input embeddings")
    array = _load_file(base, metadata, mmap_mode)
    if output_dtype is not None:
        array = _numpy().asarray(array).astype(output_dtype, copy=False)
    return array


def load_prefix_last_token_embeddings(
    manifest_or_path: Mapping[str, Any] | str | Path,
    *,
    mmap_mode: str | None = "r",
    output_dtype: str | None = "float32",
) -> Any:
    manifest, base = _manifest_and_base(manifest_or_path)
    global_files = manifest.get("global_files", {})
    metadata = global_files.get("prefix_last_token_embeddings")
    if not isinstance(metadata, Mapping):
        raise StorageIntegrityError(
            "Branch cache has no prefix last-token input embeddings"
        )
    array = _load_file(base, metadata, mmap_mode)
    if output_dtype is not None:
        array = _numpy().asarray(array).astype(output_dtype, copy=False)
    return array


class BranchStore:
    """Small convenience facade used by downstream analysis stages."""

    def __init__(self, path_or_directory: str | Path) -> None:
        self.manifest = load_manifest(path_or_directory)

    def iter_layer(self, layer_id: int, *, mmap_mode: str | None = "r"):
        return iter_layer_chunks(self.manifest, layer_id, mmap_mode=mmap_mode)

    def load_layer(self, layer_id: int, *, output_dtype: str = "float32"):
        return load_layer_arrays(self.manifest, layer_id, output_dtype=output_dtype)
