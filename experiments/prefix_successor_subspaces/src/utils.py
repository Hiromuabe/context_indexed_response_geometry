from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class ConfigurationError(ValueError):
    """Raised when a run would require guessing a scientific condition."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load JSON or YAML without ever installing a missing parser.

    YAML is the public configuration format for this experiment.  JSON remains
    accepted because it is a YAML subset and is useful in minimal test
    environments.
    """

    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to read non-JSON YAML configs in the existing "
                "server environment; no package installation was attempted"
            ) from exc
        else:
            payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ConfigurationError(f"Configuration root must be a mapping: {config_path}")
    return payload


def require_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"Missing mapping config section: {key}")
    return value


def require_int(config: Mapping[str, Any], key: str, *, minimum: int = 0) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigurationError(f"{key} must be an integer >= {minimum}")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def model_request_metadata(
    config: Mapping[str, Any], cli_model_path: str | None = None
) -> dict[str, Any]:
    """Return the checkpoint request that must match reusable model artifacts."""

    from prefix_displacement.model_loading import resolve_model_source

    model = require_mapping(config, "model")
    source, loading_kwargs = resolve_model_source(model, cli_model_path)
    source_path = Path(source)
    normalized_source = (
        str(source_path.expanduser().resolve())
        if source_path.exists() or source.startswith(("/", ".", "~"))
        else source
    )
    request = {
        "source": normalized_source,
        "revision": loading_kwargs.get("revision", "LOCAL_DIRECTORY"),
        "local_files_only": bool(loading_kwargs.get("local_files_only", False)),
        "trust_remote_code": bool(loading_kwargs.get("trust_remote_code", False)),
    }
    return {**request, "request_hash": stable_hash(request)}


def file_sha256(path: str | Path, *, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}"
    )
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    return destination


def write_json_exclusive(path: str | Path, payload: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    return destination


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def git_commit(cwd: str | Path | None = None) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        from prefix_displacement.runtime import seed_everything as seed_runtime
    except ImportError:
        return
    try:
        seed_runtime(seed)
    except ImportError:
        # Data-only stages remain usable in a minimal environment.  Model stages
        # call require_torch explicitly and will still fail with a precise import.
        return


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path
    manifests: Path
    prefixes_jsonl: Path
    prefix_manifest: Path
    problem_splits: Path
    candidates_json: Path
    candidate_manifest: Path
    branches_dir: Path
    logs: Path


def artifact_paths(config: Mapping[str, Any]) -> ArtifactPaths:
    root_value = config.get("results_root")
    if not isinstance(root_value, str) or not root_value:
        raise ConfigurationError("results_root must be a non-empty path")
    root = Path(root_value)
    manifests = root / "manifests"
    data = require_mapping(config, "data")
    candidates = require_mapping(config, "candidates")
    extraction = require_mapping(config, "extraction")
    return ArtifactPaths(
        root=root,
        manifests=manifests,
        prefixes_jsonl=Path(data.get("prefixes_jsonl", manifests / "prefixes.jsonl")),
        prefix_manifest=Path(
            data.get("prefix_manifest", manifests / "prefixes_manifest.json")
        ),
        problem_splits=Path(
            data.get("problem_splits", manifests / "problem_splits.json")
        ),
        candidates_json=Path(
            candidates.get("output_json", manifests / "candidate_tokens.json")
        ),
        candidate_manifest=Path(
            candidates.get("manifest", manifests / "candidate_tokens_manifest.json")
        ),
        branches_dir=Path(
            extraction.get("output_dir", root / "hidden_states" / "branches")
        ),
        logs=root / "logs",
    )


def completed_artifact_matches(
    manifest_path: str | Path,
    *,
    config_hash: str,
    required_hashes: Mapping[str, str] | None = None,
) -> bool:
    path = Path(manifest_path)
    if not path.exists():
        return False
    manifest = read_json(path)
    if not isinstance(manifest, Mapping) or not manifest.get("complete"):
        return False
    if manifest.get("config_hash") != config_hash:
        raise RuntimeError(
            f"Completed artifact at {path} belongs to a different configuration"
        )
    for key, expected in (required_hashes or {}).items():
        if manifest.get(key) != expected:
            raise RuntimeError(
                f"Completed artifact at {path} has mismatched {key}"
            )
    return True


def fail_if_oom(error: RuntimeError, conditions: Mapping[str, Any]) -> None:
    if "out of memory" not in str(error).lower():
        raise error
    raise RuntimeError(
        "CUDA OOM; batch size, sequence length, candidate count, layer set, and "
        "data count were not changed automatically. Conditions: "
        + json.dumps(conditions, sort_keys=True)
    ) from error
