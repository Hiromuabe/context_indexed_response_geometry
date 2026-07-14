from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import platform
from pathlib import Path
from typing import Any


def _json_default(value: Any) -> Any:
    """Convert NumPy scalars produced by diagnostics to JSON-native values."""
    try:
        import numpy as np
        if isinstance(value, np.generic):
            return value.item()
    except ImportError:
        pass
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def load_config(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML is required for non-JSON YAML configuration") from exc
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError("configuration root must be a mapping")
    return value


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: str | Path, value: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return destination


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL row {number} is not an object")
                rows.append(value)
    return rows


def atomic_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, default=_json_default) + "\n")
    os.replace(temporary, destination)
    return destination


def seed_all(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def result_root(config: dict[str, Any]) -> Path:
    return Path(str(config.get("results_root", "results/prefix_response_subspaces")))


def ensure_layout(config: dict[str, Any]) -> Path:
    root = result_root(config)
    for name in ("manifests", "prefix_pool", "matches", "candidate_tokens", "hidden_states", "residuals", "subspaces", "permutation", "functional", "metrics", "figures", "tables", "logs"):
        (root / name).mkdir(parents=True, exist_ok=True)
    snapshot = root / "manifests/config_resolved.yaml"
    serialized = json.dumps(config, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    snapshot_changed = snapshot.exists() and snapshot.read_text(encoding="utf-8") != serialized
    if snapshot_changed:
        completed_stage_manifests = []
        for candidate in (root / "manifests").glob("*.json"):
            if candidate.name == "metadata.json":
                continue
            try:
                if bool(read_json(candidate).get("complete")):
                    completed_stage_manifests.append(candidate)
            except (OSError, ValueError, AttributeError):
                completed_stage_manifests.append(candidate)
        if completed_stage_manifests:
            raise RuntimeError(
                f"resolved config snapshot mismatch: {snapshot}; completed artifacts "
                f"exist: {[str(path) for path in completed_stage_manifests]}"
            )
        snapshot.write_text(serialized, encoding="utf-8")
    elif not snapshot.exists():
        snapshot.write_text(serialized, encoding="utf-8")
    metadata_path = root / "manifests/metadata.json"
    if snapshot_changed and metadata_path.exists():
        metadata_path.unlink()
    if not metadata_path.exists():
        metadata: dict[str, Any] = {"config_hash": stable_hash(config), "seed": config.get("seed"), "git_commit": git_commit(), "python": platform.python_version(), "platform": platform.platform()}
        try:
            import numpy as np
            metadata["numpy"] = np.__version__
        except ImportError: metadata["numpy"] = "UNAVAILABLE"
        try:
            import torch
            metadata.update({"torch": torch.__version__, "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(), "gpu_count": torch.cuda.device_count(), "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})
        except ImportError: metadata["torch"] = "UNAVAILABLE"
        atomic_json(metadata_path, metadata)
    return root


def stage_is_complete(path: Path, config: dict[str, Any], inputs: dict[str, str] | None = None) -> bool:
    if not path.exists():
        return False
    manifest = read_json(path)
    if manifest.get("config_hash") != stable_hash(config):
        raise RuntimeError(f"existing stage has a different configuration: {path}")
    for key, value in (inputs or {}).items():
        if manifest.get(key) != value:
            raise RuntimeError(f"existing stage has mismatched {key}: {path}")
    return bool(manifest.get("complete"))
