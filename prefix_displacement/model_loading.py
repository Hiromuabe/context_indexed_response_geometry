from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping


MODEL_PATH_ENV = "BLACK2026_MODEL_PATH"


def resolve_model_source(
    model_config: Mapping[str, Any], cli_model_path: str | None = None
) -> tuple[str, dict[str, Any]]:
    """Prefer an explicit local path, then an environment override, then config."""
    source = cli_model_path or os.environ.get(MODEL_PATH_ENV) or model_config["checkpoint"]
    source = str(Path(source).expanduser()) if source.startswith(("/", ".", "~")) else source
    local_directory = Path(source).is_dir()
    kwargs: dict[str, Any] = {
        "local_files_only": bool(model_config.get("local_files_only", True)),
        "trust_remote_code": bool(model_config.get("trust_remote_code", False)),
    }
    if local_directory:
        # revision applies to Hub repositories, not an already-resolved directory.
        kwargs["local_files_only"] = True
    else:
        kwargs["revision"] = model_config.get("revision", "main")
    return source, kwargs


def explain_model_load_failure(source: str, error: BaseException) -> RuntimeError:
    return RuntimeError(
        "Qwen checkpoint is not available locally and this server cannot reach the "
        "Hugging Face Hub. Point to the existing model directory with either "
        "`--model-path /absolute/path/to/Qwen2.5-Math-1.5B` or "
        f"`{MODEL_PATH_ENV}=/absolute/path/to/Qwen2.5-Math-1.5B`. "
        "The directory must contain config.json, tokenizer files, and model weights. "
        f"Attempted source: {source!r}. Original error: {error}"
    )
