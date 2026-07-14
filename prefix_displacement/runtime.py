from __future__ import annotations

import json
import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

from .schema import require_torch


def seed_everything(seed: int) -> None:
    torch = require_torch()
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unwrap_model(model: Any) -> Any:
    torch = require_torch()
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def primary_device() -> Any:
    torch = require_torch()
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def prepare_data_parallel(model: Any) -> tuple[Any, Any, list[int]]:
    """Move to cuda:0 first, then wrap only when more than one GPU exists."""
    torch = require_torch()
    device = primary_device()
    model = model.to(device)
    device_ids = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(
            model,
            device_ids=device_ids,
            output_device=0,
        )
    return model, device, device_ids


def resolve_precision(requested: str = "auto") -> tuple[str, Any]:
    torch = require_torch()
    if not torch.cuda.is_available():
        return "float32", torch.float32
    def all_visible_gpus_support_bf16() -> bool:
        checker = getattr(torch.cuda, "is_bf16_supported", None)
        if checker is None or torch.cuda.device_count() < 1:
            return False
        # DataParallel replicates onto every visible device.  A GPU-0-only
        # capability check is unsafe on heterogeneous servers.
        for index in range(torch.cuda.device_count()):
            with torch.cuda.device(index):
                if not bool(checker()):
                    return False
        return True

    bf16_supported = all_visible_gpus_support_bf16()
    if requested == "auto":
        if bf16_supported:
            return "bfloat16", torch.bfloat16
        return "float16", torch.float16
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if requested not in mapping:
        raise ValueError("precision must be auto, bfloat16, float16, or float32")
    if requested == "bfloat16" and not bf16_supported:
        return "float16", torch.float16
    return requested, mapping[requested]


def autocast_context(device: Any, dtype: Any):
    torch = require_torch()
    if device.type != "cuda" or dtype == torch.float32:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def state_dict_without_parallel_prefix(model: Any) -> dict[str, Any]:
    return unwrap_model(model).state_dict()


def _strip_module_prefix(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    if state_dict and all(str(key).startswith("module.") for key in state_dict):
        return {str(key)[7:]: value for key, value in state_dict.items()}
    return dict(state_dict)


def load_model_state(model: Any, state_dict: Mapping[str, Any], *, strict: bool = True):
    return unwrap_model(model).load_state_dict(_strip_module_prefix(state_dict), strict=strict)


def save_training_checkpoint(
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any | None,
    epoch: int,
    global_step: int,
    config: Mapping[str, Any],
    checkpoint_path: str | Path,
) -> Path:
    torch = require_torch()
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = model.module.state_dict() if isinstance(
        model, torch.nn.DataParallel
    ) else model.state_dict()
    torch.save(
        {
            "model": state_dict,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "global_step": global_step,
            "config": dict(config),
        },
        path,
    )
    return path


def load_training_checkpoint(
    path: str | Path,
    *,
    model: Any,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    torch = require_torch()
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    load_model_state(model, payload["model"], strict=strict)
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    return payload


def batch_size_metadata(per_device_batch_size: int) -> dict[str, int]:
    torch = require_torch()
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    effective_devices = max(gpu_count, 1)
    return {
        "gpu_count": gpu_count,
        "per_device_batch_size": per_device_batch_size,
        "global_batch_size": per_device_batch_size * effective_devices,
    }


def gpu_memory_snapshot(batch_index: int) -> dict[str, Any]:
    torch = require_torch()
    devices = []
    for index in range(torch.cuda.device_count()):
        devices.append(
            {
                "device": index,
                "name": torch.cuda.get_device_name(index),
                "allocated_bytes": int(torch.cuda.memory_allocated(index)),
                "reserved_bytes": int(torch.cuda.memory_reserved(index)),
                "max_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
            }
        )
    return {"batch_index": batch_index, "devices": devices}


def write_json_exclusive(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output
