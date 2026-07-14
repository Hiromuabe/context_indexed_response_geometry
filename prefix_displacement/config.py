from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


UNRESOLVED_VALUES = {None, "", "UNKNOWN", "REQUIRED_FROM_AUDIT"}


class ConfigError(ValueError):
    """Raised when a scientific setting is absent or inconsistent."""


def load_json_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ConfigError(f"Top-level config must be an object: {config_path}")
    return value


def require_resolved(value: Any, dotted_name: str) -> Any:
    try:
        unresolved = value in UNRESOLVED_VALUES
    except TypeError:
        unresolved = False
    if unresolved:
        raise ConfigError(
            f"{dotted_name} is unresolved; fill it from audited evidence before "
            "running a real cache build"
        )
    return value


@dataclass(frozen=True)
class SplitConfig:
    unit: str
    seed: int
    train_ratio: float
    dev_ratio: float
    test_ratio: float
    registry_path: Path


@dataclass(frozen=True)
class CacheConfig:
    output_dir: Path
    max_records_per_shard: int
    max_bytes_per_shard: int | None
    storage_dtype: str
    zero_norm_epsilon: float
    delta_atol: float
    delta_rtol: float


def parse_split_config(config: dict[str, Any]) -> SplitConfig:
    raw = config.get("split")
    if not isinstance(raw, dict):
        raise ConfigError("split must be an object")
    ratios = raw.get("ratios")
    if not isinstance(ratios, dict):
        raise ConfigError("split.ratios must be an object")

    unit = str(require_resolved(raw.get("unit"), "split.unit"))
    if unit != "gsm8k_problem_id":
        raise ConfigError(f"split.unit must be gsm8k_problem_id, got {unit!r}")

    seed_raw = require_resolved(raw.get("seed"), "split.seed")
    if isinstance(seed_raw, bool) or not isinstance(seed_raw, int):
        raise ConfigError("split.seed must be an integer")

    parsed_ratios: dict[str, float] = {}
    for name in ("train", "dev", "test"):
        raw_value = require_resolved(ratios.get(name), f"split.ratios.{name}")
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ConfigError(f"split.ratios.{name} must be numeric")
        parsed_ratios[name] = float(raw_value)
        if not 0.0 < parsed_ratios[name] < 1.0:
            raise ConfigError(f"split.ratios.{name} must be strictly between 0 and 1")

    ratio_sum = sum(parsed_ratios.values())
    if abs(ratio_sum - 1.0) > 1e-12:
        raise ConfigError(f"split ratios must sum to 1.0, got {ratio_sum:.17g}")

    registry_path = Path(
        str(require_resolved(raw.get("registry_path"), "split.registry_path"))
    )
    return SplitConfig(
        unit=unit,
        seed=seed_raw,
        train_ratio=parsed_ratios["train"],
        dev_ratio=parsed_ratios["dev"],
        test_ratio=parsed_ratios["test"],
        registry_path=registry_path,
    )


def parse_cache_config(config: dict[str, Any]) -> CacheConfig:
    raw = config.get("cache")
    if not isinstance(raw, dict):
        raise ConfigError("cache must be an object")

    max_records = raw.get("max_records_per_shard")
    if isinstance(max_records, bool) or not isinstance(max_records, int) or max_records <= 0:
        raise ConfigError("cache.max_records_per_shard must be a positive integer")

    max_bytes_raw = raw.get("max_bytes_per_shard")
    if max_bytes_raw is None:
        max_bytes = None
    elif isinstance(max_bytes_raw, bool) or not isinstance(max_bytes_raw, int) or max_bytes_raw <= 0:
        raise ConfigError("cache.max_bytes_per_shard must be null or a positive integer")
    else:
        max_bytes = max_bytes_raw

    storage_dtype = str(require_resolved(raw.get("storage_dtype"), "cache.storage_dtype"))
    if storage_dtype not in {"float16", "bfloat16", "float32"}:
        raise ConfigError("cache.storage_dtype must be float16, bfloat16, or float32")

    return CacheConfig(
        output_dir=Path(str(require_resolved(raw.get("output_dir"), "cache.output_dir"))),
        max_records_per_shard=max_records,
        max_bytes_per_shard=max_bytes,
        storage_dtype=storage_dtype,
        zero_norm_epsilon=float(raw.get("zero_norm_epsilon", 1e-12)),
        delta_atol=float(raw.get("delta_atol", 1e-5)),
        delta_rtol=float(raw.get("delta_rtol", 1e-4)),
    )


def resolved_scientific_metadata(config: dict[str, Any]) -> dict[str, Any]:
    """Return model/margin metadata, failing before any real cache write."""
    output: dict[str, Any] = {}
    required = {
        "model": (
            "checkpoint",
            "revision",
            "layer",
            "hook",
            "dtype",
            "attention_implementation",
        ),
        "margin": (
            "definition",
            "candidate_answers",
            "multi_token_aggregation",
            "teacher_forcing",
            "gradient_location",
            "evaluation_position",
        ),
    }
    for section, names in required.items():
        raw_section = config.get(section)
        if not isinstance(raw_section, dict):
            raise ConfigError(f"{section} must be an object")
        output[section] = {}
        for name in names:
            value = require_resolved(raw_section.get(name), f"{section}.{name}")
            output[section][name] = value

    if output["margin"]["gradient_location"] != "arrival":
        raise ConfigError("margin.gradient_location must be arrival")
    if isinstance(output["model"]["layer"], bool) or not isinstance(
        output["model"]["layer"], int
    ):
        raise ConfigError("model.layer must be an integer")
    return output
