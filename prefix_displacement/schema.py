from __future__ import annotations

import math
import re
import string
from typing import Any, Mapping


BOUNDARY_CLASSES = {
    "newline",
    "sentence_punctuation",
    "other_punctuation",
    "whitespace_prefixed_token",
    "number_token",
    "operator_token",
    "ordinary",
}

METADATA_FIELDS = (
    "problem_id",
    "trajectory_id",
    "transition_id",
    "token_index",
    "current_token_id",
    "current_token_text",
    "next_token_id",
    "next_token_text",
    "absolute_position",
    "relative_generated_position",
    "boundary_class",
    "surprisal",
    "correctness",
    "trace_correct",
    "split",
)

TENSOR_FIELDS = (
    "h_departure",
    "h_arrival",
    "delta",
    "g_arrival",
    "baseline_margin",
)


class TransitionSchemaError(ValueError):
    """Raised when a transition violates the cache contract."""


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for tensor cache operations. No installation was "
            "attempted; run this in the existing GPU-server environment."
        ) from exc
    return torch


def classify_boundary(token_text: str) -> str:
    if "\n" in token_text:
        return "newline"
    stripped = token_text.strip()
    if stripped and re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", stripped):
        return "number_token"
    if stripped and all(character in "+-*/=<>^%" for character in stripped):
        return "operator_token"
    if stripped and all(character in ".!?" for character in stripped):
        return "sentence_punctuation"
    if stripped and all(character in string.punctuation for character in stripped):
        return "other_punctuation"
    if token_text[:1].isspace():
        return "whitespace_prefixed_token"
    return "ordinary"


def _required_text(record: Mapping[str, Any], field: str) -> str:
    value = record.get(field)
    if value is None:
        raise TransitionSchemaError(f"Missing required field: {field}")
    text = str(value)
    if not text:
        raise TransitionSchemaError(f"{field} must be non-empty")
    return text


def _required_int(record: Mapping[str, Any], field: str, *, minimum: int = 0) -> int:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TransitionSchemaError(f"{field} must be an integer")
    if value < minimum:
        raise TransitionSchemaError(f"{field} must be >= {minimum}")
    return value


def _storage_dtype(torch: Any, name: str):
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise TransitionSchemaError(f"Unsupported storage dtype: {name}") from exc


def normalize_transition_record(
    record: Mapping[str, Any],
    *,
    storage_dtype: str,
    delta_atol: float,
    delta_rtol: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Detach tensors to CPU, validate them, and return metadata/tensor dicts."""
    torch = require_torch()

    correctness_value = record.get("correctness", record.get("trace_correct"))
    if not isinstance(correctness_value, bool):
        raise TransitionSchemaError("correctness/trace_correct must be boolean")
    if "correctness" in record and "trace_correct" in record:
        if record["correctness"] != record["trace_correct"]:
            raise TransitionSchemaError("correctness and trace_correct disagree")

    current_text = str(record.get("current_token_text", record.get("current_token", "")))
    next_text = str(record.get("next_token_text", record.get("next_token", "")))
    if not current_text:
        raise TransitionSchemaError("current_token_text/current_token must be non-empty")
    if not next_text:
        raise TransitionSchemaError("next_token_text/next_token must be non-empty")

    boundary_class = str(record.get("boundary_class") or classify_boundary(next_text))
    if boundary_class not in BOUNDARY_CLASSES:
        raise TransitionSchemaError(
            f"boundary_class must be one of {sorted(BOUNDARY_CLASSES)}, got {boundary_class!r}"
        )

    surprisal = record.get("surprisal")
    if isinstance(surprisal, bool) or not isinstance(surprisal, (int, float)):
        raise TransitionSchemaError("surprisal must be numeric")
    surprisal = float(surprisal)
    if not math.isfinite(surprisal) or surprisal < 0.0:
        raise TransitionSchemaError("surprisal must be finite and non-negative")

    metadata = {
        "problem_id": _required_text(record, "problem_id"),
        "trajectory_id": _required_text(record, "trajectory_id"),
        "transition_id": _required_text(record, "transition_id"),
        "token_index": _required_int(record, "token_index"),
        "current_token_id": _required_int(record, "current_token_id"),
        "current_token_text": current_text,
        "next_token_id": _required_int(record, "next_token_id"),
        "next_token_text": next_text,
        "absolute_position": _required_int(record, "absolute_position"),
        "relative_generated_position": _required_int(
            record, "relative_generated_position"
        ),
        "boundary_class": boundary_class,
        "surprisal": surprisal,
        "correctness": correctness_value,
        "trace_correct": correctness_value,
        "split": _required_text(record, "split"),
    }

    dtype = _storage_dtype(torch, storage_dtype)
    tensors: dict[str, Any] = {}
    for field in ("h_departure", "h_arrival", "delta", "g_arrival"):
        value = record.get(field)
        if value is None:
            raise TransitionSchemaError(f"Missing required tensor: {field}")
        tensor = torch.as_tensor(value).detach().to(device="cpu", dtype=dtype).contiguous()
        if tensor.ndim != 1:
            raise TransitionSchemaError(f"{field} must have shape [d], got {tuple(tensor.shape)}")
        if not bool(torch.isfinite(tensor).all()):
            raise TransitionSchemaError(f"{field} contains NaN or Inf")
        tensors[field] = tensor

    dimensions = {int(tensors[field].shape[0]) for field in tensors}
    if len(dimensions) != 1:
        raise TransitionSchemaError(f"Hidden/gradient dimensions disagree: {dimensions}")

    margin_value = record.get("baseline_margin", record.get("answer_margin"))
    if margin_value is None:
        raise TransitionSchemaError("baseline_margin/answer_margin is required")
    margin = torch.as_tensor(margin_value).detach().to(device="cpu", dtype=torch.float32)
    if margin.numel() != 1 or not bool(torch.isfinite(margin).all()):
        raise TransitionSchemaError("baseline_margin must be one finite scalar")
    tensors["baseline_margin"] = margin.reshape(())

    try:
        torch.testing.assert_close(
            tensors["delta"].float(),
            (tensors["h_arrival"] - tensors["h_departure"]).float(),
            atol=delta_atol,
            rtol=delta_rtol,
        )
    except AssertionError as exc:
        raise TransitionSchemaError(
            "delta must equal h_arrival - h_departure within configured tolerance"
        ) from exc

    return metadata, tensors


def record_tensor_nbytes(tensors: Mapping[str, Any]) -> int:
    return sum(int(tensor.numel() * tensor.element_size()) for tensor in tensors.values())
