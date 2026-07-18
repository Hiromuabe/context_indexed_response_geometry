from __future__ import annotations

import os
import random
import re
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from prefix_displacement.model_loading import MODEL_PATH_ENV, resolve_model_source

from .subspaces import explained_variance, top_svd
from .utils import ensure_layout, read_json


OPERATOR_PATTERN = re.compile(r"[+\-*/=<>%^×÷]")


def review_roots(config: dict[str, Any]) -> tuple[Path, Path]:
    """Return immutable main-run inputs and isolated review outputs."""
    output_root = ensure_layout(config)
    source_root = Path(str(config.get("source_results_root", output_root)))
    if not source_root.is_dir():
        raise FileNotFoundError(f"source_results_root does not exist: {source_root}")
    return source_root, output_root


def _manifest_model_candidates(source_root: Path | None, model_config: dict[str, Any]) -> list[dict[str, Any]]:
    """Recover the exact model source used by a completed parent experiment."""
    if source_root is None:
        return []
    candidates: list[dict[str, Any]] = []
    for relative in ("manifests/candidate_tokens.json", "manifests/hidden_states.json"):
        path = source_root / relative
        if not path.is_file():
            continue
        payload = read_json(path)
        metadata = payload.get("model", {})
        source = metadata.get("model_source")
        if not source:
            continue
        source = str(Path(str(source)).expanduser()) if str(source).startswith(("/", ".", "~")) else str(source)
        kwargs: dict[str, Any] = {
            "local_files_only": True,
            "trust_remote_code": bool(model_config.get("trust_remote_code", False)),
        }
        if not Path(source).is_dir():
            revision = metadata.get("resolved_revision") or model_config.get("revision", "main")
            if revision:
                kwargs["revision"] = revision
        candidates.append({"source": source, "kwargs": kwargs, "mode": f"parent_manifest:{relative}"})
    return candidates


def load_review_tokenizer(
    config: dict[str, Any],
    model_path: str | None = None,
    *,
    source_root: Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load a matching tokenizer without silently waiting on the HF Hub.

    Resolution is explicit path/environment override, parent-run manifest, then
    the configured Hugging Face cache. Network access is opt-in because these
    review stages only need tokenizer files and normally run after extraction.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("transformers is required to decode review experiment inputs") from exc

    model_config = config["model"]
    explicit_override = model_path or os.environ.get(MODEL_PATH_ENV)
    candidates: list[dict[str, Any]] = []
    if explicit_override:
        expanded_override = Path(str(explicit_override)).expanduser()
        if str(explicit_override).startswith(("/", ".", "~")) and not expanded_override.is_dir():
            raise FileNotFoundError(
                f"model path does not exist: {expanded_override}. Replace the example path with a real "
                "checkpoint directory, or use `run_review_experiments --temporary-model-download`."
            )
        source, kwargs = resolve_model_source(model_config, model_path)
        kwargs = {**kwargs, "local_files_only": True}
        candidates.append({"source": source, "kwargs": kwargs, "mode": "explicit_model_path"})
    else:
        candidates.extend(_manifest_model_candidates(source_root, model_config))
        source, kwargs = resolve_model_source(model_config)
        candidates.append({
            "source": source,
            "kwargs": {**kwargs, "local_files_only": True},
            "mode": "local_hf_cache" if not Path(source).is_dir() else "configured_local_path",
        })

    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, str | None]] = set()
    for candidate in candidates:
        key = (str(candidate["source"]), candidate["kwargs"].get("revision"))
        if key not in seen:
            seen.add(key)
            unique.append(candidate)

    errors: list[str] = []
    for candidate in unique:
        source = str(candidate["source"])
        print(
            f"[review_tokenizer] source={source!r} mode={candidate['mode']} local_files_only=True",
            flush=True,
        )
        started = time.monotonic()
        try:
            tokenizer = AutoTokenizer.from_pretrained(source, **candidate["kwargs"])
        except Exception as exc:
            errors.append(f"{candidate['mode']} {source!r}: {type(exc).__name__}: {exc}")
            continue
        elapsed = time.monotonic() - started
        is_fast = bool(getattr(tokenizer, "is_fast", False))
        print(
            f"[review_tokenizer] LOADED source={source!r} fast={is_fast} elapsed={elapsed:.2f}s",
            flush=True,
        )
        return tokenizer, {
            "source": source,
            "mode": str(candidate["mode"]),
            "local_files_only": True,
            "load_elapsed_seconds": elapsed,
            "is_fast": is_fast,
        }

    allow_download = bool(config.get("review_experiments", {}).get("allow_hub_tokenizer_download", False))
    if allow_download and not explicit_override:
        source, kwargs = resolve_model_source(model_config)
        kwargs = {**kwargs, "local_files_only": False}
        print(f"[review_tokenizer] source={source!r} mode=hub_download local_files_only=False", flush=True)
        try:
            tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
        except Exception as exc:
            errors.append(f"hub_download {source!r}: {type(exc).__name__}: {exc}")
        else:
            return tokenizer, {"source": source, "mode": "hub_download", "local_files_only": False}

    attempts = "\n  - ".join(errors) if errors else "no tokenizer sources were available"
    raise RuntimeError(
        "Tokenizer files were not found locally; the review pipeline will not contact the "
        "Hugging Face Hub unless explicitly enabled. Re-run with "
        "`--model-path /absolute/path/to/Qwen2.5-Math-1.5B` or set "
        f"`{MODEL_PATH_ENV}=/absolute/path/to/Qwen2.5-Math-1.5B`. To permit a "
        "tokenizer-only Hub download instead, set "
        "`review_experiments.allow_hub_tokenizer_download=true`. "
        "For a full non-persistent pipeline download, run "
        "`run_review_experiments --temporary-model-download`. "
        f"Attempts:\n  - {attempts}"
    )


def review_token_category(text: str) -> str:
    """Coarse distribution label used by the candidate-transfer controls."""
    stripped = text.strip()
    if not stripped or text[:1].isspace():
        return "whitespace"
    if any(character.isdigit() for character in stripped):
        return "number"
    if stripped and all(character in "+-*/=<>%^×÷" for character in stripped):
        return "operator"
    if stripped and all(character.isalpha() for character in stripped):
        return "word"
    return "other"


def operation_signature(text: str) -> str:
    operators = sorted(set(OPERATOR_PATTERN.findall(text)))
    return "".join(operators) if operators else "none"


def deterministic_context_split(prefix_ids: Iterable[str], seed: int) -> tuple[list[str], list[str]]:
    values = sorted(map(str, prefix_ids))
    random.Random(int(seed)).shuffle(values)
    midpoint = len(values) // 2
    if midpoint == 0 or midpoint == len(values):
        raise ValueError("independent candidate selection needs at least two contexts")
    return sorted(values[:midpoint]), sorted(values[midpoint:])


def select_candidates_from_logits(
    logits: np.ndarray,
    *,
    tokenizer: Any,
    total: int,
    proposal_top_k: int,
    decode_batch_size: int = 512,
    progress_label: str | None = None,
) -> list[dict[str, Any]]:
    """Apply the paper's proposal/coverage selection to one context sample."""
    values = np.asarray(logits, dtype=np.float32)
    if values.ndim != 2 or not len(values):
        raise ValueError("logits must have shape [selection_context, vocabulary]")
    started = time.monotonic()
    values = values - values.max(axis=1, keepdims=True)
    probabilities = np.exp(values)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    top_k = min(int(proposal_top_k), values.shape[1])
    top_indices = np.argpartition(probabilities, -top_k, axis=1)[:, -top_k:]
    proposed = np.unique(top_indices)
    if progress_label:
        print(
            f"[review_candidates] PHASE set={progress_label} phase=softmax_topk "
            f"proposed={len(proposed)} elapsed={time.monotonic()-started:.2f}s",
            flush=True,
        )
    special = set(map(int, getattr(tokenizer, "all_special_ids", [])))

    # Every token can occur at most once in a context's top-k row, so a single
    # bincount is exactly the old per-token `mean(any(top_indices == id))`.
    # Mean probabilities are gathered for all proposals in one vectorized pass.
    coverage = np.bincount(top_indices.reshape(-1), minlength=values.shape[1])[proposed]
    coverage = coverage.astype(np.float64) / float(values.shape[0])
    mean_probability = probabilities[:, proposed].mean(axis=0)
    order = np.lexsort((proposed, -mean_probability, -coverage))
    proposed = proposed[order]
    coverage = coverage[order]
    mean_probability = mean_probability[order]
    valid = np.asarray([
        int(token_id) not in special and int(token_id) < len(tokenizer)
        for token_id in proposed
    ], dtype=bool)
    proposed = proposed[valid]
    coverage = coverage[valid]
    mean_probability = mean_probability[valid]
    if progress_label:
        print(
            f"[review_candidates] PHASE set={progress_label} phase=vectorized_statistics "
            f"valid_proposed={len(proposed)} elapsed={time.monotonic()-started:.2f}s",
            flush=True,
        )

    rows: list[dict[str, Any]] = []
    categories: set[str] = set()
    required_categories = {"number", "operator", "word", "whitespace", "other"}
    backend = getattr(tokenizer, "backend_tokenizer", None) or getattr(tokenizer, "_tokenizer", None)
    use_backend_batch = backend is not None and hasattr(backend, "decode_batch")
    batch_size = max(1, 4096 if use_backend_batch else int(decode_batch_size))
    last_report = time.monotonic()
    scanned = 0
    for start in range(0, len(proposed), batch_size):
        batch_ids = list(map(int, proposed[start : start + batch_size]))
        decode_kwargs = {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
        sequences = [[token_id] for token_id in batch_ids]
        if use_backend_batch:
            try:
                texts = backend.decode_batch(sequences, skip_special_tokens=False)
            except TypeError:
                texts = backend.decode_batch(sequences, False)
        elif hasattr(tokenizer, "batch_decode"):
            texts = tokenizer.batch_decode(sequences, **decode_kwargs)
        else:
            texts = [tokenizer.decode([token_id], **decode_kwargs) for token_id in batch_ids]
        scanned += len(batch_ids)
        for offset, (token_id, text) in enumerate(zip(batch_ids, texts)):
            if not text or "\ufffd" in text:
                continue
            if any(unicodedata.category(character) in {"Cc", "Cs"} and character not in "\n\t" for character in text):
                continue
            category = review_token_category(text)
            index = start + offset
            rows.append({
                "token_id": token_id,
                "text": text,
                "mean_probability": float(mean_probability[index]),
                "coverage": float(coverage[index]),
                "review_category": category,
            })
            categories.add(category)
        # Once every seeded category and enough score-ordered rows are known,
        # lower-ranked proposals cannot affect the selected set.
        if len(rows) >= int(total) and required_categories.issubset(categories):
            break
        if progress_label and time.monotonic() - last_report >= 2.0:
            print(
                f"[review_candidates] PHASE set={progress_label} phase=decode "
                f"scanned={scanned}/{len(proposed)} valid={len(rows)} "
                f"categories={','.join(sorted(categories))} elapsed={time.monotonic()-started:.2f}s",
                flush=True,
            )
            last_report = time.monotonic()

    if progress_label:
        decoder = "rust_backend" if use_backend_batch else "python_tokenizer"
        print(
            f"[review_candidates] PHASE set={progress_label} phase=decode_done decoder={decoder} "
            f"scanned={scanned}/{len(proposed)} valid={len(rows)} elapsed={time.monotonic()-started:.2f}s",
            flush=True,
        )

    selected: list[dict[str, Any]] = []
    for category in ("number", "operator", "word", "whitespace", "other"):
        match = next((row for row in rows if row["review_category"] == category), None)
        if match is not None:
            selected.append(match)
    selected.extend(row for row in rows if row not in selected)
    if len(selected) < int(total):
        raise RuntimeError(f"only {len(selected)} valid tokens for requested independent set of {total}")
    return selected[: int(total)]


def distribution_groups(candidate_rows: list[dict[str, Any]]) -> dict[str, list[int]]:
    """Return high/low probability and lexical-distribution token IDs."""
    if not candidate_rows:
        return {}
    ordered = sorted(candidate_rows, key=lambda row: (float(row["mean_probability"]), int(row["token_id"])))
    midpoint = len(ordered) // 2
    groups: dict[str, list[int]] = {
        "low_probability": [int(row["token_id"]) for row in ordered[:midpoint]],
        "high_probability": [int(row["token_id"]) for row in ordered[midpoint:]],
    }
    for row in candidate_rows:
        category = str(row.get("review_category") or review_token_category(str(row.get("text", ""))))
        groups.setdefault(category, []).append(int(row["token_id"]))
    return {key: sorted(set(values)) for key, values in groups.items() if values}


def shuffled_context(tokens: list[int], seed: int) -> list[int]:
    """Destroy token order while preserving length, endpoints, and token multiset."""
    values = list(map(int, tokens))
    if len(values) < 4:
        return values
    middle = values[1:-1]
    original = list(middle)
    random.Random(int(seed)).shuffle(middle)
    if middle == original and len(middle) > 1:
        middle = middle[1:] + middle[:1]
    return [values[0], *middle, values[-1]]


def build_context_control_records(
    prefixes: list[dict[str, Any]],
    decode: Callable[[list[int]], str],
    *,
    seed: int,
    target_groups: tuple[str, ...] = ("analysis_dev", "analysis_test"),
    minimum_timepoint_gap: int = 2,
    max_targets: int | None = None,
    max_auxiliary: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Construct exact-length, shuffled, longitudinal, and operation controls."""
    targets = [row for row in prefixes if row.get("problem_group") in set(target_groups)]
    targets.sort(key=lambda row: str(row["prefix_id"]))
    if max_targets is not None:
        targets = targets[: int(max_targets)]
    decoded = {str(row["prefix_id"]): decode(list(map(int, row["prefix_token_ids"]))) for row in prefixes}
    signatures = {key: operation_signature(value) for key, value in decoded.items()}
    by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_operation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in prefixes:
        by_length[int(row["prefix_length"])].append(row)
        by_operation[signatures[str(row["prefix_id"])]].append(row)
    records: list[dict[str, Any]] = []
    counts: dict[str, int] = defaultdict(int)

    def append_control(target: dict[str, Any], kind: str, tokens: list[int], source: dict[str, Any] | None, **metadata: Any) -> None:
        ordinal = counts[kind]
        counts[kind] += 1
        records.append({
            "context_id": f"{target['prefix_id']}::{kind}::{ordinal}",
            "role": "control", "control_type": kind,
            "target_prefix_id": str(target["prefix_id"]),
            "target_problem_id": str(target["problem_id"]),
            "source_prefix_id": str(source["prefix_id"]) if source is not None else str(target["prefix_id"]),
            "prefix_token_ids": list(map(int, tokens)), "prefix_length": len(tokens),
            "exact_target_length": len(tokens) == int(target["prefix_length"]),
            "same_last_token": bool(tokens and int(tokens[-1]) == int(target["prefix_token_ids"][-1])),
            **metadata,
        })

    auxiliary_rows = sorted(
        (row for row in prefixes if row.get("problem_group") == "auxiliary"),
        key=lambda row: str(row["prefix_id"]),
    )
    if max_auxiliary is not None:
        auxiliary_rows = auxiliary_rows[: int(max_auxiliary)]
    for row in auxiliary_rows:
        records.append({
            "context_id": str(row["prefix_id"]), "role": "auxiliary", "control_type": "none",
            "target_prefix_id": None, "target_problem_id": str(row["problem_id"]),
            "source_prefix_id": str(row["prefix_id"]), "prefix_token_ids": list(map(int, row["prefix_token_ids"])),
            "prefix_length": int(row["prefix_length"]), "exact_target_length": True, "same_last_token": True,
        })
    for target_index, target in enumerate(targets):
        target_id = str(target["prefix_id"])
        records.append({
            "context_id": target_id, "role": "target", "control_type": "self",
            "target_prefix_id": target_id, "target_problem_id": str(target["problem_id"]),
            "source_prefix_id": target_id, "prefix_token_ids": list(map(int, target["prefix_token_ids"])),
            "prefix_length": int(target["prefix_length"]), "exact_target_length": True, "same_last_token": True,
        })
        exact_pool = [row for row in by_length[int(target["prefix_length"])] if row["problem_id"] != target["problem_id"]]
        if exact_pool:
            source = sorted(exact_pool, key=lambda row: str(row["prefix_id"]))[(int(seed) + target_index) % len(exact_pool)]
            append_control(target, "exact_length_random", source["prefix_token_ids"], source)
        shuffled = shuffled_context(target["prefix_token_ids"], int(seed) + 1009 * (target_index + 1))
        if shuffled != list(map(int, target["prefix_token_ids"])):
            append_control(target, "token_order_shuffled", shuffled, target)
        full = list(map(int, target["prefix_token_ids"] + target.get("evaluation_suffix_token_ids", [])))
        position = len(target["prefix_token_ids"]) - 1
        alternatives = [candidate for candidate in (position - int(minimum_timepoint_gap), position + int(minimum_timepoint_gap)) if 1 <= candidate < len(full) - 1]
        for candidate in alternatives:
            append_control(target, "same_problem_timepoint", full[: candidate + 1], target, timepoint_offset=int(candidate - position))
        signature = signatures[target_id]
        operation_pool = [row for row in by_operation[signature] if row["problem_id"] != target["problem_id"]] if signature != "none" else []
        if operation_pool:
            source = min(operation_pool, key=lambda row: (abs(int(row["prefix_length"]) - int(target["prefix_length"])), str(row["prefix_id"])))
            append_control(target, "operation_matched", source["prefix_token_ids"], source, operation_signature=signature)
    diagnostics = {
        "target_count": len(targets), "auxiliary_count": len(auxiliary_rows),
        "record_count": len(records), "control_counts": dict(counts),
        "definitions": {
            "exact_length_random": "different problem, exact token length",
            "token_order_shuffled": "same token multiset and first/last token, shuffled interior",
            "same_problem_timepoint": "same trajectory at +/- minimum_timepoint_gap tokens when available",
            "operation_matched": "different problem with identical decoded operator signature; nearest length",
        },
    }
    return records, diagnostics


def projection_cell_energies(samples: np.ndarray, basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(samples, dtype=np.float64)
    projected = values @ np.asarray(basis, dtype=np.float64)
    return np.square(projected).sum(axis=-1), np.square(values).sum(axis=-1)


def auxiliary_token_statistics(
    states: np.ndarray, auxiliary_indices: np.ndarray, token_indices: Iterable[int], *, chunk_size: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """Compute auxiliary token/grand means without a large prefix tensor."""
    auxiliary = np.asarray(auxiliary_indices, dtype=np.int64)
    tokens = np.asarray(list(token_indices), dtype=np.int64)
    if not len(auxiliary) or not len(tokens):
        raise ValueError("auxiliary and token index sets must be non-empty")
    total = np.zeros((len(tokens), states.shape[2]), dtype=np.float64)
    for start in range(0, len(auxiliary), max(1, int(chunk_size))):
        selected = auxiliary[start : start + max(1, int(chunk_size))]
        block = np.asarray(states[selected[:, None], tokens[None, :], :], dtype=np.float32)
        total += block.sum(axis=0, dtype=np.float64)
    token_mean = (total / float(len(auxiliary))).astype(np.float32)
    grand_mean = token_mean.mean(axis=0, dtype=np.float64).astype(np.float32)
    return token_mean, grand_mean


def center_context_block(
    states: np.ndarray, context_index: int, token_indices: Iterable[int], token_mean: np.ndarray, grand_mean: np.ndarray
) -> np.ndarray:
    """Apply the exact paper double-centering formula to one context row."""
    tokens = np.asarray(list(token_indices), dtype=np.int64)
    block = np.asarray(states[int(context_index), tokens, :], dtype=np.float32)
    prefix_mean = block.mean(axis=0, dtype=np.float64).astype(np.float32)
    residual = block - prefix_mean[None] - np.asarray(token_mean, dtype=np.float32) + np.asarray(grand_mean, dtype=np.float32)[None]
    residual -= residual.mean(axis=0, dtype=np.float64).astype(np.float32)[None]
    return residual


def evaluate_transfer(source: np.ndarray, target: np.ndarray, rank: int) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    basis = top_svd(source, int(rank), allow_rank_reduction=True)
    numerator, denominator = projection_cell_energies(target, basis)
    return explained_variance(target, basis), numerator, denominator, basis
