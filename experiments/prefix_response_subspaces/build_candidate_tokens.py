from __future__ import annotations

import argparse
import random
import time
import unicodedata

import numpy as np

from .src.data import pad_token_rows
from .src.model import assert_output_order, load_next_token_model
from .src.utils import atomic_json, ensure_layout, file_sha256, load_config, read_jsonl, stable_hash, stage_is_complete


def token_category(text: str) -> str:
    stripped = text.strip()
    if "\n" in text: return "newline"
    if any(ch.isdigit() for ch in stripped): return "number"
    if stripped and all(ch in "+-*/=<>%^×÷" for ch in stripped): return "operator"
    if stripped and all(unicodedata.category(ch).startswith("P") for ch in stripped): return "punctuation"
    if stripped and all(ch.isalpha() for ch in stripped): return "word"
    return "subword"


def stratified_partition(rows: list[dict], calibration: int, folds: int, seed: int) -> tuple[list[int], list[list[int]]]:
    group_count = folds + 1
    target = calibration
    if len(rows) != target * group_count:
        raise ValueError("candidate total must equal calibration * (folds + 1)")
    buckets = [[] for _ in range(group_count)]
    strata = {}
    for index, row in enumerate(rows):
        key = (row["probability_band"], row["coverage_band"], row["category"])
        strata.setdefault(key, []).append(index)
    for ordinal, key in enumerate(sorted(strata, key=str)):
        indices = strata[key]
        random.Random(seed + ordinal * 7919).shuffle(indices)
        for index in indices:
            minimum = min(map(len, buckets))
            choices = [b for b in range(group_count) if len(buckets[b]) == minimum]
            bucket = choices[(ordinal + len(indices)) % len(choices)]
            buckets[bucket].append(index)
    # Deterministically rebalance sparse strata rounding without using results.
    while any(len(bucket) != target for bucket in buckets):
        donor = max(range(group_count), key=lambda i: len(buckets[i]))
        receiver = min(range(group_count), key=lambda i: len(buckets[i]))
        buckets[receiver].append(buckets[donor].pop())
    return sorted(buckets[0]), [sorted(bucket) for bucket in buckets[1:]]


def stratified_nested_order(rows: list[dict], indices: list[int], seed: int) -> list[int]:
    """Return a deterministic nested order interleaving all calibration strata."""
    grouped = {}
    for index in indices:
        row = rows[int(index)]
        key = (row["probability_band"], row["coverage_band"], row["category"])
        grouped.setdefault(key, []).append(int(index))
    keys = sorted(grouped, key=str)
    for ordinal, key in enumerate(keys):
        random.Random(seed + 1543 * (ordinal + 1)).shuffle(grouped[key])
    ordered = []
    while len(ordered) < len(indices):
        for key in keys:
            if grouped[key]:
                ordered.append(grouped[key].pop())
    if sorted(ordered) != sorted(map(int, indices)):
        raise AssertionError("stratified calibration order changed membership")
    return ordered


def logit_scored_indices(prefixes: list[dict], sparse_paper_logits: bool) -> list[int]:
    required = {"candidate_selection", "analysis_dev", "analysis_test"}
    return [i for i,row in enumerate(prefixes) if not sparse_paper_logits or row["problem_group"] in required]


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--model-path")
    args = parser.parse_args(); config = load_config(args.config); root = ensure_layout(config)
    prefix_path = root / "prefix_pool/prefixes.jsonl"; prefix_hash = file_sha256(prefix_path)
    manifest_path = root / "manifests/candidate_tokens.json"
    if stage_is_complete(manifest_path, config, {"prefixes_sha256": prefix_hash}): print(manifest_path); return
    import torch
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required for model-conditioned candidate selection")
    prefixes = read_jsonl(prefix_path); loaded = load_next_token_model(config, args.model_path)
    # Paper controls never need full-vocabulary logits for auxiliary, Global
    # training, or Wrong-donor prefixes.  Score only rows used to propose
    # candidates or evaluate full-vocabulary ranks.  Legacy prediction matching
    # keeps the original full-prefix axis.
    sparse_paper_logits = str(config.get("controls", {}).get("kind", "prediction_matched")) == "wrong_prefix"
    scored_indices = logit_scored_indices(prefixes, sparse_paper_logits)
    scored_prefixes = [prefixes[i] for i in scored_indices]
    batch_size = max(1, int(config["extraction"]["per_device_batch_size"]) * max(1, len(loaded.device_ids)))
    logits_path = root / "prefix_pool/next_token_logits.npy"; probability_path = root / "prefix_pool/next_token_probabilities.npy"
    # Qwen checkpoints can expose more LM-head rows than tokenizer.vocab_size
    # (reserved/unused output IDs). Allocate from the actual forward output,
    # not from tokenizer metadata.
    storage_name = str(config["candidates"].get("next_token_logits_storage_dtype", "float32"))
    if storage_name not in {"float16", "float32"}:
        raise ValueError("candidates.next_token_logits_storage_dtype must be float16 or float32")
    logits_storage_dtype = np.float32 if storage_name == "float32" else np.float16
    all_logits = None
    started=time.monotonic(); print(f"[candidate_logits] scored_prefixes={len(scored_prefixes)} full_prefixes={len(prefixes)} sparse={sparse_paper_logits}",flush=True)
    with torch.no_grad():
        for start in range(0, len(scored_prefixes), batch_size):
            batch = scored_prefixes[start:start+batch_size]
            ids, mask, positions = pad_token_rows([row["prefix_token_ids"] for row in batch], loaded.tokenizer.pad_token_id)
            sample = torch.arange(start, start+len(batch), dtype=torch.long)
            logits, observed = loaded.model(ids.to(loaded.device), mask.to(loaded.device), positions.to(loaded.device), sample.to(loaded.device))
            assert_output_order(sample, observed)
            logits_numpy = logits.float().cpu().numpy()
            if all_logits is None:
                all_logits = np.lib.format.open_memmap(
                    logits_path,
                    mode="w+",
                    dtype=logits_storage_dtype,
                    shape=(len(scored_prefixes), int(logits_numpy.shape[-1])),
                )
            elif int(logits_numpy.shape[-1]) != int(all_logits.shape[-1]):
                raise RuntimeError(
                    "model output vocabulary width changed between batches: "
                    f"{logits_numpy.shape[-1]} != {all_logits.shape[-1]}"
                )
            all_logits[start:start+len(batch)] = logits_numpy.astype(logits_storage_dtype)
            completed=start+len(batch); rate=completed/max(time.monotonic()-started,1e-9); print(f"[candidate_logits] {completed}/{len(scored_prefixes)} prefixes rate={rate:.2f}/s eta={(len(scored_prefixes)-completed)/max(rate,1e-9)/60:.1f}m",flush=True)
    if all_logits is None:
        raise RuntimeError("prefix pool is empty; no next-token logits were produced")
    all_logits.flush()
    all_probabilities = np.lib.format.open_memmap(probability_path, mode="w+", dtype=np.float16, shape=all_logits.shape)
    entropy = np.empty(len(scored_prefixes), dtype=np.float32)
    for start in range(0, len(scored_prefixes), 32):
        chunk = np.asarray(all_logits[start:start+32], dtype=np.float32); chunk -= chunk.max(axis=1, keepdims=True); probability = np.exp(chunk); probability /= probability.sum(axis=1, keepdims=True)
        all_probabilities[start:start+len(probability)] = probability.astype(np.float16); entropy[start:start+len(probability)] = -np.sum(probability*np.log(np.clip(probability, 1e-12, None)), axis=1)
    all_probabilities.flush(); np.save(root / "prefix_pool/next_token_entropy.npy", entropy)
    selector = np.asarray([row["problem_group"] == "candidate_selection" for row in scored_prefixes])
    selector_indices = np.flatnonzero(selector); selected_logits = np.asarray(all_logits[selector_indices], dtype=np.float32); selected_logits -= selected_logits.max(axis=1, keepdims=True); probabilities = np.exp(selected_logits); probabilities /= probabilities.sum(axis=1, keepdims=True)
    top_k = min(int(config["candidates"]["proposal_top_k"]), probabilities.shape[1])
    top_indices = np.argpartition(probabilities, -top_k, axis=1)[:, -top_k:]
    proposed = np.unique(top_indices)
    special = set(map(int, loaded.tokenizer.all_special_ids)); candidates = []
    for token_id in proposed:
        # LM heads may contain reserved rows beyond the tokenizer vocabulary.
        # They have no valid text representation and cannot be forced tokens.
        if int(token_id) in special or int(token_id) >= len(loaded.tokenizer): continue
        text = loaded.tokenizer.decode([int(token_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if not text or "\ufffd" in text or any(unicodedata.category(ch) in {"Cc", "Cs"} and ch not in "\n\t" for ch in text): continue
        p = probabilities[:, int(token_id)]
        candidates.append({"token_id": int(token_id), "text": text, "mean_probability": float(p.mean()), "coverage": float(np.mean(np.any(top_indices == int(token_id), axis=1))), "category": token_category(text)})
    total = int(config["candidates"]["total"])
    candidates.sort(key=lambda row: (-row["coverage"], -row["mean_probability"], row["token_id"]))
    # Seed every observed lexical category, then fill by model-conditioned score.
    selected = []
    for category in ("number", "operator", "punctuation", "newline", "word", "subword"):
        match = next((row for row in candidates if row["category"] == category), None)
        if match is not None and match not in selected: selected.append(match)
    selected.extend(row for row in candidates if row not in selected)
    selected = selected[:total]
    if len(selected) != total: raise RuntimeError(f"only {len(selected)} valid candidate tokens for requested {total}")
    for key, bands in (("mean_probability", int(config["candidates"]["probability_bands"])), ("coverage", int(config["candidates"]["coverage_bands"]))):
        order = sorted(range(total), key=lambda i: (selected[i][key], selected[i]["token_id"]))
        for rank, index in enumerate(order): selected[index]["probability_band" if key == "mean_probability" else "coverage_band"] = min(bands-1, rank * bands // total)
    calibration, fold_eval = stratified_partition(selected, int(config["candidates"]["calibration"]), int(config["candidates"]["folds"]), int(config["seed"]))
    calibration_stability_order = stratified_nested_order(selected, calibration, int(config["seed"]) + 4049)
    analysis_indices = sorted(index for fold in fold_eval for index in fold)
    folds = []
    for fold_id, evaluation in enumerate(fold_eval):
        train = sorted(set(analysis_indices) - set(evaluation))
        folds.append({"fold_id": fold_id, "train_indices": train, "evaluation_indices": evaluation, "train_token_ids": [selected[i]["token_id"] for i in train], "evaluation_token_ids": [selected[i]["token_id"] for i in evaluation]})
    payload = {"candidate_tokens": selected, "candidate_token_ids": [row["token_id"] for row in selected], "calibration_indices": calibration, "calibration_token_ids": [selected[i]["token_id"] for i in calibration], "calibration_stability_order": calibration_stability_order, "calibration_stability_token_ids": [selected[i]["token_id"] for i in calibration_stability_order], "analysis_indices": analysis_indices, "folds": folds}
    output = root / "candidate_tokens/candidate_tokens.json"; atomic_json(output, payload)
    entropy_path = root / "prefix_pool/next_token_entropy.npy"
    atomic_json(manifest_path, {"complete": True, "config_hash": stable_hash(config), "prefixes_sha256": prefix_hash, "prefix_axis_ids": [row["prefix_id"] for row in scored_prefixes], "full_prefix_count":len(prefixes), "logit_prefix_count":len(scored_prefixes), "sparse_paper_logits":sparse_paper_logits, "model_output_vocabulary_size": int(all_logits.shape[-1]), "tokenizer_vocabulary_size": int(len(loaded.tokenizer)), "next_token_logits_storage_dtype": storage_name, "candidate_tokens": str(output), "candidate_tokens_sha256": file_sha256(output), "candidate_set_hash": stable_hash(payload["candidate_token_ids"]), "next_token_logits": str(logits_path), "next_token_logits_sha256": file_sha256(logits_path), "next_token_probabilities": str(probability_path), "next_token_probabilities_sha256": file_sha256(probability_path), "next_token_entropy": str(entropy_path), "next_token_entropy_sha256": file_sha256(entropy_path), "model": loaded.metadata})
    print(manifest_path)


if __name__ == "__main__": main()
