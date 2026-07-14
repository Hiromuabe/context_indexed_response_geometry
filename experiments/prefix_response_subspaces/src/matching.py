from __future__ import annotations

import numpy as np

from .metrics import js_divergence


def softmax32(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax for matching diagnostics.

    The historical name is retained for API compatibility, but accumulation
    and output are float64 so low-probability tails do not underflow to zero and
    spuriously push Jensen-Shannon distance to log(2).
    """
    x = np.asarray(logits, dtype=np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    result = np.exp(x)
    result /= result.sum(axis=-1, keepdims=True)
    return result


def paired_js_from_logits(left: np.ndarray, right: np.ndarray, vocabulary_size: int | None = None) -> np.ndarray:
    if vocabulary_size is not None:
        left = np.asarray(left)[..., :int(vocabulary_size)]
        right = np.asarray(right)[..., :int(vocabulary_size)]
    return js_divergence(softmax32(left), softmax32(right))


def match_prefixes(records: list[dict], *, query_indices: np.ndarray, candidate_indices: np.ndarray, logits: np.ndarray, top_token_ids: np.ndarray, tokenizer_vocabulary_size: int) -> list[dict]:
    top_tokens = np.asarray(top_token_ids, dtype=np.int64)
    if top_tokens.ndim != 2 or top_tokens.shape[0] != len(records) or top_tokens.shape[1] < 20:
        raise ValueError("top_token_ids must have shape [prefix, >=20]")
    rows = []
    for query in map(int, query_indices):
        problem_disjoint = [int(k) for k in candidate_indices if records[int(k)]["problem_id"] != records[query]["problem_id"]]
        same_top1_all = [k for k in problem_disjoint if int(top_tokens[k, 0]) == int(top_tokens[query, 0])]
        query_top5, query_top20 = set(top_tokens[query, :5].tolist()), set(top_tokens[query, :20].tolist())
        all_overlaps = {k: (len(query_top5 & set(top_tokens[k, :5].tolist())), len(query_top20 & set(top_tokens[k, :20].tolist()))) for k in same_top1_all}
        same_top1 = [k for k in same_top1_all if records[k]["prefix_length_bin"] == records[query]["prefix_length_bin"] and records[k]["reasoning_progress_bin"] == records[query]["reasoning_progress_bin"]]
        if not same_top1:
            rows.append({"prefix_id": records[query]["prefix_id"], "problem_id": records[query]["problem_id"], "matched_prefix_id": None, "matched_problem_id": None, "js_distance": None, "matched": False, "unmatched_reason": "no different-problem prefix with same top-1, length bin, and progress bin", "problem_disjoint_pool_size": len(problem_disjoint), "same_top1_problem_disjoint_pool_size": len(same_top1_all), "same_top1_same_bin_pool_size": 0, "maximum_top5_overlap": 0, "maximum_top20_overlap": 0, "candidate_pool_size": 0})
            continue
        overlaps = {k: all_overlaps[k] for k in same_top1}
        maximum_top5 = max(value[0] for value in overlaps.values())
        top5_pool = [k for k in same_top1 if overlaps[k][0] == maximum_top5]
        maximum_top20 = max(overlaps[k][1] for k in top5_pool)
        pool = [k for k in top5_pool if overlaps[k][1] == maximum_top20]
        distances = np.empty(len(pool), dtype=np.float64)
        query_logits = np.asarray(logits[query], dtype=np.float32)
        for start in range(0, len(pool), 32):
            chunk = pool[start:start+32]
            distances[start:start+len(chunk)] = paired_js_from_logits(np.repeat(query_logits[None], len(chunk), axis=0), logits[chunk], tokenizer_vocabulary_size)
        best_local = min(range(len(pool)), key=lambda x: (float(distances[x]), str(records[pool[x]]["prefix_id"])))
        matched = pool[best_local]
        rows.append({"prefix_id": records[query]["prefix_id"], "problem_id": records[query]["problem_id"], "matched_prefix_id": records[matched]["prefix_id"], "matched_problem_id": records[matched]["problem_id"], "js_distance": float(distances[best_local]), "matched": True, "unmatched_reason": None, "same_top1_token_id": int(top_tokens[query, 0]), "top5_overlap": int(overlaps[matched][0]), "top20_overlap": int(overlaps[matched][1]), "problem_disjoint_pool_size": len(problem_disjoint), "same_top1_problem_disjoint_pool_size": len(same_top1_all), "same_top1_same_bin_pool_size": len(same_top1), "maximum_top5_overlap": int(maximum_top5), "maximum_top20_overlap": int(maximum_top20), "candidate_pool_size": len(pool)})
    return rows
