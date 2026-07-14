from __future__ import annotations

import argparse
import json

import numpy as np

from .src.matching import match_prefixes, paired_js_from_logits, softmax32
from .src.metrics import LOG_TWO, normalized_js_distance
from .src.utils import atomic_json, atomic_jsonl, ensure_layout, file_sha256, load_config, read_jsonl, stable_hash, stage_is_complete


def assess_match_quality(
    development_distances,
    evaluation_distances,
    random_distances,
    *,
    development_quantile: float,
    maximum_normalized_js: float,
    minimum_random_median_improvement: float,
    minimum_good_evaluation_matches: int,
):
    log_two = LOG_TWO
    development = np.asarray(development_distances, dtype=np.float64)
    evaluation = np.asarray(evaluation_distances, dtype=np.float64)
    random_values = np.asarray(random_distances, dtype=np.float64)
    configured_limit = float(maximum_normalized_js) * log_two
    development_quantile_value = float(np.quantile(development, development_quantile)) if len(development) else configured_limit
    good_threshold = min(development_quantile_value, configured_limit)
    random_median = float(np.median(random_values)) if len(random_values) else float("nan")
    development_median = float(np.median(development)) if len(development) else float("nan")
    evaluation_median = float(np.median(evaluation)) if len(evaluation) else float("nan")
    denominator = max(random_median, 1e-12)
    development_improvement = (random_median - development_median) / denominator if np.isfinite(random_median) and np.isfinite(development_median) else float("nan")
    evaluation_improvement = (random_median - evaluation_median) / denominator if np.isfinite(random_median) and np.isfinite(evaluation_median) else float("nan")
    development_pass = (
        development_median <= configured_limit
        and development_improvement >= minimum_random_median_improvement
    )
    good_evaluation_count = int(np.sum(evaluation <= good_threshold))
    evaluation_pass = (
        good_evaluation_count >= minimum_good_evaluation_matches
        and evaluation_improvement >= minimum_random_median_improvement
    )
    return {
        "good_match_threshold": good_threshold,
        "development_quantile_threshold": development_quantile_value,
        "configured_absolute_threshold": configured_limit,
        "maximum_normalized_js": float(maximum_normalized_js),
        "development_median_js": development_median,
        "evaluation_median_js": evaluation_median,
        "random_median_js": random_median,
        "development_median_normalized_js": normalized_js_distance(development_median),
        "evaluation_median_normalized_js": normalized_js_distance(evaluation_median),
        "random_median_normalized_js": normalized_js_distance(random_median),
        "development_random_median_improvement": development_improvement,
        "evaluation_random_median_improvement": evaluation_improvement,
        "minimum_random_median_improvement": float(minimum_random_median_improvement),
        "good_development_count": int(np.sum(development <= good_threshold)),
        "good_evaluation_count": good_evaluation_count,
        "minimum_good_evaluation_matches": int(minimum_good_evaluation_matches),
        "development_quality_pass": bool(development_pass),
        "evaluation_quality_pass": bool(evaluation_pass),
        "match_quality_pass": bool(development_pass and evaluation_pass),
    }


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); args = parser.parse_args()
    config = load_config(args.config); root = ensure_layout(config)
    prefix_path = root / "prefix_pool/prefixes.jsonl"; logits_path = root / "prefix_pool/next_token_logits.npy"
    inputs = {"prefixes_sha256": file_sha256(prefix_path), "logits_sha256": file_sha256(logits_path)}
    manifest_path = root / "manifests/matches.json"
    if stage_is_complete(manifest_path, config, inputs): print(manifest_path); return
    records = read_jsonl(prefix_path); logits = np.load(logits_path, mmap_mode="r")
    candidate_stage_manifest = json.loads((root / "manifests/candidate_tokens.json").read_text(encoding="utf-8"))
    tokenizer_vocabulary_size = int(candidate_stage_manifest["tokenizer_vocabulary_size"])
    top_token_ids = np.empty((len(records), 20), dtype=np.int64)
    for start in range(0, len(records), 32):
        probabilities = softmax32(logits[start:start+32])
        entropy = -np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, None)), axis=1)
        for offset, value in enumerate(entropy): records[start+offset]["next_token_entropy"] = float(value)
        valid_logits = np.asarray(logits[start:start+32, :tokenizer_vocabulary_size], dtype=np.float32)
        unordered = np.argpartition(valid_logits, -20, axis=1)[:, -20:]
        unordered_logits = np.take_along_axis(valid_logits, unordered, axis=1)
        ordering = np.argsort(-unordered_logits, axis=1, kind="stable")
        top_token_ids[start:start+len(valid_logits)] = np.take_along_axis(unordered, ordering, axis=1)
    donors = np.asarray([i for i, row in enumerate(records) if row["problem_group"] in {"analysis_train", "matching_pool"}], dtype=np.int64)
    development = np.asarray([i for i, row in enumerate(records) if row["problem_group"] == "analysis_dev"], dtype=np.int64)
    evaluation = np.asarray([i for i, row in enumerate(records) if row["problem_group"] == "analysis_test"], dtype=np.int64)
    kwargs = {"candidate_indices": donors, "logits": logits, "top_token_ids": top_token_ids, "tokenizer_vocabulary_size": tokenizer_vocabulary_size}
    dev_matches = match_prefixes(records, query_indices=development, **kwargs)
    test_matches = match_prefixes(records, query_indices=evaluation, **kwargs)
    rng = np.random.default_rng(int(config["seed"]) + 719)
    random_pairs = []
    count = int(config["matching"]["random_pairs"])
    eligible = np.concatenate((development, evaluation))
    attempts = 0
    while len(random_pairs) < count and attempts < count * 20:
        left = int(rng.choice(eligible)); right = int(rng.choice(donors)); attempts += 1
        if records[left]["problem_id"] == records[right]["problem_id"]: continue
        random_pairs.append((left, right))
    random_distances = []
    for start in range(0, len(random_pairs), 32):
        pairs = random_pairs[start:start+32]
        distances = paired_js_from_logits(logits[[left for left, _ in pairs]], logits[[right for _, right in pairs]], tokenizer_vocabulary_size)
        random_distances.extend(map(float, distances))
    quality = assess_match_quality(
        [row["js_distance"] for row in dev_matches if row["matched"]],
        [row["js_distance"] for row in test_matches if row["matched"]],
        random_distances,
        development_quantile=float(config["matching"]["development_good_match_quantile"]),
        maximum_normalized_js=float(config["matching"]["maximum_normalized_js"]),
        minimum_random_median_improvement=float(config["matching"]["minimum_random_median_improvement"]),
        minimum_good_evaluation_matches=int(config["matching"]["minimum_good_evaluation_matches"]),
    )
    threshold = float(quality["good_match_threshold"])
    for row in dev_matches + test_matches:
        row["normalized_js_distance"] = normalized_js_distance(row["js_distance"]) if row["matched"] else None
        row["good_match"] = bool(row["matched"] and row["js_distance"] <= threshold)
        row["split"] = "development" if row in dev_matches else "evaluation"
    matched_evaluation_count = sum(bool(row["matched"]) for row in test_matches)
    matched_development_count = sum(bool(row["matched"]) for row in dev_matches)
    development_coverage = matched_development_count / max(1, len(development))
    evaluation_coverage = matched_evaluation_count / max(1, len(evaluation))
    minimum_development_coverage = float(config["matching"]["minimum_matched_development_coverage"])
    minimum_coverage = float(config["matching"]["minimum_matched_evaluation_coverage"])
    prebranch_gate_pass = bool(quality["match_quality_pass"] and development_coverage >= minimum_development_coverage and evaluation_coverage >= minimum_coverage)
    quality.update({"requested_development_queries": int(len(development)), "matched_development_queries": int(matched_development_count), "matched_development_coverage": development_coverage, "minimum_matched_development_coverage": minimum_development_coverage, "requested_evaluation_queries": int(len(evaluation)), "matched_evaluation_queries": int(matched_evaluation_count), "matched_evaluation_coverage": evaluation_coverage, "minimum_matched_evaluation_coverage": minimum_coverage, "prebranch_matching_gate_pass": prebranch_gate_pass, "matched_control_identifiable": prebranch_gate_pass, "paper_recommendation": "retain Matched control" if prebranch_gate_pass else "drop Matched control and any beyond-current-prediction title/claim"})
    output = root / "matches/prefix_matches.jsonl"; atomic_jsonl(output, dev_matches + test_matches)
    diagnostics = root / "matches/diagnostics.json"
    diagnostic_payload = {**quality, "selection_order": ["same top-1 predicted token", "compute top-5 and top-20 overlap", "same prefix-length and reasoning-progress bins", "different problem ID (hard exclusion)", "maximum top-5 then top-20 overlap within eligible candidates", "minimum full-tokenizer-vocabulary normalized JS"], "normalized_js_definition": "D_NJS = D_JS / log(2)", "absolute_threshold_provenance": "matching.maximum_normalized_js in immutable YAML; fixed before geometry", "threshold_fixed_before_geometry": True, "threshold_source": "minimum of development quantile and preconfigured absolute normalized-JS limit", "js_units": "natural_log_nats", "js_maximum": LOG_TWO, "matching_softmax_dtype": "float64", "matching_source": "saved_float32_next_token_logits restricted to valid tokenizer vocabulary IDs", "tokenizer_vocabulary_size": tokenizer_vocabulary_size, "random_js_distance": random_distances, "development_match_distances": [row["js_distance"] for row in dev_matches if row["matched"]], "evaluation_match_distances": [row["js_distance"] for row in test_matches if row["matched"]], "unmatched_development": [row for row in dev_matches if not row["matched"]], "unmatched_evaluation": [row for row in test_matches if not row["matched"]]}
    atomic_json(diagnostics, diagnostic_payload)
    prebranch_path = root / "matches/prebranch_matching_gate.json"; atomic_json(prebranch_path, {key: diagnostic_payload[key] for key in ("prebranch_matching_gate_pass", "matched_control_identifiable", "paper_recommendation", "requested_development_queries", "matched_development_queries", "matched_development_coverage", "requested_evaluation_queries", "matched_evaluation_queries", "matched_evaluation_coverage", "good_evaluation_count", "maximum_normalized_js", "normalized_js_definition")})
    atomic_json(manifest_path, {"complete": True, "config_hash": stable_hash(config), **inputs, "matches": str(output), "matches_sha256": file_sha256(output), "diagnostics": str(diagnostics), "prebranch_gate": str(prebranch_path), "good_match_threshold": threshold, "match_quality_pass": quality["match_quality_pass"], "prebranch_matching_gate_pass": prebranch_gate_pass})
    print(manifest_path)


if __name__ == "__main__": main()
