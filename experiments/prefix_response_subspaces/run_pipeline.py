from __future__ import annotations

import argparse
import sys

import numpy as np

from . import analyze_functional_recovery, analyze_geometry, build_candidate_tokens, build_prefix_pool, compute_contrast_residuals, extract_successor_states, make_figures, make_tables, match_prefixes
from .src.utils import atomic_json, ensure_layout, load_config, read_json, stable_hash


STAGES = [build_prefix_pool, build_candidate_tokens, match_prefixes, extract_successor_states, compute_contrast_residuals, analyze_geometry, analyze_functional_recovery, make_figures, make_tables]


def _run(module, config_path, model_path):
    prior = sys.argv; sys.argv = [module.__name__, "--config", config_path] + (["--model-path", model_path] if model_path and module in {build_prefix_pool, build_candidate_tokens, extract_successor_states, analyze_functional_recovery} else [])
    try: module.main()
    finally: sys.argv = prior


def finalize(config):
    root = ensure_layout(config); residual = read_json(root / "manifests/residuals.json"); geometry = read_json(root / "metrics/geometry_summary.json")
    match_diagnostics = read_json(root / "matches/diagnostics.json")
    functional_manifest_path = root / "manifests/functional.json"; functional_manifest = read_json(functional_manifest_path) if functional_manifest_path.exists() else {"skipped": True}
    functional = read_json(root / "functional/summary.json") if (root / "functional/summary.json").exists() else {}
    from .src.residualization import centered_residual_ev, explicit_contrast_ev
    from .src.subspaces import top_svd
    first_residual = np.load(residual["entries"][0]["path"])["train_residuals"][0]
    gate_rank = min(4, int(np.linalg.matrix_rank(first_residual)))
    gate_basis = top_svd(first_residual, gate_rank)
    gate0_error = abs(explicit_contrast_ev(first_residual, gate_basis) - centered_residual_ev(first_residual, gate_basis))
    gate0 = gate0_error < 1e-9
    gates = {
        "gate_0_contrast_ev_equivalence": gate0,
        "gate_1_row_centering": bool(residual["gate_1_row_centering"]),
        "gate_2_permutation_recentering": bool(read_json(root / "permutation/null_summary.json")["recentered"]),
        "gate_3_oracle_reinjection": bool(functional.get("oracle_pass", False)),
        "gate_4_delta_global_positive": float(geometry["delta_global"]["mean"]) > 0,
        "gate_5_delta_matched_positive_including_good_matches": bool(match_diagnostics.get("prebranch_matching_gate_pass", False)) and int(match_diagnostics.get("good_evaluation_count", 0)) >= int(match_diagnostics.get("minimum_good_evaluation_matches", 1)) and float(geometry["delta_matched"]["mean"]) > 0 and float(geometry["delta_matched_good_matches"]["mean"]) > 0,
        "gate_6_delta_content_positive": float(geometry["delta_content"]["mean"]) > 0,
        "gate_7a_local_gain_positive_and_above_global": bool(functional) and bool(functional.get("functional_global_claim_eligible", False)) and float(functional.get("G_local", {}).get("mean", float("-inf"))) > 0 and float(functional.get("G_local_minus_global", {}).get("mean", float("-inf"))) > 0,
        "gate_7b_valid_matching_and_local_above_matched": bool(functional) and bool(functional.get("functional_matched_claim_eligible", False)) and float(functional.get("G_local_minus_matched", {}).get("mean", float("-inf"))) > 0,
        "diagnostic_prebranch_matching_identifiable": bool(match_diagnostics.get("prebranch_matching_gate_pass", False)),
        "diagnostic_permutation_inference_valid": bool(geometry.get("permutation_exchangeability", {}).get("permutation_inference_valid", False)),
        "diagnostic_rank0_stability_valid": bool(functional.get("rank0_stability_pass", False)),
    }
    atomic_json(root / "gate_results.json", gates)
    hidden = read_json(root / "manifests/hidden_states.json"); candidates = read_json(root / "candidate_tokens/candidate_tokens.json"); matches = match_diagnostics
    lines = ["# Prefix-response-subspace summary", "", f"- Model: {hidden['model']['model_source']} ({hidden['model']['resolved_revision']})", f"- Prefixes extracted: {sum(item['shape'][0] for item in hidden['layers'][:1])}", f"- Candidate tokens: {len(candidates['candidate_token_ids'])}", f"- M/V split: {len(candidates['calibration_indices'])}/{len(candidates['analysis_indices'])}", f"- Layers: {[item['layer'] for item in hidden['layers']]}", f"- Primary rank: {config['analysis']['primary_rank']}", "- Normalized JS definition: D_NJS = D_JS / log(2)", f"- Good-match threshold (development, JS nats): {matches['good_match_threshold']:.6g}", f"- Fixed absolute normalized-JS ceiling: {matches.get('maximum_normalized_js')}", f"- Match median normalized JS: dev={matches.get('development_median_normalized_js')}, evaluation={matches.get('evaluation_median_normalized_js')}, random={matches.get('random_median_normalized_js')}", f"- Pre-branch matched-control identifiability: {'PASS' if matches.get('prebranch_matching_gate_pass') else 'FAIL'} (matched={matches.get('matched_evaluation_queries')}/{matches.get('requested_evaluation_queries')}, good={matches.get('good_evaluation_count')})", f"- Paper recommendation: {matches.get('paper_recommendation')}", "", "## Geometry", ""]
    for key in ("delta_global", "delta_matched", "delta_matched_good_matches", "delta_content"): lines.append(f"- {key}: {geometry[key]}")
    lines.append(f"- Continuous NJS–Delta_matched relationship: {geometry.get('matched_js_delta_continuous_relationship')}")
    primary_top_k = int(config["analysis"]["high_probability_primary_top_k"])
    lines += ["", "## Sensitivity and null analyses", "", f"- Outlier-removed delta_content: {geometry.get('delta_content_outlier_removed')}", f"- High-probability definition: {geometry.get('high_probability_definition')}", f"- Top-{primary_top_k} deltas: global={geometry.get(f'delta_global_top{primary_top_k}')}, matched={geometry.get(f'delta_matched_top{primary_top_k}')}, content={geometry.get(f'delta_content_top{primary_top_k}')}", f"- Permutation exchangeability: {geometry.get('permutation_exchangeability')}", f"- Permutation p-values (NaN when exchangeability audit fails): global={geometry.get('delta_global_permutation_p')}, matched={geometry.get('delta_matched_permutation_p')}, content={geometry.get('delta_content_permutation_p')}", "", "## Functional", "", json_string(functional if functional else {"status": "not run because Experiment 1 gate failed"}), "", "## Gates and diagnostics", ""] + [f"- {key}: {'PASS' if value else 'FAIL/NOT RUN'}" for key, value in gates.items()]
    (root / "summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def json_string(value):
    import json
    return "```json\n" + json.dumps(value, indent=2, sort_keys=True) + "\n```"


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--model-path"); parser.add_argument("--from-stage", choices=[m.__name__.split(".")[-1] for m in STAGES]); parser.add_argument("--through-stage", choices=[m.__name__.split(".")[-1] for m in STAGES]); args = parser.parse_args()
    names = [m.__name__.split(".")[-1] for m in STAGES]; start = names.index(args.from_stage) if args.from_stage else 0; stop = names.index(args.through_stage)+1 if args.through_stage else len(STAGES)
    for module in STAGES[start:stop]: _run(module, args.config, args.model_path)
    if stop == len(STAGES): finalize(load_config(args.config))


if __name__ == "__main__": main()
