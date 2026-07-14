from __future__ import annotations

import argparse
import os
import sys
import time

# Rank-0 FP32 QA is the last stage, but cuBLAS determinism has to be configured
# before any earlier stage initializes CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from . import analyze_first_layer_mechanism, analyze_optimal_value_control, analyze_optimal_value_functional, analyze_paper_functional, analyze_paper_geometry, analyze_rank_saturation, analyze_shared_backbone, analyze_value_space_functional, build_candidate_tokens, build_prefix_pool, compute_contrast_residuals, extract_first_layer_mechanism, extract_successor_states, extract_value_output_spans, make_paper_figures, make_paper_tables, run_paper_replication, select_wrong_prefixes, verify_rank0_outliers
from .refresh_paper_geometry import refresh_geometry
from .src.utils import atomic_json, ensure_layout, load_config, read_json


ADDITIONAL_STAGES=[analyze_rank_saturation,analyze_shared_backbone,extract_first_layer_mechanism,analyze_first_layer_mechanism,extract_value_output_spans,analyze_optimal_value_control]
STAGES=[build_prefix_pool,build_candidate_tokens,select_wrong_prefixes,extract_successor_states,compute_contrast_residuals,analyze_paper_geometry,*ADDITIONAL_STAGES,analyze_paper_functional,analyze_value_space_functional,analyze_optimal_value_functional,verify_rank0_outliers,make_paper_figures,make_paper_tables]


def _require_recomputable_analysis(config):
    root=ensure_layout(config)
    required=[root/"manifests/hidden_states.json",root/"manifests/residuals.json",root/"manifests/candidate_tokens.json",root/"candidate_tokens/candidate_tokens.json",root/"controls/wrong_prefixes.jsonl"]
    missing=[str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            "--recompute-analysis requires an already completed paper branching "
            f"run under {root}. Missing: {missing}. For the first paper run, "
            "remove --recompute-analysis. Artifacts from a different results_root "
            "are not silently mixed because their prefix/token axes may differ."
        )


def _run(module,config_path,model_path,force=False,skip_content=False):
    previous=sys.argv; sys.argv=[module.__name__,"--config",config_path]+(["--model-path",model_path] if model_path and module in {build_prefix_pool,build_candidate_tokens,extract_successor_states,extract_first_layer_mechanism,extract_value_output_spans,analyze_paper_functional,analyze_value_space_functional,analyze_optimal_value_functional,verify_rank0_outliers,run_paper_replication} else [])+(["--force"] if force and module in {analyze_paper_geometry,analyze_rank_saturation,analyze_shared_backbone,analyze_first_layer_mechanism,extract_value_output_spans,analyze_optimal_value_control,analyze_paper_functional,analyze_value_space_functional,analyze_optimal_value_functional,verify_rank0_outliers,run_paper_replication} else [])+(["--skip-content-appendix"] if skip_content and module in {analyze_paper_geometry,run_paper_replication} else [])
    name=module.__name__.split(".")[-1]; started=time.monotonic(); print(f"[paper_pipeline] START {name}",flush=True)
    try: module.main(); print(f"[paper_pipeline] DONE {name} elapsed={(time.monotonic()-started)/60:.1f}m",flush=True)
    finally: sys.argv=previous


def _finalize(config):
    root=ensure_layout(config); geometry=read_json(root/"metrics/paper_geometry_summary.json")
    if "exact_fraction" not in geometry.get("conditional_global_coverage",{}) or "eligible_exact_both_fraction" not in geometry.get("top_k_coverage",{}).get(str(int(config["analysis"]["high_probability_primary_top_k"])),{}):
        refresh_geometry(config,root); geometry=read_json(root/"metrics/paper_geometry_summary.json")
    functional=read_json(root/"functional/paper_summary.json"); top_k=int(config["analysis"]["high_probability_primary_top_k"])
    gates={
        "experiment1_conditional_global_exact_coverage_ge_99pct":float(geometry["conditional_global_coverage"]["exact_fraction"])>=.99,
        "experiment1_conditional_global_fallback_complete":bool(geometry["conditional_global_coverage"]["complete_with_fallback"]),
        "experiment1_wrong_controls_complete":bool(geometry["wrong_control_diagnostics"]["all_evaluation_targets_complete"]) and bool(geometry["wrong_basis_coverage"]["complete"]),
        "experiment1_wrong_exact_bin_coverage_ge_99pct":float(geometry["wrong_exact_bin_coverage"]["fraction"])>=.99,
        "experiment1_rotation_coverage_complete":bool(geometry["rotation_coverage"]["complete"]),
        "experiment1_between_rotation_above_within_noise":bool(geometry["rotation_coverage"]["complete"]) and float(geometry["R_between_minus_within"]["ci_low"])>0,
        "experiment1_low_dimensional_r90_median_le_32":float(geometry["r90"]["median"])<=32,
        "experiment1_local_above_conditional_global":float(geometry["delta_conditional_global"]["mean"])>0,
        "experiment1_local_above_wrong_prefix":float(geometry["delta_wrong_exact_bin"]["mean"])>0,
        "experiment1_top256_effect_positive_on_eligible_subset":float(geometry[f"delta_conditional_global_top{top_k}"]["ci_low"])>0 and float(geometry[f"delta_wrong_top{top_k}_exact_bin"]["ci_low"])>0,
        "diagnostic_top256_exact_coverage_ge_99pct":float(geometry["top_k_coverage"][str(top_k)]["eligible_exact_both_fraction"])>=.99,
        "experiment1_permutation_conditional_global":float(geometry.get("delta_conditional_global_permutation_p",1))<=.05,
        "experiment1_permutation_wrong_prefix":float(geometry.get("delta_wrong_permutation_p",1))<=.05,
        "experiment2_oracle":bool(functional["oracle_pass"]),
        "experiment2_control_coverage_complete":bool(functional["control_coverage"]["complete"]),
        "experiment2_local_gain_positive":bool(functional["gate_functional_local_positive"]),
        "experiment2_local_above_conditional_global":bool(functional["gate_functional_local_above_conditional_global"]),
        "experiment2_local_above_wrong_prefix":bool(functional["gate_functional_local_above_wrong"]),
    }; atomic_json(root/"paper_gate_results.json",gates)
    additional={}
    for name,path in {
        "rank_saturation":root/"metrics/rank_saturation_summary.json",
        "shared_backbone":root/"metrics/shared_backbone_summary.json",
        "first_layer_mechanism":root/"metrics/first_layer_mechanism_summary.json",
        "value_space_functional":root/"functional/value_space_summary.json",
        "optimal_value_control":root/"metrics/optimal_value_control_summary.json",
        "optimal_value_functional":root/"functional/optimal_value_summary.json",
        "rank0_outlier_fp32":root/"functional/rank0_outlier_fp32_summary.json",
    }.items():
        additional[name]=read_json(path) if path.exists() else {"status":"not run"}
    if additional["rank_saturation"].get("status")!="not run":
        gates["additional_rank_saturation_compact_supported"]=bool(additional["rank_saturation"]["claim_compact_supported"])
    if additional["shared_backbone"].get("status")!="not run":
        gates["additional_specificity_after_shared_removal"]=bool(additional["shared_backbone"]["gate_prefix_specificity_remains_after_shared_removal"])
    atomic_json(root/"paper_gate_results.json",gates)
    replication_path=root/"fixed_replications/summary.json"; replication=read_json(replication_path) if replication_path.exists() else {"status":"not run"}; top_coverage=geometry["top_k_coverage"][str(top_k)]; lines=["# Final paper experiment summary","",f"- Main model: {config['model']['checkpoint']}",f"- Selected layer/rank: {geometry['selected_layer']}/{geometry['selected_rank']}",f"- r90: {geometry['r90']}","","## Experiment 1","",f"- Delta conditional Global (exact bins): {geometry['delta_conditional_global']}",f"- Delta Wrong-prefix: {geometry['delta_wrong']}",f"- Top-{top_k} pooled eligible subset: {top_coverage['eligible_exact_both_prefixes']}/{top_coverage['expected_prefixes']} exact prefixes; Global={geometry[f'delta_conditional_global_top{top_k}']}, Wrong exact={geometry[f'delta_wrong_top{top_k}_exact_bin']}","- Top-k effects are a coverage-limited sensitivity analysis, not evidence for all evaluation prefixes.",f"- Rotation Local/Global: {geometry['d_rotation_local_conditional_global']}",f"- Rotation Local/Wrong: {geometry['d_rotation_local_wrong_mean']}",f"- Split-half R_between - R_within: {geometry['R_between_minus_within']}","","## Experiment 2","",f"```json\n{__import__('json').dumps(functional,indent=2,sort_keys=True)}\n```","","## Additional mechanism and structure analyses","",f"```json\n{__import__('json').dumps(additional,indent=2,sort_keys=True)}\n```","","## Fixed-condition cross-model replication","",f"```json\n{__import__('json').dumps(replication,indent=2,sort_keys=True)}\n```","","## Gates",""]+[f"- {key}: {'PASS' if value else 'FAIL'}" for key,value in gates.items()]; (root/"paper_summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--model-path"); parser.add_argument("--replication-model-path",help="NAME=PATH; run run_paper_replication directly to supply multiple model paths"); parser.add_argument("--from-stage",choices=[m.__name__.split(".")[-1] for m in STAGES]); parser.add_argument("--through-stage",choices=[m.__name__.split(".")[-1] for m in STAGES]); parser.add_argument("--skip-replication",action="store_true"); parser.add_argument("--skip-additional-experiments",action="store_true"); parser.add_argument("--recompute-analysis",action="store_true"); parser.add_argument("--skip-content-appendix",action="store_true"); args=parser.parse_args(); active_stages=[module for module in STAGES if not (args.skip_additional_experiments and module in {*ADDITIONAL_STAGES,analyze_value_space_functional,analyze_optimal_value_functional,verify_rank0_outliers})]; names=[m.__name__.split(".")[-1] for m in active_stages]
    if args.recompute_analysis and (args.from_stage or args.through_stage): raise ValueError("--recompute-analysis cannot be combined with stage bounds")
    if args.recompute_analysis: _require_recomputable_analysis(load_config(args.config))
    start=names.index("analyze_paper_geometry") if args.recompute_analysis else (names.index(args.from_stage) if args.from_stage else 0); stop=names.index(args.through_stage)+1 if args.through_stage else len(active_stages)
    for module in active_stages[start:stop]: _run(module,args.config,args.model_path,args.recompute_analysis,args.skip_content_appendix)
    if stop==len(active_stages):
        config=load_config(args.config)
        if bool(config.get("replication",{}).get("enabled",False)) and not args.skip_replication: _run(run_paper_replication,args.config,args.replication_model_path,args.recompute_analysis,args.skip_content_appendix)
        _finalize(config)


if __name__=="__main__": main()
