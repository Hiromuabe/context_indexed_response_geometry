from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .analyze_paper_geometry import _select_primary
from .refresh_paper_geometry import refresh_geometry
from .src.statistics import permutation_pvalue, problem_bootstrap
from .src.utils import atomic_json, ensure_layout, file_sha256, load_config, read_json, stable_hash


STRING_KEYS={"problem_id","prefix_id","split","conditional_stratum","conditional_global_resolved_stratum"}
BOOL_KEYS={"wrong_control_exact_length_bin","conditional_global_exact_length_bin"}


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle: raw=list(csv.DictReader(handle))
    rows=[]
    for source in raw:
        row={}
        for key,value in source.items():
            if key in STRING_KEYS: row[key]=value
            elif key in BOOL_KEYS: row[key]=value.lower()=="true"
            elif value=="": row[key]=float("nan")
            elif key in {"layer","fold","rank","effective_local_rank","wrong_prefix_count","split_a_tokens","split_b_tokens","split_half_effective_rank","local_global_effective_rank","local_wrong_min_effective_rank","between_min_effective_rank"} or key.startswith("top") and key.endswith("_token_count"): row[key]=int(value)
            else:
                try: row[key]=float(value)
                except ValueError: row[key]=value
        rows.append(row)
    return rows


def main() -> None:
    parser=argparse.ArgumentParser(description="Finalize paper geometry from CSV/null artifacts after a late JSON-write failure")
    parser.add_argument("--config",required=True); args=parser.parse_args(); config=load_config(args.config); root=ensure_layout(config)
    if bool(config.get("replication_mode",False)): raise ValueError("late finalization currently targets the main paper run")
    rows_path=root/"metrics/paper_geometry_rows.csv"; rotation_path=root/"metrics/paper_rotation_rows.csv"; energy_path=root/"metrics/interaction_energy.csv"; permutation_path=root/"permutation/paper_null_summary.json"
    required=[rows_path,rotation_path,energy_path,permutation_path]
    missing=[str(path) for path in required if not path.is_file()]
    if missing: raise FileNotFoundError(f"cannot finalize; missing late-stage artifacts: {missing}")
    all_rows=_rows(rows_path); rotation_rows=_rows(rotation_path); energy_rows=_rows(energy_path); permutation=read_json(permutation_path)
    ranks=list(map(int,config["analysis"]["ranks"])); top_ks=list(map(int,config["analysis"]["high_probability_top_ks"])); high_minimum=int(config["analysis"]["high_probability_min_tokens"])
    layers=sorted({int(row["layer"]) for row in all_rows}); dev_rows=[row for row in all_rows if row["split"]=="development" and bool(row.get("conditional_global_exact_length_bin",True))]
    selected_layer,selected_rank,selection_diagnostics=_select_primary(dev_rows,layers,ranks,int(config["analysis"]["selection_rank"]),float(config["analysis"]["r90_fraction"]))
    test_primary=[row for row in all_rows if row["split"]=="evaluation" and int(row["layer"])==selected_layer and int(row["rank"])==selected_rank]
    expected_rows=int(config["data"]["evaluation_prefixes"])*int(config["candidates"]["folds"]); expected_wrong=int(config["controls"]["wrong_prefixes_per_target"])
    complete_wrong=sum(int(row["wrong_prefix_count"])==expected_wrong for row in test_primary); complete_rotation=sum(int(row["wrong_prefix_count"])==expected_wrong and int(row["split_half_effective_rank"])==selected_rank and int(row["between_min_effective_rank"])==selected_rank and np.isfinite(row["R_between_minus_within"]) for row in rotation_rows)
    wrong_diagnostics=read_json(root/"manifests/wrong_prefixes.json")["diagnostics"]
    summary={"selected_layer":selected_layer,"selected_rank":selected_rank,"selection_split":"analysis_dev","selection_rules":config["selection"],"selection_diagnostics":selection_diagnostics,"content_appendix_computed":any("delta_content_appendix" in row and np.isfinite(row["delta_content_appendix"]) for row in all_rows),"conditional_global_definition":{"covariance":"mean_i K_i / trace(K_i)","conditioning":["prefix_length_bin","reasoning_progress_bin"],"prefix_weighting":"equal after total-response-energy normalization"},"rotation_distance_definition":"1 - ||U_i^T U_k||_F^2 / r","conditional_global_coverage":{"expected_prefix_fold_rows":expected_rows,"observed_prefix_fold_rows":len(test_primary),"complete":len(test_primary)==expected_rows},"wrong_basis_coverage":{"expected_prefix_fold_rows":expected_rows,"complete_prefix_fold_rows":complete_wrong,"wrong_prefixes_per_target":expected_wrong,"complete":len(test_primary)==expected_rows and complete_wrong==expected_rows},"rotation_coverage":{"expected_prefix_fold_rows":expected_rows,"observed_prefix_fold_rows":len(rotation_rows),"complete_primary_rank_rows":complete_rotation,"complete":len(rotation_rows)==expected_rows and complete_rotation==expected_rows},"wrong_control_diagnostics":wrong_diagnostics,"interaction_energy_by_layer":energy_rows}
    bootstrap_args={"replicates":int(config["statistics"]["bootstrap_replicates"]),"seed":int(config["seed"]),"ci":float(config["statistics"]["ci"])}
    exact_conditional=[row for row in test_primary if bool(row.get("conditional_global_exact_length_bin",True))]
    summary["delta_conditional_global"]=problem_bootstrap(np.asarray([row["delta_conditional_global"] for row in exact_conditional]),np.asarray([row["problem_id"] for row in exact_conditional]),**bootstrap_args)
    for metric in ("delta_wrong",*[name for cutoff in top_ks for name in (f"delta_conditional_global_top{cutoff}",f"delta_wrong_top{cutoff}")]): summary[metric]=problem_bootstrap(np.asarray([row[metric] for row in test_primary]),np.asarray([row["problem_id"] for row in test_primary]),**bootstrap_args)
    exact=[row for row in test_primary if bool(row["wrong_control_exact_length_bin"])]; exact_problems={row["problem_id"] for row in exact}; summary["wrong_exact_bin_coverage"]={"expected_problems":int(config["data"]["evaluation_prefixes"]),"exact_bin_problems":len(exact_problems),"exact_bin_prefix_fold_rows":len(exact),"fraction":len(exact_problems)/max(1,int(config["data"]["evaluation_prefixes"]))}
    for metric in ("delta_wrong",*[f"delta_wrong_top{cutoff}" for cutoff in top_ks]): summary[f"{metric}_exact_bin"]=problem_bootstrap(np.asarray([row[metric] for row in exact]),np.asarray([row["problem_id"] for row in exact]),replicates=bootstrap_args["replicates"],seed=int(config["seed"])+43,ci=bootstrap_args["ci"])
    summary["top_k_coverage"]={}
    for cutoff in top_ks:
        eligible=[row for row in test_primary if int(row[f"top{cutoff}_token_count"])>=high_minimum]; problems={row["problem_id"] for row in eligible}; summary["top_k_coverage"][str(cutoff)]={"minimum_tokens_per_prefix_fold":high_minimum,"expected_prefix_fold_rows":expected_rows,"eligible_prefix_fold_rows":len(eligible),"expected_problems":int(config["data"]["evaluation_prefixes"]),"eligible_problems":len(problems),"complete":len(eligible)==expected_rows and len(problems)==int(config["data"]["evaluation_prefixes"])}
    for metric in ("d_rotation_local_conditional_global","d_rotation_local_wrong_mean","R_within","R_between","R_between_minus_within"): summary[metric]=problem_bootstrap(np.asarray([row[metric] for row in rotation_rows]),np.asarray([row["problem_id"] for row in rotation_rows]),replicates=bootstrap_args["replicates"],seed=int(config["seed"])+271,ci=bootstrap_args["ci"])
    r90=[]
    for prefix_id in sorted({row["prefix_id"] for row in all_rows if row["split"]=="evaluation"}):
        curve={rank:np.mean([row["ev_local"] for row in all_rows if row["split"]=="evaluation" and int(row["layer"])==selected_layer and int(row["rank"])==rank and row["prefix_id"]==prefix_id]) for rank in ranks}; reference=curve[max(ranks)]; achieved=[rank for rank in ranks if curve[rank]>=float(config["analysis"]["r90_fraction"])*reference]; r90.append({"prefix_id":prefix_id,"r90":min(achieved) if achieved else None,"ev_rank64":reference})
    summary["r90"]={"median":float(np.median([row["r90"] for row in r90 if row["r90"] is not None])),"fraction_le_32":float(np.mean([row["r90"]<=32 for row in r90 if row["r90"] is not None])),"rows":r90}
    diagnostics=permutation["diagnostics"]; summary["permutation_diagnostics"]=diagnostics
    for metric in ("delta_conditional_global","delta_wrong"):
        observed=summary["delta_wrong_exact_bin"]["mean"] if metric=="delta_wrong" else summary[metric]["mean"]; raw=permutation_pvalue(observed,np.asarray(permutation["null"][metric])); summary[f"{metric}_permutation_p_raw"]=raw; summary[f"{metric}_permutation_p"]=raw if diagnostics["permutation_inference_valid"] else float("nan")
    summary_path=root/"metrics/paper_geometry_summary.json"; atomic_json(summary_path,summary)
    residual_path=root/"manifests/residuals.json"; hidden_path=root/"manifests/hidden_states.json"; wrong_path=root/"controls/wrong_prefixes.jsonl"; inputs={"residuals_sha256":file_sha256(residual_path),"hidden_states_sha256":file_sha256(hidden_path),"wrong_prefixes_sha256":file_sha256(wrong_path),"content_appendix_mode":"compute" if summary["content_appendix_computed"] else "skip"}
    manifest_path=root/"manifests/paper_geometry.json"; atomic_json(manifest_path,{"complete":True,"config_hash":stable_hash(config),**inputs,"rows":str(rows_path),"rows_sha256":file_sha256(rows_path),"rotation_rows":str(rotation_path),"rotation_rows_sha256":file_sha256(rotation_path),"energy":str(energy_path),"summary":str(summary_path),"permutation":str(permutation_path),"selected_layer":selected_layer,"selected_rank":selected_rank,"recovered_from_late_json_failure":True}); print(refresh_geometry(config,root))


if __name__=="__main__": main()
