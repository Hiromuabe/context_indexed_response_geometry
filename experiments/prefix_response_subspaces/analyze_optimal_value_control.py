from __future__ import annotations

import argparse, csv
import numpy as np

from .analyze_paper_geometry import _fit_controls
from .src.residualization import center_train_and_evaluation
from .src.statistics import problem_bootstrap
from .src.subspaces import explained_variance, top_svd
from .src.utils import atomic_json, ensure_layout, file_sha256, load_config, read_json, read_jsonl, stable_hash, stage_is_complete


def _write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows: path.write_text("",encoding="utf-8"); return
    with path.open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def optimal_basis(train: np.ndarray, span: np.ndarray | None, rank: int) -> tuple[np.ndarray,int,bool]:
    """Best response-energy rank-r basis constrained to ``span``."""
    local=top_svd(train,rank,allow_rank_reduction=True)
    if span is None:
        return local,int(local.shape[1]),True
    c=np.asarray(span,dtype=np.float64)
    coordinates=train@c
    within=top_svd(coordinates,min(rank,c.shape[1]),allow_rank_reduction=True)
    basis=c@within; basis=np.linalg.qr(basis,mode="reduced")[0]
    return basis,int(basis.shape[1]),False


def full_span_energy_fraction(target: np.ndarray, span: np.ndarray | None) -> float:
    """Fraction of held-out interaction energy retained in the complete value span."""
    target=np.asarray(target,dtype=np.float64)
    denominator=float(np.square(target).sum())
    if denominator<=1e-12: return float("nan")
    if span is None: return 1.0
    return explained_variance(target,span)


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--force",action="store_true"); args=parser.parse_args()
    config=load_config(args.config); root=ensure_layout(config)
    paths={"spans":root/"manifests/value_output_spans.json","mechanism":root/"manifests/first_layer_mechanism_states.json","candidates":root/"candidate_tokens/candidate_tokens.json","hidden":root/"manifests/hidden_states.json"}
    inputs={f"{k}_sha256":file_sha256(v) for k,v in paths.items()}; manifest=root/"manifests/optimal_value_control.json"
    if not args.force and stage_is_complete(manifest,config,inputs): print(manifest); return
    spans=read_json(paths["spans"]); mechanism=read_json(paths["mechanism"]); candidates=read_json(paths["candidates"]); hidden=read_json(paths["hidden"]); prefixes=read_jsonl(hidden["prefix_snapshot"])
    span_rows={int(r["prefix_index"]):r for r in spans["rows"]}; arrays={e["site"]:np.load(e["path"],mmap_mode="r") for e in mechanism["sites"]}
    auxiliary=np.asarray([i for i,p in enumerate(prefixes) if p["problem_group"]=="auxiliary"],dtype=np.int64); nonaux=np.asarray([i for i,p in enumerate(prefixes) if p["problem_group"]!="auxiliary"],dtype=np.int64)
    rank=int(config.get("first_layer_mechanism",{}).get("value_space_rank",64)); rows=[]; basis_root=root/"subspaces/optimal_value"; basis_root.mkdir(parents=True,exist_ok=True)
    for site,z in arrays.items():
        for fold in candidates["folds"]:
            fold_id=int(fold["fold_id"]); train,held=center_train_and_evaluation(z[nonaux],z[auxiliary],fold["train_indices"],fold["evaluation_indices"]); local,_=_fit_controls(train.residuals,prefixes,nonaux,rank)
            for axis,full in enumerate(nonaux):
                p=prefixes[int(full)]
                if p["problem_group"] not in {"analysis_dev","analysis_test"}: continue
                info=span_rows.get(int(full))
                if info is None: continue
                span=None if info["full_hidden_space"] else np.load(info["path"])
                ustar,effective,identical=optimal_basis(train.residuals[axis],span,rank); uloc=local[p["prefix_id"]][:,:effective]; target=held.residuals[axis]
                ev_local=explained_variance(target,uloc); ev_value=explained_variance(target,ustar); ev_full_span=full_span_energy_fraction(target,span)
                path=""
                if site=="post_mlp":
                    target_path=basis_root/f"site_post_mlp_fold_{fold_id}_prefix_{int(full):05d}.npy"; np.save(target_path,ustar.astype(np.float32)); path=str(target_path)
                rows.append({"problem_id":p["problem_id"],"prefix_id":p["prefix_id"],"prefix_index":int(full),"split":"development" if p["problem_group"]=="analysis_dev" else "evaluation","site":site,"fold":fold_id,"requested_rank":rank,"effective_rank":effective,"value_span_rank":int(info["output_span_rank"]),"value_span_is_full_hidden":bool(info["full_hidden_space"]),"optimal_value_identical_to_local_by_construction":identical,"ev_local":ev_local,"ev_optimal_value":ev_value,"delta_local_optimal_value":ev_local-ev_value,"full_value_span_interaction_fraction":ev_full_span,"outside_value_span_interaction_fraction":1.0-ev_full_span,"basis_path":path,"basis_sha256":file_sha256(path) if path else ""})
            print(f"[optimal_value_geometry] site={site} fold={fold_id}",flush=True)
    bootstrap={"replicates":int(config["statistics"]["bootstrap_replicates"]),"seed":int(config["seed"])+2411,"ci":float(config["statistics"]["ci"])}; summary={"rank":rank,"sites":{},"span_diagnostics":{"prefixes":len(span_rows),"full_hidden_space_prefixes":sum(r["full_hidden_space"] for r in span_rows.values()),"full_hidden_space_fraction":sum(r["full_hidden_space"] for r in span_rows.values())/max(1,len(span_rows)),"output_span_rank_min":min(r["output_span_rank"] for r in span_rows.values()),"output_span_rank_median":float(np.median([r["output_span_rank"] for r in span_rows.values()])),"output_span_rank_max":max(r["output_span_rank"] for r in span_rows.values())}}
    for site in arrays:
        selected=[r for r in rows if r["site"]==site and r["split"]=="evaluation"]
        estimate=problem_bootstrap(np.asarray([r["delta_local_optimal_value"] for r in selected]),np.asarray([r["problem_id"] for r in selected]),**bootstrap)
        retained=problem_bootstrap(np.asarray([r["full_value_span_interaction_fraction"] for r in selected]),np.asarray([r["problem_id"] for r in selected]),**{**bootstrap,"seed":bootstrap["seed"]+1})
        outside=problem_bootstrap(np.asarray([r["outside_value_span_interaction_fraction"] for r in selected]),np.asarray([r["problem_id"] for r in selected]),**{**bootstrap,"seed":bootstrap["seed"]+2})
        informative=[r for r in selected if not r["optimal_value_identical_to_local_by_construction"]]
        summary["sites"][site]={"delta_EV_local_minus_optimal_value":estimate,"full_value_span_interaction_fraction":retained,"outside_value_span_interaction_fraction":outside,"evaluation_rows":len(selected),"informative_rows":len(informative),"claim_supported":bool(informative) and float(estimate["ci_low"])>0}
    rows_path=root/"metrics/optimal_value_control_rows.csv"; summary_path=root/"metrics/optimal_value_control_summary.json"; _write_csv(rows_path,rows); atomic_json(summary_path,summary); atomic_json(manifest,{"complete":True,"config_hash":stable_hash(config),**inputs,"rows":str(rows_path),"rows_sha256":file_sha256(rows_path),"summary":str(summary_path),"summary_sha256":file_sha256(summary_path)}); print(manifest)


if __name__=="__main__": main()
