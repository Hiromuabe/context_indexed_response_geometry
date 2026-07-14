from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict

import numpy as np

from prefix_displacement.runtime import prepare_data_parallel
from experiments.prefix_successor_subspaces.src.model import _load_backbone_and_tokenizer

from .analyze_functional_recovery import FunctionalRecoveryForward, problem_aggregated_mean, rank0_anchor_max_difference
from .analyze_paper_geometry import _fit_controls, _resolve_conditional_basis
from .src.data import pad_token_rows
from .src.statistics import problem_bootstrap, problem_ratio_bootstrap
from .src.utils import atomic_json, ensure_layout, file_sha256, load_config, read_json, read_jsonl, stable_hash, stage_is_complete


FUNCTIONAL_CHECKPOINT_VERSION=5


def summarize_absolute_recovery(rows, config):
    """Summarize absolute JS distances and Rank-0-normalized recovery."""
    ids=np.asarray([row["problem_id"] for row in rows])
    bootstrap={"replicates":int(config["statistics"]["bootstrap_replicates"]),"seed":int(config["seed"])+1701,"ci":float(config["statistics"]["ci"])}
    result={
        "functional_scale_definition":{
            "distance":"Jensen-Shannon distance from the unmodified next-token distribution",
            "recovery_fraction":"sum(D_rank0 - D_condition) / sum(D_rank0)",
            "bootstrap_unit":"problem_id",
            "note":"Ratio-of-totals is used instead of the mean cell-wise ratio to avoid instability at small D_rank0.",
        }
    }
    for key in ("D_oracle","D_rank0","D_local","D_conditional_global","D_wrong_mean"):
        values=np.asarray([float(row[key]) for row in rows],dtype=np.float64)
        result[key]=problem_bootstrap(values,ids,**bootstrap)
    d0=np.asarray([float(row["D_rank0"]) for row in rows],dtype=np.float64)
    for label,key in (("local","D_local"),("conditional_global","D_conditional_global"),("wrong_mean","D_wrong_mean")):
        distance=np.asarray([float(row[key]) for row in rows],dtype=np.float64)
        result[f"recovery_fraction_{label}"]=problem_ratio_bootstrap(d0-distance,d0,ids,**{**bootstrap,"seed":bootstrap["seed"]+len(result)})
    return result


def refresh_summary_only(root, config):
    rows_path=root/"functional/paper_cell_summary.csv"; summary_path=root/"functional/paper_summary.json"
    if not rows_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"summary-only refresh requires {rows_path} and {summary_path}")
    with rows_path.open(newline="",encoding="utf-8") as handle:
        rows=list(csv.DictReader(handle))
    summary=read_json(summary_path); summary.update(summarize_absolute_recovery(rows,config)); atomic_json(summary_path,summary); print(summary_path)


def _write_csv(path, rows):
    if not rows: path.write_text("",encoding="utf-8"); return
    keys=[]
    for row in rows:
        for key in row:
            if key not in keys: keys.append(key)
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--model-path"); parser.add_argument("--force",action="store_true"); parser.add_argument("--preflight-only",action="store_true"); parser.add_argument("--summary-only",action="store_true",help="refresh absolute-distance and recovery-fraction summaries from saved CSV rows without model forward"); args=parser.parse_args(); config=load_config(args.config); root=ensure_layout(config)
    if args.summary_only:
        refresh_summary_only(root,config); return
    geometry_path=root/"metrics/paper_geometry_summary.json"; hidden_path=root/"manifests/hidden_states.json"; residual_path=root/"manifests/residuals.json"; wrong_path=root/"controls/wrong_prefixes.jsonl"
    inputs={"geometry_sha256":file_sha256(geometry_path),"hidden_sha256":file_sha256(hidden_path),"residuals_sha256":file_sha256(residual_path),"wrong_prefixes_sha256":file_sha256(wrong_path)}; manifest_path=root/"manifests/paper_functional.json"
    if not args.force and stage_is_complete(manifest_path,config,inputs): print(manifest_path); return
    import torch
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required for paper functional recovery")
    geometry=read_json(geometry_path); layer=int(geometry["selected_layer"]); rank=int(geometry["selected_rank"]); hidden=read_json(hidden_path); residual=read_json(residual_path); candidates=read_json(root/"candidate_tokens/candidate_tokens.json"); prefixes=read_jsonl(hidden["prefix_snapshot"]); wrong_rows=read_jsonl(wrong_path); wrong_map={row["prefix_id"]:row["wrong_prefix_ids"] for row in wrong_rows}; relaxed_wrong_targets={row["prefix_id"] for row in wrong_rows if int(row.get("relaxed_length_wrong_prefixes",0))>0}
    layer_entry=next(item for item in hidden["layers"] if int(item["layer"])==layer); z=np.asarray(np.load(layer_entry["successor_path"],mmap_mode="r"),dtype=np.float32)
    backbone,tokenizer,source,precision_name,_dtype=_load_backbone_and_tokenizer(config,args.model_path); forward,device,device_ids=prepare_data_parallel(FunctionalRecoveryForward.build(backbone,layer,js_only=True)); forward.eval()
    rows=[]; oracle_distances=[]; checkpoint_root=root/"functional/checkpoints"; checkpoint_root.mkdir(parents=True,exist_ok=True); oracle_tolerance=float(config["functional"]["oracle_tolerance"])
    for residual_entry in residual["entries"]:
        if int(residual_entry["layer"])!=layer: continue
        bundle=np.load(residual_entry["path"]); train_r=bundle["train_residuals"]; nonaux=bundle["nonauxiliary_prefix_indices"]; test_ids={prefixes[int(index)]["prefix_id"] for index in nonaux if prefixes[int(index)]["problem_group"]=="analysis_test"}; required=test_ids|{wrong_id for prefix_id in test_ids for wrong_id in wrong_map.get(prefix_id,[])}; local,conditional=_fit_controls(train_r,prefixes,nonaux,rank,required)
        calibration=np.asarray(candidates["calibration_indices"],dtype=np.int64); calibration_order=np.asarray(candidates["calibration_stability_order"],dtype=np.int64); evaluation_tokens=np.asarray(candidates["folds"][int(residual_entry["fold"])]["evaluation_indices"],dtype=np.int64); auxiliary=np.asarray([i for i,row in enumerate(prefixes) if row["problem_group"]=="auxiliary"],dtype=np.int64); aux_token=z[auxiliary].mean(axis=0); aux_calibration=z[np.ix_(auxiliary,calibration)].mean(axis=(0,1)); records=[]
        for i,prefix in enumerate(prefixes):
            if prefix["problem_group"]!="analysis_test": continue
            local_basis=local[prefix["prefix_id"]]; global_basis,global_exact,global_length_distance,_resolved_stratum=_resolve_conditional_basis(conditional,prefix); wrong_exact=prefix["prefix_id"] not in relaxed_wrong_targets; wrong_bases=[(wrong_id,local.get(wrong_id)) for wrong_id in wrong_map.get(prefix["prefix_id"],[])]; wrong_bases=[(wrong_id,basis) for wrong_id,basis in wrong_bases if basis is not None]; prefix_calibration=z[i,calibration].mean(axis=0)
            available=[local_basis,*([global_basis] if global_basis is not None else []),*[basis for _wrong_id,basis in wrong_bases]]; effective_rank=min(basis.shape[1] for basis in available); local_basis=local_basis[:,:effective_rank]; global_basis=global_basis[:,:effective_rank] if global_basis is not None else None; wrong_bases=[(wrong_id,basis[:,:effective_rank]) for wrong_id,basis in wrong_bases]
            for token_index in evaluation_tokens:
                oracle=z[i,token_index]; baseline=prefix_calibration+aux_token[token_index]-aux_calibration; interaction=oracle-baseline; replacements={"Oracle":oracle,"Rank-0":baseline,"Local":baseline+local_basis@(local_basis.T@interaction)}
                if global_basis is not None: replacements["ConditionalGlobal"]=baseline+global_basis@(global_basis.T@interaction)
                for wrong_id,wrong_basis in wrong_bases:
                    if wrong_basis is not None: replacements[f"Wrong::{wrong_id}"]=baseline+wrong_basis@(wrong_basis.T@interaction)
                for condition,replacement in replacements.items(): records.append({"prefix_index":i,"problem_id":prefix["problem_id"],"prefix_id":prefix["prefix_id"],"candidate_index":int(token_index),"candidate_token_id":int(candidates["candidate_token_ids"][int(token_index)]),"condition":condition,"effective_rank":int(effective_rank),"conditional_global_exact_length_bin":bool(global_exact),"conditional_global_length_bin_distance":global_length_distance,"wrong_control_exact_length_bin":bool(wrong_exact),"replacement":replacement})
        for i,prefix in enumerate(prefixes):
            if prefix["problem_group"]!="analysis_dev": continue
            reference_prefix=z[i,calibration].mean(axis=0)
            for token_index in evaluation_tokens: records.append({"prefix_index":i,"problem_id":prefix["problem_id"],"prefix_id":prefix["prefix_id"],"candidate_index":int(token_index),"candidate_token_id":int(candidates["candidate_token_ids"][int(token_index)]),"condition":"Rank-0-reference","replacement":reference_prefix+aux_token[token_index]-aux_calibration})
            for size in config["functional"]["calibration_stability_sizes"]:
                # Full-M is an algebraic anchor for Rank-0-reference.  Use the
                # identical index order and reduction, not merely the same set
                # in a different order, so floating-point summation cannot
                # create a spurious anchor failure.
                subset=calibration if int(size)==len(calibration) else calibration_order[:int(size)]; prefix_mean=z[i,subset].mean(axis=0); auxiliary_grand=aux_calibration if int(size)==len(calibration) else z[np.ix_(auxiliary,subset)].mean(axis=(0,1))
                for token_index in evaluation_tokens: records.append({"prefix_index":i,"problem_id":prefix["problem_id"],"prefix_id":prefix["prefix_id"],"candidate_index":int(token_index),"candidate_token_id":int(candidates["candidate_token_ids"][int(token_index)]),"condition":f"Rank-0-M{int(size)}","replacement":prefix_mean+aux_token[token_index]-auxiliary_grand})
        records.sort(key=lambda row:(int(row["prefix_index"]),int(row["candidate_index"]),str(row["condition"])))
        batch_size=int(config["functional"]["per_device_batch_size"])*max(1,len(device_ids))
        fold=int(residual_entry["fold"]); checkpoint_rows_path=checkpoint_root/f"fold_{fold}_rows.jsonl"; checkpoint_meta_path=checkpoint_root/f"fold_{fold}.json"
        checkpoint_key=stable_hash({"version":FUNCTIONAL_CHECKPOINT_VERSION,"inputs":inputs,"config_hash":stable_hash(config),"model_source":source,"precision":precision_name,"layer":layer,"rank":rank,"fold":fold,"record_count":len(records),"batch_size":batch_size})
        checkpoint_meta=read_json(checkpoint_meta_path) if checkpoint_meta_path.is_file() else {}
        if args.force or checkpoint_meta.get("checkpoint_key")!=checkpoint_key or not checkpoint_rows_path.is_file():
            checkpoint_rows_path.write_text("",encoding="utf-8"); atomic_json(checkpoint_meta_path,{"checkpoint_key":checkpoint_key,"complete":False,"completed_records":0}); fold_rows=[]
        else:
            fold_rows=read_jsonl(checkpoint_rows_path)
        completed_records=len(fold_rows)
        if completed_records>len(records): raise RuntimeError(f"functional checkpoint has {completed_records} rows for only {len(records)} records")
        for checkpoint_index in ({0,completed_records-1} if completed_records else set()):
            expected=records[checkpoint_index]; observed=fold_rows[checkpoint_index]
            if (observed.get("prefix_id"),int(observed.get("candidate_index",-1)),observed.get("condition"))!=(expected["prefix_id"],int(expected["candidate_index"]),expected["condition"]): raise RuntimeError("functional checkpoint record order does not match regenerated records")
        existing_oracles=[float(row["js"]) for row in fold_rows if row["condition"]=="Oracle"]
        if existing_oracles and max(existing_oracles)>oracle_tolerance: raise RuntimeError("saved functional checkpoint contains a failed Oracle batch; rerun with --force after updating the implementation")
        functional_started=time.monotonic(); total_batches=(len(records)+batch_size-1)//batch_size; report_every=max(1,total_batches//20); completed_batches=completed_records//batch_size; print(f"[functional] fold={fold} records={len(records)} batches={total_batches} global_batch={batch_size} resume_records={completed_records}",flush=True)
        with checkpoint_rows_path.open("a",encoding="utf-8") as checkpoint_handle:
            for start in range(completed_records,len(records),batch_size):
                batch=records[start:start+batch_size]; sequences=[prefixes[row["prefix_index"]]["prefix_token_ids"]+[row["candidate_token_id"]] for row in batch]; ids,mask,positions=pad_token_rows(sequences,tokenizer.pad_token_id); replacement=torch.from_numpy(np.stack([row["replacement"] for row in batch])).float(); oracle_mask=torch.tensor([row["condition"]=="Oracle" for row in batch]); sample=torch.arange(start,start+len(batch)); cell=torch.tensor([int(row["prefix_index"])*len(candidates["candidate_token_ids"])+int(row["candidate_index"]) for row in batch],dtype=torch.long); js,kl,top1,overlap,logit_difference,observed=forward(ids.to(device),mask.to(device),positions.to(device),replacement.to(device),oracle_mask.to(device),sample.to(device),cell.to(device))
                if not torch.equal(observed.cpu(),sample): raise RuntimeError("DataParallel changed functional batch order")
                batch_rows=[]
                for axis,row in enumerate(batch):
                    clean={key:value for key,value in row.items() if key!="replacement"}; clean.update({"layer":layer,"rank":rank,"fold":fold,"js":float(js[axis].cpu())}); batch_rows.append(clean)
                batch_oracles=[float(row["js"]) for row in batch_rows if row["condition"]=="Oracle"]
                batch_oracle_max=max(batch_oracles,default=0.0)
                if batch_oracle_max>oracle_tolerance: raise RuntimeError(f"Oracle reinjection failed immediately at fold={fold} batch={start//batch_size+1}: maximum JS={batch_oracle_max}")
                if start==completed_records: print(f"[functional] ORACLE PREFLIGHT PASS fold={fold} batch={start//batch_size+1} max_js={batch_oracle_max:.3g}",flush=True)
                for clean in batch_rows: checkpoint_handle.write(json.dumps(clean,sort_keys=True,ensure_ascii=False)+"\n")
                checkpoint_handle.flush(); fold_rows.extend(batch_rows)
                if args.preflight_only:
                    atomic_json(checkpoint_meta_path,{"checkpoint_key":checkpoint_key,"complete":False,"completed_records":len(fold_rows),"rows":str(checkpoint_rows_path),"oracle_preflight_pass":True})
                    print(f"[functional] preflight-only checkpoint saved at {checkpoint_rows_path}; rerun without --preflight-only to resume",flush=True); return
                batch_number=start//batch_size+1
                if batch_number%report_every==0 or batch_number==total_batches:
                    elapsed_batches=batch_number-completed_batches; rate=elapsed_batches/max(time.monotonic()-functional_started,1e-9); print(f"[functional] fold={fold} {batch_number}/{total_batches} batches rate={rate:.3f}/s eta={(total_batches-batch_number)/max(rate,1e-9)/60:.1f}m oracle_max={max([float(row['js']) for row in fold_rows if row['condition']=='Oracle'],default=0.0):.3g}",flush=True)
        atomic_json(checkpoint_meta_path,{"checkpoint_key":checkpoint_key,"complete":True,"completed_records":len(fold_rows),"rows":str(checkpoint_rows_path)})
        rows.extend(fold_rows); oracle_distances.extend(float(row["js"]) for row in fold_rows if row["condition"]=="Oracle")
    oracle_max=max(oracle_distances,default=float("inf")); oracle_pass=oracle_max<=oracle_tolerance
    if not oracle_pass: raise RuntimeError(f"Oracle reinjection failed: maximum JS={oracle_max}")
    cells=defaultdict(dict)
    for row in rows: cells[(row["prefix_id"],row["fold"],row["candidate_index"])][row["condition"]]=row
    recovery=[]
    for key,conditions in cells.items():
        if "Oracle" not in conditions or "Rank-0" not in conditions: continue
        d0=conditions["Rank-0"]["js"]; oracle=conditions["Oracle"]["js"]
        for condition,row in conditions.items():
            if condition in {"Oracle","Rank-0"} or condition.startswith("Rank-0-"): continue
            recovery.append({"problem_id":row["problem_id"],"prefix_id":key[0],"fold":key[1],"candidate_index":key[2],"condition":condition,"conditional_global_exact_length_bin":bool(row.get("conditional_global_exact_length_bin",False)),"conditional_global_length_bin_distance":row.get("conditional_global_length_bin_distance"),"wrong_control_exact_length_bin":bool(row.get("wrong_control_exact_length_bin",False)),"d_oracle":oracle,"d_rank0":d0,"distance":row["js"],"gain":d0-row["js"]})
    cell_recovery=defaultdict(dict)
    for row in recovery: cell_recovery[(row["prefix_id"],row["fold"],row["candidate_index"])][row["condition"]]=row
    aggregate=[]
    for key,conditions in cell_recovery.items():
        if "Local" not in conditions: continue
        local_row=conditions["Local"]; wrong=[row for name,row in conditions.items() if name.startswith("Wrong::")]; global_row=conditions.get("ConditionalGlobal")
        aggregate.append({"problem_id":local_row["problem_id"],"prefix_id":key[0],"fold":key[1],"candidate_index":key[2],"wrong_prefix_count":len(wrong),"conditional_global_exact_length_bin":bool(local_row["conditional_global_exact_length_bin"]),"conditional_global_length_bin_distance":local_row["conditional_global_length_bin_distance"],"wrong_control_exact_length_bin":bool(local_row["wrong_control_exact_length_bin"]),"G_local":local_row["gain"],"G_conditional_global":global_row["gain"] if global_row else float("nan"),"G_wrong_mean":float(np.mean([row["gain"] for row in wrong])) if wrong else float("nan"),"D_oracle":local_row["d_oracle"],"D_rank0":local_row["d_rank0"],"D_local":local_row["distance"],"D_conditional_global":global_row["distance"] if global_row else float("nan"),"D_wrong_mean":float(np.mean([row["distance"] for row in wrong])) if wrong else float("nan")})
    expected_cells=int(config["data"]["evaluation_prefixes"])*sum(len(fold["evaluation_indices"]) for fold in candidates["folds"]); expected_wrong=int(config["controls"]["wrong_prefixes_per_target"]); global_cells=sum(np.isfinite(row["G_conditional_global"]) for row in aggregate); complete_wrong_cells=sum(int(row["wrong_prefix_count"])==expected_wrong for row in aggregate); coverage_complete=len(aggregate)==expected_cells and global_cells==expected_cells and complete_wrong_cells==expected_cells
    exact_global=[row for row in aggregate if bool(row["conditional_global_exact_length_bin"])]; exact_wrong=[row for row in aggregate if bool(row["wrong_control_exact_length_bin"])]
    summary={"selected_layer":layer,"selected_rank":rank,"oracle_max_js":oracle_max,"oracle_pass":oracle_pass,"control_coverage":{"expected_cells":expected_cells,"observed_local_cells":len(aggregate),"conditioned_global_cells_with_fallback":global_cells,"conditional_global_exact_cells":len(exact_global),"conditional_global_fallback_cells":len(aggregate)-len(exact_global),"conditional_global_exact_fraction":len(exact_global)/max(1,expected_cells),"complete_wrong_prefix_cells":complete_wrong_cells,"wrong_exact_cells":len(exact_wrong),"wrong_exact_fraction":len(exact_wrong)/max(1,expected_cells),"wrong_prefixes_per_cell":expected_wrong,"conditional_global_fallback":"nearest length bin within the same reasoning-progress bin","complete":coverage_complete}}
    for metric in ("G_local","G_conditional_global","G_wrong_mean"):
        summary[metric]=problem_bootstrap(np.asarray([row[metric] for row in aggregate]),np.asarray([row["problem_id"] for row in aggregate]),replicates=int(config["statistics"]["bootstrap_replicates"]),seed=int(config["seed"]),ci=float(config["statistics"]["ci"]))
    for control in ("conditional_global","wrong_mean"):
        metric=f"G_local_minus_{control}"; values=np.asarray([row["G_local"]-row[f"G_{control}"] for row in aggregate]); summary[metric]=problem_bootstrap(values,np.asarray([row["problem_id"] for row in aggregate]),replicates=int(config["statistics"]["bootstrap_replicates"]),seed=int(config["seed"])+17,ci=float(config["statistics"]["ci"]))
    for control,selected in (("conditional_global",exact_global),("wrong_mean",exact_wrong)):
        metric=f"G_local_minus_{control}_exact_bin"; values=np.asarray([row["G_local"]-row[f"G_{control}"] for row in selected]); summary[metric]=problem_bootstrap(values,np.asarray([row["problem_id"] for row in selected]),replicates=int(config["statistics"]["bootstrap_replicates"]),seed=int(config["seed"])+43,ci=float(config["statistics"]["ci"]))
    summary.update(summarize_absolute_recovery(aggregate,config))
    full_size=len(candidates["calibration_indices"]); anchor=rank0_anchor_max_difference(rows,full_size); stability={str(size):problem_aggregated_mean([row for row in rows if row["condition"]==f"Rank-0-M{int(size)}"]) for size in config["functional"]["calibration_stability_sizes"]}; finite=[value for value in stability.values() if np.isfinite(value)]; relative=(max(finite)-min(finite))/max(max(finite),1e-12); stability_pass=relative<=float(config["functional"]["rank0_stability_relative_tolerance"]) and anchor<=float(config["functional"]["rank0_anchor_tolerance"])
    exact_global_valid=len(exact_global)/max(1,expected_cells)>=.99; exact_wrong_valid=len(exact_wrong)/max(1,expected_cells)>=.99
    summary.update({"rank0_stability":stability,"rank0_stability_relative_range":relative,"rank0_full_M_anchor_max_abs_difference":anchor,"rank0_stability_pass":stability_pass,"gate_functional_local_positive":coverage_complete and summary["G_local"]["mean"]>0,"gate_functional_local_above_conditional_global":coverage_complete and exact_global_valid and summary["G_local_minus_conditional_global_exact_bin"]["mean"]>0,"gate_functional_local_above_wrong":coverage_complete and exact_wrong_valid and summary["G_local_minus_wrong_mean_exact_bin"]["mean"]>0})
    rows_path=root/"functional/paper_distribution_rows.csv"; recovery_path=root/"functional/paper_recovery_rows.csv"; aggregate_path=root/"functional/paper_cell_summary.csv"; summary_path=root/"functional/paper_summary.json"; _write_csv(rows_path,rows); _write_csv(recovery_path,recovery); _write_csv(aggregate_path,aggregate); atomic_json(summary_path,summary); atomic_json(manifest_path,{"complete":True,"config_hash":stable_hash(config),**inputs,"summary":str(summary_path),"rows":str(rows_path),"model_source":source,"precision":precision_name}); print(manifest_path)


if __name__=="__main__":
    main()
