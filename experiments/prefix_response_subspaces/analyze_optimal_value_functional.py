from __future__ import annotations

import argparse, csv, json, time
import numpy as np
from prefix_displacement.runtime import prepare_data_parallel
from experiments.prefix_successor_subspaces.src.model import _load_backbone_and_tokenizer
from .analyze_functional_recovery import FunctionalRecoveryForward
from .src.data import pad_token_rows
from .src.statistics import problem_bootstrap
from .src.utils import atomic_json, ensure_layout, file_sha256, load_config, read_json, read_jsonl, stable_hash, stage_is_complete


def _csv(path):
    with path.open(newline="",encoding="utf-8") as h:return list(csv.DictReader(h))
def _write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]) if rows else []); 
        if rows:w.writeheader();w.writerows(rows)


def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--model-path");p.add_argument("--force",action="store_true");a=p.parse_args();config=load_config(a.config);root=ensure_layout(config)
    paths={"optimal":root/"manifests/optimal_value_control.json","optimal_rows":root/"metrics/optimal_value_control_rows.csv","functional":root/"functional/paper_cell_summary.csv","summary":root/"functional/paper_summary.json","hidden":root/"manifests/hidden_states.json","candidates":root/"candidate_tokens/candidate_tokens.json"};inputs={f"{k}_sha256":file_sha256(v) for k,v in paths.items()};manifest=root/"manifests/optimal_value_functional.json"
    if not a.force and stage_is_complete(manifest,config,inputs):print(manifest);return
    paper=read_json(paths["summary"])
    if int(paper["selected_layer"])!=0 or not paper["oracle_pass"]:raise RuntimeError("optimal value Functional requires a passed layer-0 main Functional run")
    hidden=read_json(paths["hidden"]);prefixes=read_jsonl(hidden["prefix_snapshot"]);candidates=read_json(paths["candidates"]);cells={(r["prefix_id"],int(r["fold"]),int(r["candidate_index"])):r for r in _csv(paths["functional"])}
    geometry=[r for r in _csv(paths["optimal_rows"]) if r["site"]=="post_mlp" and r["split"]=="evaluation"]
    basis={(r["prefix_id"],int(r["fold"])):r for r in geometry};z=np.asarray(np.load(next(e for e in hidden["layers"] if int(e["layer"])==0)["successor_path"],mmap_mode="r"),dtype=np.float32)
    aux=np.asarray([i for i,prefix in enumerate(prefixes) if prefix["problem_group"]=="auxiliary"]);cal=np.asarray(candidates["calibration_indices"]);aux_token=z[aux].mean(0);aux_cal=z[np.ix_(aux,cal)].mean((0,1)); pending=[];rows=[]
    for i,prefix in enumerate(prefixes):
        if prefix["problem_group"]!="analysis_test":continue
        prefix_cal=z[i,cal].mean(0)
        for fold in candidates["folds"]:
            f=int(fold["fold_id"]);info=basis[(prefix["prefix_id"],f)];u=np.load(info["basis_path"])
            for t in map(int,fold["evaluation_indices"]):
                source=cells[(prefix["prefix_id"],f,t)];base={"problem_id":prefix["problem_id"],"prefix_id":prefix["prefix_id"],"prefix_index":i,"fold":f,"candidate_index":t,"candidate_token_id":int(candidates["candidate_token_ids"][t]),"D_rank0":float(source["D_rank0"]),"D_local":float(source["D_local"]),"G_local":float(source["G_local"]),"value_span_is_full_hidden":info["value_span_is_full_hidden"]=="True","effective_rank":int(info["effective_rank"])}
                if info["optimal_value_identical_to_local_by_construction"]=="True":
                    rows.append({**base,"D_optimal_value":base["D_local"],"G_optimal_value":base["G_local"],"G_local_minus_optimal_value":0.0,"forward_skipped_identical":True})
                else:
                    oracle=z[i,t];baseline=prefix_cal+aux_token[t]-aux_cal;residual=oracle-baseline;pending.append({**base,"replacement":baseline+u@(u.T@residual)})
    if pending:
        import torch
        if not torch.cuda.is_available():raise RuntimeError("CUDA is required for informative optimal-value cells")
        backbone,tokenizer,source,precision,_=_load_backbone_and_tokenizer(config,a.model_path);forward,device,device_ids=prepare_data_parallel(FunctionalRecoveryForward.build(backbone,0,js_only=True));forward.eval();batch_size=int(config["functional"]["per_device_batch_size"])*max(1,len(device_ids));started=time.monotonic()
        with torch.no_grad():
            for start in range(0,len(pending),batch_size):
                batch=pending[start:start+batch_size];ids,mask,pos=pad_token_rows([prefixes[r["prefix_index"]]["prefix_token_ids"]+[r["candidate_token_id"]] for r in batch],tokenizer.pad_token_id);replacement=torch.from_numpy(np.stack([r["replacement"] for r in batch])).float();sample=torch.arange(start,start+len(batch));cell=torch.tensor([r["prefix_index"]*len(candidates["candidate_token_ids"])+r["candidate_index"] for r in batch]);js,*_,observed=forward(ids.to(device),mask.to(device),pos.to(device),replacement.to(device),torch.zeros(len(batch),dtype=torch.bool,device=device),sample.to(device),cell.to(device))
                if not torch.equal(observed.cpu(),sample):raise RuntimeError("DataParallel changed batch order")
                for n,r in enumerate(batch):
                    clean={k:v for k,v in r.items() if k!="replacement"};d=float(js[n].cpu());g=clean["D_rank0"]-d;rows.append({**clean,"D_optimal_value":d,"G_optimal_value":g,"G_local_minus_optimal_value":clean["G_local"]-g,"forward_skipped_identical":False})
                print(f"[optimal_value_functional] {min(start+len(batch),len(pending))}/{len(pending)} elapsed={(time.monotonic()-started)/60:.1f}m",flush=True)
    expected=int(config["data"]["evaluation_prefixes"])*sum(len(f["evaluation_indices"]) for f in candidates["folds"])
    if len(rows)!=expected:raise RuntimeError(f"optimal-value Functional coverage {len(rows)}/{expected}")
    rows.sort(key=lambda r:(r["prefix_index"],r["fold"],r["candidate_index"]));ids=np.asarray([r["problem_id"] for r in rows]);boot={"replicates":int(config["statistics"]["bootstrap_replicates"]),"seed":int(config["seed"])+2437,"ci":float(config["statistics"]["ci"])};summary={"coverage":{"expected_cells":expected,"observed_cells":len(rows),"complete":len(rows)==expected},"forward_skipped_identical_cells":sum(r["forward_skipped_identical"] for r in rows)}
    for metric in ("G_optimal_value","G_local_minus_optimal_value"):summary[metric]=problem_bootstrap(np.asarray([r[metric] for r in rows]),ids,**boot)
    informative=[r for r in rows if not r["forward_skipped_identical"]];summary["informative_cells"]=len(informative);summary["claim_supported"]=bool(informative) and float(summary["G_local_minus_optimal_value"]["ci_low"])>0
    out=root/"functional/optimal_value_rows.csv";summary_path=root/"functional/optimal_value_summary.json";_write(out,rows);atomic_json(summary_path,summary);atomic_json(manifest,{"complete":True,"config_hash":stable_hash(config),**inputs,"rows":str(out),"summary":str(summary_path)});print(manifest)

if __name__=="__main__":main()
