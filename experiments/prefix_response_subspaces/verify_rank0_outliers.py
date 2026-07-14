from __future__ import annotations

import argparse, copy, csv, os
from collections import defaultdict
import numpy as np

# This must be set before the process creates its first CUDA context.  Setting
# it inside ``main`` is too late when this module is run as the final stage of
# an already GPU-active pipeline.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from experiments.prefix_successor_subspaces.src.model import _load_backbone_and_tokenizer
from .analyze_functional_recovery import FunctionalRecoveryForward
from .src.data import pad_token_rows
from .src.utils import atomic_json, ensure_layout, file_sha256, load_config, read_json, read_jsonl, stable_hash, stage_is_complete


def _read_csv(path):
    with path.open(newline="",encoding="utf-8") as h:return list(csv.DictReader(h))
def _write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]) if rows else []); 
        if rows:w.writeheader();w.writerows(rows)


def select_outlier_cells(rows,full_size,count=8):
    paired=defaultdict(dict)
    for row in rows:
        if row["condition"] in {"Rank-0-reference",f"Rank-0-M{full_size}"}:paired[(row["prefix_id"],int(row["fold"]),int(row["candidate_index"]))][row["condition"]]=float(row["js"])
    ranked=[]
    for key,values in paired.items():
        if len(values)==2:ranked.append((abs(values["Rank-0-reference"]-values[f"Rank-0-M{full_size}"]),key,values))
    return sorted(ranked,reverse=True)[:count]


def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--model-path");p.add_argument("--force",action="store_true");p.add_argument("--count",type=int,default=8);a=p.parse_args();config=load_config(a.config);root=ensure_layout(config)
    paths={"distribution":root/"functional/paper_distribution_rows.csv","functional_summary":root/"functional/paper_summary.json","hidden":root/"manifests/hidden_states.json","candidates":root/"candidate_tokens/candidate_tokens.json"};inputs={f"{k}_sha256":file_sha256(v) for k,v in paths.items()};manifest=root/"manifests/rank0_outlier_fp32.json"
    if not a.force and stage_is_complete(manifest,config,inputs):print(manifest);return
    import torch
    if not torch.cuda.is_available():raise RuntimeError("CUDA is required for FP32 Rank-0 verification")
    hidden=read_json(paths["hidden"]);prefixes=read_jsonl(hidden["prefix_snapshot"]);candidates=read_json(paths["candidates"]);full=len(candidates["calibration_indices"]);selected=select_outlier_cells(_read_csv(paths["distribution"]),full,a.count);index={r["prefix_id"]:i for i,r in enumerate(prefixes)}
    layer=int(read_json(paths["functional_summary"])["selected_layer"]);z=np.asarray(np.load(next(e for e in hidden["layers"] if int(e["layer"])==layer)["successor_path"],mmap_mode="r"),dtype=np.float32);aux=np.asarray([i for i,r in enumerate(prefixes) if r["problem_group"]=="auxiliary"]);cal=np.asarray(candidates["calibration_indices"]);order=np.asarray(candidates["calibration_stability_order"]);aux_token=z[aux].mean(0)
    records=[]
    for original_difference,(prefix_id,fold,token),old in selected:
        i=index[prefix_id];reference=z[i,cal].astype(np.float64).mean(0)+z[aux,token].astype(np.float64).mean(0)-z[np.ix_(aux,cal)].astype(np.float64).mean((0,1));subset=order[:full];mfull=z[i,subset].astype(np.float64).mean(0)+z[aux,token].astype(np.float64).mean(0)-z[np.ix_(aux,subset)].astype(np.float64).mean((0,1))
        for condition,replacement in (("Rank-0-reference",reference),(f"Rank-0-M{full}",mfull)):records.append({"prefix_index":i,"prefix_id":prefix_id,"problem_id":prefixes[i]["problem_id"],"fold":fold,"candidate_index":token,"candidate_token_id":int(candidates["candidate_token_ids"][token]),"condition":condition,"replacement":replacement.astype(np.float32),"original_bf16_js":old[condition],"original_bf16_pair_abs_difference":original_difference})
    qa=copy.deepcopy(config);qa["model"]=copy.deepcopy(config["model"]);qa["model"].update({"precision":"float32","attention_implementation":"eager"});torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    backbone,tokenizer,source,precision,_=_load_backbone_and_tokenizer(qa,a.model_path);device=torch.device("cuda:0");forward=FunctionalRecoveryForward.build(backbone,layer,js_only=True).to(device);forward.eval();ids,mask,pos=pad_token_rows([prefixes[r["prefix_index"]]["prefix_token_ids"]+[r["candidate_token_id"]] for r in records],tokenizer.pad_token_id);replacement=torch.from_numpy(np.stack([r["replacement"] for r in records])).to(device);sample=torch.arange(len(records));cell=torch.tensor([r["prefix_index"]*len(candidates["candidate_token_ids"])+r["candidate_index"] for r in records])
    with torch.no_grad():js,*_,observed=forward(ids.to(device),mask.to(device),pos.to(device),replacement,torch.zeros(len(records),dtype=torch.bool,device=device),sample.to(device),cell.to(device))
    if not torch.equal(observed.cpu(),sample):raise RuntimeError("FP32 verification changed batch order")
    rows=[]
    for n,r in enumerate(records):rows.append({k:v for k,v in r.items() if k!="replacement"}|{"fp32_deterministic_js":float(js[n].cpu())})
    pairs=defaultdict(dict)
    for r in rows:pairs[(r["prefix_id"],r["fold"],r["candidate_index"])][r["condition"]]=r["fp32_deterministic_js"]
    differences=[abs(v["Rank-0-reference"]-v[f"Rank-0-M{full}"]) for v in pairs.values()];functional=read_json(paths["functional_summary"]);summary={"selected_cells":len(pairs),"selection":"largest original BF16 duplicate-anchor absolute differences","fp32_deterministic_max_pair_abs_difference":max(differences,default=float("nan")),"fp32_deterministic_mean_pair_abs_difference":float(np.mean(differences)) if differences else float("nan"),"all_pairs_within_original_anchor_tolerance":max(differences,default=float("inf"))<=float(config["functional"]["rank0_anchor_tolerance"]),"development_only_cells":True,"main_test_functional_rows_recomputed":0,"main_functional_summary_sha256_unchanged":file_sha256(paths["functional_summary"]),"main_functional_gates":{"local_positive":functional["gate_functional_local_positive"],"local_above_global":functional["gate_functional_local_above_conditional_global"],"local_above_wrong":functional["gate_functional_local_above_wrong"]},"precision":"float32","device":"cuda:0","deterministic_algorithms":True,"tf32":False,"attention_implementation":"eager"}
    out=root/"functional/rank0_outlier_fp32_rows.csv";summary_path=root/"functional/rank0_outlier_fp32_summary.json";_write(out,rows);atomic_json(summary_path,summary);atomic_json(manifest,{"complete":True,"config_hash":stable_hash(config),**inputs,"rows":str(out),"summary":str(summary_path),"model_source":source,"precision":precision});print(manifest)

if __name__=="__main__":main()
