from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from .src.mechanism import first_layer_value_output_span, load_first_layer_mechanism_model
from .src.utils import atomic_json, ensure_layout, file_sha256, load_config, read_json, read_jsonl, stable_hash, stage_is_complete


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--model-path"); parser.add_argument("--force",action="store_true"); args=parser.parse_args()
    config=load_config(args.config); root=ensure_layout(config); hidden_path=root/"manifests/hidden_states.json"
    inputs={"hidden_states_sha256":file_sha256(hidden_path)}; manifest=root/"manifests/value_output_spans.json"
    if not args.force and stage_is_complete(manifest,config,inputs): print(manifest); return
    import torch
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required for value/output span extraction")
    hidden=read_json(hidden_path); prefixes=read_jsonl(hidden["prefix_snapshot"]); loaded=load_first_layer_mechanism_model(config,args.model_path)
    excluded=set(map(int,config.get("first_layer_mechanism",{}).get("value_space_excluded_positions",config.get("content",{}).get("attention_sink_positions",[0]))))
    if config.get("first_layer_mechanism",{}).get("value_space_exclude_bos",True): excluded.add(0)
    output_root=root/"hidden_states/value_output_spans"; output_root.mkdir(parents=True,exist_ok=True)
    checkpoint=output_root/"checkpoint.json"; key=stable_hash({"version":1,"config":stable_hash(config),"inputs":inputs,"excluded":sorted(excluded)})
    state=read_json(checkpoint) if checkpoint.is_file() and not args.force else {}; rows=list(state.get("rows",[])) if state.get("key")==key else []
    targets=[i for i,p in enumerate(prefixes) if p["problem_group"] in {"analysis_dev","analysis_test"}]
    if [r["prefix_index"] for r in rows] != targets[:len(rows)]: raise RuntimeError("value-span checkpoint prefix order mismatch")
    with torch.no_grad():
        for number in range(len(rows),len(targets)):
            i=targets[number]; basis,diag=first_layer_value_output_span(loaded,list(map(int,prefixes[i]["prefix_token_ids"])),excluded)
            row={"prefix_index":i,"prefix_id":prefixes[i]["prefix_id"],**diag}
            if basis is not None:
                path=output_root/f"prefix_{i:05d}.npy"; np.save(path,basis); row.update({"path":str(path),"sha256":file_sha256(path)})
            rows.append(row); atomic_json(checkpoint,{"key":key,"complete":False,"rows":rows})
            if (number+1)%max(1,len(targets)//20)==0 or number+1==len(targets): print(f"[value_output_span] {number+1}/{len(targets)}",flush=True)
    atomic_json(manifest,{"complete":True,"config_hash":stable_hash(config),**inputs,"excluded_positions":sorted(excluded),"rows":rows,"full_hidden_space_prefixes":sum(r["full_hidden_space"] for r in rows),"target_prefixes":len(rows),"model":loaded.metadata})
    atomic_json(checkpoint,{"key":key,"complete":True,"rows":rows}); print(manifest)


if __name__=="__main__": main()
