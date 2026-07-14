from __future__ import annotations

import argparse, copy, json, sys, time
from pathlib import Path
from . import analyze_paper_geometry, build_candidate_tokens, compute_contrast_residuals, extract_successor_states, prepare_replication_prefix_pool, select_wrong_prefixes
from .src.utils import atomic_json, ensure_layout, load_config, read_json, result_root


def _call(module,config_path,model_path=None,source_model_path=None,force=False):
    argv=[module.__name__,"--config",str(config_path)]
    if model_path and module in {prepare_replication_prefix_pool,build_candidate_tokens,extract_successor_states}:argv += ["--model-path",model_path]
    if source_model_path and module is prepare_replication_prefix_pool:argv += ["--source-model-path",source_model_path]
    if force and module in {prepare_replication_prefix_pool,select_wrong_prefixes,analyze_paper_geometry}:argv.append("--force")
    if module is analyze_paper_geometry:argv.append("--skip-content-appendix")
    old=sys.argv;sys.argv=argv;started=time.monotonic();print(f"[fixed_replication] START {module.__name__.split('.')[-1]}",flush=True)
    try:module.main()
    finally:sys.argv=old
    print(f"[fixed_replication] DONE {module.__name__.split('.')[-1]} elapsed={(time.monotonic()-started)/60:.1f}m",flush=True)


OPTIONAL_SPECS = {
    "qwen3_8b_base": {
        "name": "qwen3_8b_base",
        "extraction_per_device_batch_size": 8,
        "model": {
            "checkpoint": "Qwen/Qwen3-8B-Base",
            "revision": "main",
            "model_type": "base",
            "local_files_only": False,
            "trust_remote_code": False,
            "attention_implementation": "sdpa",
            "precision": "auto",
        },
    },
    "qwen3_4b_base": {
        "name": "qwen3_4b_base",
        "model": {
            "checkpoint": "Qwen/Qwen3-4B-Base",
            "revision": "main",
            "model_type": "base",
            "local_files_only": False,
            "trust_remote_code": False,
            "attention_implementation": "sdpa",
            "precision": "auto",
        },
    },
    "qwen3_17b_base": {
        "name": "qwen3_17b_base",
        "model": {
            "checkpoint": "Qwen/Qwen3-1.7B-Base",
            "revision": "main",
            "model_type": "base",
            "local_files_only": False,
            "trust_remote_code": False,
            "attention_implementation": "sdpa",
            "precision": "auto",
        },
    },
    "qwen25_7b": {
        "name": "qwen25_7b",
        "model": {
            "checkpoint": "Qwen/Qwen2.5-7B",
            "revision": "main",
            "model_type": "base",
            "local_files_only": False,
            "trust_remote_code": False,
            "attention_implementation": "sdpa",
            "precision": "auto",
        },
    },
}


def _specs(config, requested=()):
    configured=config.get("replication_models")
    if configured:
        specs=copy.deepcopy(configured)
    else:
        qwen=copy.deepcopy(config.get("replication_model",{"checkpoint":"Qwen/Qwen2.5-1.5B","revision":"main"}))
        llama={"checkpoint":"meta-llama/Llama-3.2-3B","revision":"main","model_type":"base","local_files_only":False,"trust_remote_code":False,"attention_implementation":"sdpa","precision":"auto"}
        specs=[{"name":"qwen25_15b","model":qwen},{"name":"llama32_3b","model":llama}]
    known={spec["name"] for spec in specs}
    for name in requested:
        if name in OPTIONAL_SPECS and name not in known:
            specs.append(copy.deepcopy(OPTIONAL_SPECS[name]))
    return specs


def _replication_config(main,main_root,spec):
    rep=copy.deepcopy(main);model=copy.deepcopy(spec.get("model",spec));model["target_layers"]=[0];model["additional_target_layers"]=[];model.pop("normalized_depths",None);model.pop("layer_fractions",None)
    rep.update({"profile":f"fixed_replication_{spec['name']}","model":model,"source_model_config":copy.deepcopy(main["model"]),"replication_mode":True,"replication_independent_tokenizer":True,"source_results_root":str(main_root),"replication_confirmatory_fixed_layer":0,"replication_confirmatory_fixed_rank":64,"results_root":str(main_root/"fixed_replications"/spec["name"])})
    rep["analysis"]=copy.deepcopy(rep["analysis"]);rep["analysis"].update({"ranks":[64],"selection_rank":64,"primary_rank":64})
    if "extraction_per_device_batch_size" in spec:
        rep["extraction"]=copy.deepcopy(rep["extraction"]);rep["extraction"]["per_device_batch_size"]=int(spec["extraction_per_device_batch_size"])
    rep["replication"]=copy.deepcopy(rep.get("replication",{}));rep["replication"]["run_permutations"]=False
    return rep


def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--model-path",action="append",default=[],help="NAME=PATH; repeat for each target model");p.add_argument("--source-model-path");p.add_argument("--force",action="store_true");p.add_argument("--only",action="append",default=[]);p.add_argument("--skip-content-appendix",action="store_true");a=p.parse_args();main=load_config(a.config);main_root=result_root(main)
    if not bool(main.get("replication",{}).get("enabled",False)):print("Replication disabled in configuration");return
    paths={};
    for item in a.model_path:
        if "=" in item:name,path=item.split("=",1)
        else:name,path="qwen25_15b",item
        paths[name]=path
    output=main_root/"fixed_replications/summary.json"
    prior=read_json(output) if output.is_file() else {}
    summaries=dict(prior.get("models",{}))
    specs=_specs(main,a.only)
    known={spec["name"] for spec in specs}
    unknown=[name for name in a.only if name not in known]
    if unknown:
        p.error(f"unknown replication model(s): {', '.join(unknown)}")
    for spec in specs:
        name=spec["name"]
        if a.only and name not in a.only:continue
        rep=_replication_config(main,main_root,spec);root=ensure_layout(rep);config_path=root/"replication_config.json";config_path.write_text(json.dumps(rep,indent=2,sort_keys=True)+"\n",encoding="utf-8");target_path=paths.get(name)
        _call(prepare_replication_prefix_pool,config_path,target_path,a.source_model_path,a.force)
        _call(build_candidate_tokens,config_path,target_path)
        _call(select_wrong_prefixes,config_path,force=a.force)
        _call(extract_successor_states,config_path,target_path)
        _call(compute_contrast_residuals,config_path)
        _call(analyze_paper_geometry,config_path,force=a.force)
        geometry=read_json(root/"metrics/paper_geometry_summary.json");model_metadata=read_json(root/"manifests/hidden_states.json")["model"];post_mlp=next((row for row in geometry.get("interaction_energy_by_layer",[]) if int(row["layer"])==0),None);summaries[name]={"checkpoint":rep["model"]["checkpoint"],"resolved_revision":model_metadata.get("resolved_revision","UNKNOWN"),"hidden_size":model_metadata.get("hidden_size"),"num_decoder_layers":model_metadata.get("num_decoder_layers"),"precision":model_metadata.get("precision"),"device_ids":model_metadata.get("device_ids"),"parallelism":model_metadata.get("parallelism"),"fixed_layer":0,"fixed_rank":64,"delta_global":geometry["delta_conditional_global"],"delta_wrong":geometry.get("delta_wrong_exact_bin",geometry["delta_wrong"]),"R_between_minus_within":geometry["R_between_minus_within"],"post_mlp_interaction_fraction":post_mlp["interaction_fraction_eta"] if post_mlp else None,"pre_attention_interaction_fraction":"not run (optional)","post_attention_interaction_fraction":"not run (optional)","passes":{"delta_global_positive":float(geometry["delta_conditional_global"]["ci_low"])>0,"delta_wrong_positive":float(geometry.get("delta_wrong_exact_bin",geometry["delta_wrong"])["ci_low"])>0,"rotation_above_noise":float(geometry["R_between_minus_within"]["ci_low"])>0},"functional_run":False,"independent_tokenizer_candidates":True}
    atomic_json(output,{"models":summaries,"confirmatory_conditions":{"layer":0,"rank":64,"no_layer_or_rank_reselection":True},"functional_scope":"main model only"});print(output)

if __name__=="__main__":main()
