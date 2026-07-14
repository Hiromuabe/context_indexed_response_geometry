from __future__ import annotations

import argparse
from pathlib import Path
from prefix_displacement.model_loading import resolve_model_source
from .src.utils import atomic_json, atomic_jsonl, ensure_layout, file_sha256, load_config, read_jsonl, stable_hash, stage_is_complete


def _tokenizer(model_config,model_path=None):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("transformers is required in the existing GPU environment") from exc
    source,kwargs=resolve_model_source(model_config,model_path);tok=AutoTokenizer.from_pretrained(source,**kwargs)
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    return tok,source


def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--source-model-path");p.add_argument("--model-path");p.add_argument("--force",action="store_true");a=p.parse_args();config=load_config(a.config);root=ensure_layout(config);source_root=Path(config["source_results_root"]);source_path=source_root/"prefix_pool/prefixes.jsonl";inputs={"source_prefixes_sha256":file_sha256(source_path)};manifest=root/"manifests/prefix_pool.json"
    if not a.force and stage_is_complete(manifest,config,inputs):print(manifest);return
    source_tok,source_name=_tokenizer(config["source_model_config"],a.source_model_path);target_tok,target_name=_tokenizer(config["model"],a.model_path);rows=[]
    for row in read_jsonl(source_path):
        prefix_text=source_tok.decode(row["prefix_token_ids"],skip_special_tokens=False,clean_up_tokenization_spaces=False);suffix_text=source_tok.decode(row.get("evaluation_suffix_token_ids",[]),skip_special_tokens=False,clean_up_tokenization_spaces=False);prefix=target_tok.encode(prefix_text,add_special_tokens=False)
        if not prefix:raise RuntimeError(f"retokenization produced an empty prefix: {row['prefix_id']}")
        updated=dict(row);updated.update({"prefix_token_ids":list(map(int,prefix)),"evaluation_suffix_token_ids":list(map(int,target_tok.encode(suffix_text,add_special_tokens=False))),"prefix_length":len(prefix),"last_token_id":int(prefix[-1]),"replication_prefix_text_sha256":stable_hash(prefix_text)});rows.append(updated)
    lengths=sorted(r["prefix_length"] for r in rows);bins=int(config["data"]["length_bins"])
    for row in rows:row["prefix_length_bin"]=min(bins-1,sum(v<=row["prefix_length"] for v in lengths)*bins//(len(lengths)+1))
    out=root/"prefix_pool/prefixes.jsonl";atomic_jsonl(out,rows);atomic_json(manifest,{"complete":True,"config_hash":stable_hash(config),**inputs,"prefixes":str(out),"prefixes_sha256":file_sha256(out),"source_tokenizer":source_name,"target_tokenizer":target_name,"retokenization":"decode source prefix without cleanup; encode target without added special tokens","length_bins_recomputed":True,"counts":{g:sum(r["problem_group"]==g for r in rows) for g in sorted({r["problem_group"] for r in rows})}});print(manifest)

if __name__=="__main__":main()
