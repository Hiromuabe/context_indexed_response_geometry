from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from prefix_displacement.extraction import resolve_decoder_layers
from experiments.prefix_successor_subspaces.src.hooks import capture_layer_outputs
from experiments.prefix_successor_subspaces.src.model import _decoder_hidden

from .src.data import pad_token_rows
from .src.model import assert_output_order, load_endpoint_model
from .src.utils import atomic_json, ensure_layout, file_sha256, load_config, read_json, read_jsonl, stable_hash, stage_is_complete


def _backbone(loaded):
    wrapped = loaded.model.module if hasattr(loaded.model, "module") else loaded.model
    return wrapped.backbone


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--model-path")
    args = parser.parse_args(); config = load_config(args.config); root = ensure_layout(config)
    prefix_path = root / "prefix_pool/prefixes.jsonl"; candidate_path = root / "candidate_tokens/candidate_tokens.json"
    control_kind = str(config.get("controls", {}).get("kind", "prediction_matched"))
    match_path = root / "matches/prefix_matches.jsonl"
    wrong_path = root / "controls/wrong_prefixes.jsonl"
    inputs = {"prefixes_sha256": file_sha256(prefix_path), "candidate_tokens_sha256": file_sha256(candidate_path)}
    if control_kind == "prediction_matched":
        inputs["matches_sha256"] = file_sha256(match_path)
    elif control_kind == "wrong_prefix":
        inputs["wrong_prefixes_sha256"] = file_sha256(wrong_path)
    manifest_path = root / "manifests/hidden_states.json"
    if stage_is_complete(manifest_path, config, inputs): print(manifest_path); return
    import torch
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required for hidden-state extraction")
    prefixes_all = read_jsonl(prefix_path)
    active_groups = {"auxiliary", "analysis_train", "analysis_dev", "analysis_test"}
    matched_ids = ({row["matched_prefix_id"] for row in read_jsonl(match_path) if row.get("matched") and row.get("matched_prefix_id")} if control_kind == "prediction_matched" else set())
    wrong_ids = ({prefix_id for row in read_jsonl(wrong_path) for prefix_id in row.get("wrong_prefix_ids", [])} if control_kind == "wrong_prefix" else set())
    control_ids = matched_ids | wrong_ids
    prefixes = [row for row in prefixes_all if row["problem_group"] in active_groups or row["prefix_id"] in control_ids]
    candidate = read_json(candidate_path); token_ids = list(map(int, candidate["candidate_token_ids"]))
    loaded = load_endpoint_model(config, args.model_path)
    if bool(config.get("replication_mode", False)) and not bool(config.get("replication_independent_tokenizer", False)):
        source_root = Path(str(config["source_results_root"])); source_manifest = read_json(source_root / "manifests/candidate_tokens.json")
        if int(len(loaded.tokenizer)) != int(source_manifest["tokenizer_vocabulary_size"]):
            raise RuntimeError("Replication tokenizer vocabulary size differs from the main checkpoint")
        for token_row in candidate.get("candidate_tokens", []):
            observed = loaded.tokenizer.decode([int(token_row["token_id"])], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            if observed != token_row["text"]:
                raise RuntimeError(f"Replication tokenizer mismatch for token ID {token_row['token_id']}")
    dtype = np.float16 if config["extraction"]["storage_dtype"] == "float16" else np.float32
    layer_paths, arrays = {}, {}
    for layer in loaded.layer_ids:
        path = root / f"hidden_states/layer_{layer}_successor.npy"
        arrays[layer] = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=(len(prefixes), len(token_ids), loaded.hidden_size))
        layer_paths[layer] = path
    content_root = root / "hidden_states/content"; content_root.mkdir(parents=True, exist_ok=True)
    batch_per_device = int(config["extraction"]["per_device_batch_size"])
    global_batch = max(1, batch_per_device * max(1, len(loaded.device_ids)))
    started=time.monotonic(); report_every=max(1,len(prefixes)//100); print(f"[successor_states] prefixes={len(prefixes)} candidates={len(token_ids)} layers={loaded.layer_ids} global_batch={global_batch}",flush=True)
    with torch.no_grad():
        for prefix_index, row in enumerate(prefixes):
            prefix = list(map(int, row["prefix_token_ids"]))
            for start in range(0, len(token_ids), global_batch):
                candidates = token_ids[start:start+global_batch]
                sequences = [prefix + [token] for token in candidates]
                ids, mask, positions = pad_token_rows(sequences, loaded.tokenizer.pad_token_id)
                sample = torch.arange(start, start+len(candidates), dtype=torch.long)
                endpoints, observed = loaded.model(ids.to(loaded.device), mask.to(loaded.device), positions.to(loaded.device), sample.to(loaded.device))
                assert_output_order(sample, observed); values = endpoints.float().cpu().numpy()
                for layer_axis, layer in enumerate(loaded.layer_ids): arrays[layer][prefix_index, start:start+len(candidates)] = values[:, layer_axis]
            # Content is one natural-prefix forward on cuda:0. Temporary hooks are
            # safe here because this call deliberately bypasses DataParallel.
            backbone = _backbone(loaded); layers = resolve_decoder_layers(backbone)
            ids = torch.tensor([prefix], dtype=torch.long, device=loaded.device); mask = torch.ones_like(ids)
            with capture_layer_outputs(layers, loaded.layer_ids, positions=None) as captured:
                _decoder_hidden(backbone, input_ids=ids, attention_mask=mask)
            for layer in loaded.layer_ids:
                content = captured[layer][0].detach().float().cpu().numpy().astype(dtype)
                np.save(content_root / f"layer_{layer}_prefix_{prefix_index:05d}.npy", content)
            completed=prefix_index+1
            if completed%report_every==0 or completed==len(prefixes):
                rate=completed/max(time.monotonic()-started,1e-9); print(f"[successor_states] {completed}/{len(prefixes)} prefixes rate={rate:.3f}/s eta={(len(prefixes)-completed)/max(rate,1e-9)/60:.1f}m",flush=True)
    entries = []
    for layer in loaded.layer_ids:
        arrays[layer].flush(); entries.append({"layer": layer, "successor_path": str(layer_paths[layer]), "successor_sha256": file_sha256(layer_paths[layer]), "shape": list(arrays[layer].shape), "dtype": str(arrays[layer].dtype), "content_pattern": str(content_root / f"layer_{layer}_prefix_*.npy")})
    prefix_snapshot = root / "hidden_states/prefixes.jsonl"
    from .src.utils import atomic_jsonl
    atomic_jsonl(prefix_snapshot, prefixes)
    atomic_json(manifest_path, {"complete": True, "config_hash": stable_hash(config), **inputs, "control_kind": control_kind, "prefix_snapshot": str(prefix_snapshot), "prefix_snapshot_sha256": file_sha256(prefix_snapshot), "candidate_set_hash": stable_hash(token_ids), "model": loaded.metadata, "layers": entries})
    print(manifest_path)


if __name__ == "__main__": main()
