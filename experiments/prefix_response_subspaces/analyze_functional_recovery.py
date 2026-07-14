from __future__ import annotations

import argparse
import csv

import numpy as np

from prefix_displacement.extraction import resolve_decoder_layers
from prefix_displacement.runtime import prepare_data_parallel
from experiments.prefix_successor_subspaces.src.hooks import make_replica_local_layer_controller
from experiments.prefix_successor_subspaces.src.model import _decoder_hidden, _endpoint_logits, _load_backbone_and_tokenizer

from .src.data import pad_token_rows
from .src.statistics import problem_bootstrap
from .src.subspaces import top_svd
from .src.utils import atomic_json, ensure_layout, file_sha256, load_config, read_json, read_jsonl, stable_hash, stage_is_complete


class FunctionalRecoveryForward:
    """Normal forward plus one block-output replacement, safe under DataParallel."""
    @staticmethod
    def build(backbone, layer_index: int, *, js_only: bool = False):
        import torch
        layers = resolve_decoder_layers(backbone)
        layers[layer_index] = make_replica_local_layer_controller(layers[layer_index])

        class Forward(torch.nn.Module):
            def __init__(self, model, index, only_js):
                super().__init__(); self.backbone = model; self.layer_index = index; self.js_only = only_js

            def forward(self, input_ids, attention_mask, positions, replacement, oracle_mask, sample_index, cell_index=None):
                controller = resolve_decoder_layers(self.backbone)[self.layer_index]
                with torch.no_grad():
                    if cell_index is None:
                        first = torch.arange(len(input_ids), device=input_ids.device)
                        inverse = first
                    else:
                        # Records are sorted by cell.  Run the clean branch once
                        # per unique (prefix, token) within each replica, then
                        # broadcast clean logits/oracle states to its controls.
                        boundary = torch.ones(len(cell_index), dtype=torch.bool, device=cell_index.device)
                        if len(cell_index) > 1:
                            boundary[1:] = cell_index[1:] != cell_index[:-1]
                        first = torch.nonzero(boundary, as_tuple=False).flatten()
                        inverse = torch.cumsum(boundary.to(torch.long), dim=0) - 1
                    controller.capture_at(positions[first])
                    try:
                        original_hidden = _decoder_hidden(self.backbone, input_ids=input_ids[first], attention_mask=attention_mask[first])
                        exact_oracle_unique = controller.take_captured()
                    finally: controller.clear()
                    original_logits_unique = _endpoint_logits(self.backbone, original_hidden, positions[first])
                    def metrics(original_logits, modified_logits):
                        p_log = torch.log_softmax(original_logits.float(), dim=-1); q_log = torch.log_softmax(modified_logits.float(), dim=-1)
                        p, q = p_log.exp(), q_log.exp(); midpoint = 0.5 * (p + q); log_m = midpoint.clamp_min(1e-30).log()
                        js = 0.5 * ((p * (p_log-log_m)).sum(-1) + (q * (q_log-log_m)).sum(-1))
                        if self.js_only:
                            unused=torch.zeros_like(js)
                            return js,unused,unused,unused,unused
                        kl = (p * (p_log-q_log)).sum(-1); top1 = (original_logits.argmax(-1) == modified_logits.argmax(-1)).float()
                        k = min(5, original_logits.shape[-1]); a = original_logits.topk(k, dim=-1).indices; b = modified_logits.topk(k, dim=-1).indices
                        overlap = (a[:, :, None] == b[:, None, :]).any(-1).float().mean(-1)
                        row = torch.arange(len(a), device=a.device); top = original_logits.argmax(-1)
                        return js,kl,top1,overlap,modified_logits[row,top]-original_logits[row,top]
                    if cell_index is None:
                        active = torch.where(oracle_mask[:, None].bool(), exact_oracle_unique, replacement)
                        controller.replace_at(positions, active)
                        try: modified_hidden = _decoder_hidden(self.backbone, input_ids=input_ids, attention_mask=attention_mask)
                        finally: controller.clear()
                        modified_logits = _endpoint_logits(self.backbone, modified_hidden, positions)
                        js,kl,top1,overlap,top_logit_difference=metrics(original_logits_unique,modified_logits)
                    else:
                        # Keep the clean and every modified forward at exactly
                        # the same unique-cell batch shape.  Running the clean
                        # reference on B unique cells but interventions on B*C
                        # repeated rows changes low-precision CUDA kernels and
                        # can create a spurious non-zero Oracle JS.
                        if int(torch.unique(cell_index).numel()) != int(len(first)):
                            raise RuntimeError("cell_index rows must be contiguous within each DataParallel replica")
                        starts=first[inverse]; slots=torch.arange(len(cell_index),device=cell_index.device)-starts
                        outputs=[torch.empty(len(cell_index),device=input_ids.device,dtype=torch.float32) for _ in range(5)]
                        unique_positions=positions[first]; unique_ids=input_ids[first]; unique_mask=attention_mask[first]
                        for slot in range(int(slots.max().item())+1):
                            active_rows=torch.nonzero(slots==slot,as_tuple=False).flatten(); unique_axes=inverse[active_rows]
                            active_unique=exact_oracle_unique.clone(); chosen=torch.where(oracle_mask[active_rows,None].bool(),exact_oracle_unique[unique_axes],replacement[active_rows].to(dtype=exact_oracle_unique.dtype)); active_unique[unique_axes]=chosen
                            controller.replace_at(unique_positions,active_unique)
                            try: modified_hidden=_decoder_hidden(self.backbone,input_ids=unique_ids,attention_mask=unique_mask)
                            finally: controller.clear()
                            modified_logits_unique=_endpoint_logits(self.backbone,modified_hidden,unique_positions)
                            values=metrics(original_logits_unique[unique_axes],modified_logits_unique[unique_axes])
                            for destination,value in zip(outputs,values): destination[active_rows]=value
                        js,kl,top1,overlap,top_logit_difference=outputs
                return js, kl, top1, overlap, top_logit_difference, sample_index
        return Forward(backbone, layer_index, js_only)


def _basis(matrix, rank):
    effective = min(rank, matrix.shape[0], matrix.shape[1], int(np.linalg.matrix_rank(matrix)))
    return top_svd(matrix, effective)


def _write_csv(path, rows):
    if not rows: path.write_text("", encoding="utf-8"); return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def problem_aggregated_mean(rows, value_key="js"):
    grouped = {}
    for row in rows:
        grouped.setdefault(str(row["problem_id"]), []).append(float(row[value_key]))
    if not grouped:
        return float("nan")
    return float(np.mean([np.mean(values) for values in grouped.values()]))


def rank0_anchor_max_difference(rows, full_calibration_size):
    by_cell = {}
    for row in rows:
        if row["condition"] not in {"Rank-0-reference", f"Rank-0-M{int(full_calibration_size)}"}:
            continue
        key = (row["prefix_id"], int(row["fold"]), int(row["candidate_index"]))
        by_cell.setdefault(key, {})[row["condition"]] = float(row["js"])
    if not by_cell or any(len(conditions) != 2 for conditions in by_cell.values()):
        raise RuntimeError("rank-0 anchor comparison is missing a condition on one or more development cells")
    return float(max(abs(conditions["Rank-0-reference"] - conditions[f"Rank-0-M{int(full_calibration_size)}"]) for conditions in by_cell.values()))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--model-path")
    args = parser.parse_args(); config = load_config(args.config); root = ensure_layout(config)
    geometry_path = root / "metrics/geometry_summary.json"; hidden_path = root / "manifests/hidden_states.json"; residual_path = root / "manifests/residuals.json"
    match_diagnostics_path = root / "matches/diagnostics.json"
    inputs = {"geometry_sha256": file_sha256(geometry_path), "hidden_sha256": file_sha256(hidden_path), "residuals_sha256": file_sha256(residual_path), "match_diagnostics_sha256": file_sha256(match_diagnostics_path)}
    manifest_path = root / "manifests/functional.json"
    if stage_is_complete(manifest_path, config, inputs): print(manifest_path); return
    geometry = read_json(geometry_path)
    # Functional Local-vs-Global recovery remains identifiable even when no
    # valid prediction-matched control exists. Matched claims are gated later.
    required = ("delta_global", "delta_content")
    if not all(float(geometry[name]["mean"]) > 0 for name in required):
        atomic_json(manifest_path, {"complete": True, "skipped": True, "reason": "Experiment 1 gates 4-6 did not all pass", "config_hash": stable_hash(config), **inputs}); print(manifest_path); return
    import torch
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required for functional recovery")
    hidden_manifest, residual_manifest = read_json(hidden_path), read_json(residual_path); candidates = read_json(root / "candidate_tokens/candidate_tokens.json")
    match_diagnostics = read_json(match_diagnostics_path)
    prefixes = read_jsonl(hidden_manifest["prefix_snapshot"]); matches = {row["prefix_id"]: row["matched_prefix_id"] for row in read_jsonl(root / "matches/prefix_matches.jsonl") if row["split"] == "evaluation" and row.get("matched") and row.get("matched_prefix_id")}
    prefix_index = {row["prefix_id"]: i for i, row in enumerate(prefixes)}; layer = int(config["model"]["functional_primary_layer"]); rank = int(config["analysis"]["primary_rank"])
    layer_entry = next(item for item in hidden_manifest["layers"] if int(item["layer"]) == layer); z = np.asarray(np.load(layer_entry["successor_path"], mmap_mode="r"), dtype=np.float32)
    backbone, tokenizer, source, precision_name, _dtype = _load_backbone_and_tokenizer(config, args.model_path)
    forward, device, device_ids = prepare_data_parallel(FunctionalRecoveryForward.build(backbone, layer)); forward.eval()
    rows = []; oracle_distances = []
    residual_entries = [item for item in residual_manifest["entries"] if int(item["layer"]) == layer]
    for residual_entry in residual_entries:
        bundle = np.load(residual_entry["path"]); train_r = bundle["train_residuals"]; nonaux = bundle["nonauxiliary_prefix_indices"]
        local_map = {prefixes[int(full)]["prefix_id"]: _basis(train_r[i], rank) for i, full in enumerate(nonaux)}
        global_mask = np.asarray([prefixes[int(full)]["problem_group"] == "analysis_train" for full in nonaux]); global_basis = _basis(train_r[global_mask].reshape(-1, train_r.shape[-1]), rank)
        calibration = np.asarray(candidates["calibration_indices"], dtype=np.int64); calibration_order = np.asarray(candidates["calibration_stability_order"], dtype=np.int64); evaluation_tokens = np.asarray(candidates["folds"][int(residual_entry["fold"])]["evaluation_indices"], dtype=np.int64)
        auxiliary = np.asarray([i for i, row in enumerate(prefixes) if row["problem_group"] == "auxiliary"], dtype=np.int64)
        aux_token = z[auxiliary].mean(axis=0); aux_calibration = z[np.ix_(auxiliary, calibration)].mean(axis=(0, 1))
        records = []
        for i, prefix in enumerate(prefixes):
            if prefix["problem_group"] != "analysis_test": continue
            local = local_map[prefix["prefix_id"]]; matched_id = matches.get(prefix["prefix_id"]); matched = local_map.get(matched_id)
            prefix_calibration = z[i, calibration].mean(axis=0)
            for token_index in evaluation_tokens:
                oracle = z[i, token_index]; baseline = prefix_calibration + aux_token[token_index] - aux_calibration; residual = oracle - baseline
                replacements = {"Oracle": oracle, "Rank-0": baseline, "Local": baseline + local @ (local.T @ residual), "Global": baseline + global_basis @ (global_basis.T @ residual)}
                if matched is not None:
                    replacements["Matched"] = baseline + matched @ (matched.T @ residual)
                for condition, replacement in replacements.items(): records.append({"prefix_index": i, "problem_id": prefix["problem_id"], "prefix_id": prefix["prefix_id"], "candidate_index": int(token_index), "candidate_token_id": int(candidates["candidate_token_ids"][int(token_index)]), "condition": condition, "replacement": replacement})
        # Development-only rank-0 calibration-size stability. These rows never
        # enter the test functional-gain comparison.
        for i, prefix in enumerate(prefixes):
            if prefix["problem_group"] != "analysis_dev": continue
            reference_prefix_calibration = z[i, calibration].mean(axis=0)
            for token_index in evaluation_tokens:
                reference_baseline = reference_prefix_calibration + aux_token[token_index] - aux_calibration
                records.append({"prefix_index": i, "problem_id": prefix["problem_id"], "prefix_id": prefix["prefix_id"], "candidate_index": int(token_index), "candidate_token_id": int(candidates["candidate_token_ids"][int(token_index)]), "condition": "Rank-0-reference", "replacement": reference_baseline})
            for calibration_size in config["functional"]["calibration_stability_sizes"]:
                subset = calibration_order[:int(calibration_size)]
                prefix_calibration = z[i, subset].mean(axis=0)
                auxiliary_grand = z[np.ix_(auxiliary, subset)].mean(axis=(0, 1))
                for token_index in evaluation_tokens:
                    baseline = prefix_calibration + aux_token[token_index] - auxiliary_grand
                    records.append({"prefix_index": i, "problem_id": prefix["problem_id"], "prefix_id": prefix["prefix_id"], "candidate_index": int(token_index), "candidate_token_id": int(candidates["candidate_token_ids"][int(token_index)]), "condition": f"Rank-0-M{int(calibration_size)}", "replacement": baseline})
        batch_size = int(config["functional"]["per_device_batch_size"]) * max(1, len(device_ids))
        for start in range(0, len(records), batch_size):
            batch = records[start:start+batch_size]; sequences = [prefixes[row["prefix_index"]]["prefix_token_ids"] + [row["candidate_token_id"]] for row in batch]
            ids, mask, positions = pad_token_rows(sequences, tokenizer.pad_token_id); replacement = torch.from_numpy(np.stack([row["replacement"] for row in batch])).float(); oracle_mask = torch.tensor([row["condition"] == "Oracle" for row in batch]); sample = torch.arange(start, start+len(batch))
            outputs = forward(ids.to(device), mask.to(device), positions.to(device), replacement.to(device), oracle_mask.to(device), sample.to(device))
            js, kl, top1, overlap, logit_diff, observed = outputs
            if not torch.equal(observed.cpu(), sample): raise RuntimeError("DataParallel changed batch order")
            for offset, row in enumerate(batch):
                clean = {key: value for key, value in row.items() if key != "replacement"}; clean.update({"layer": layer, "fold": residual_entry["fold"], "js": float(js[offset].cpu()), "kl": float(kl[offset].cpu()), "top1_agreement": float(top1[offset].cpu()), "top5_overlap": float(overlap[offset].cpu()), "original_top1_logit_difference": float(logit_diff[offset].cpu())}); rows.append(clean)
                if row["condition"] == "Oracle": oracle_distances.append(clean["js"])
    oracle_max = max(oracle_distances, default=float("inf")); tolerance = float(config["functional"]["oracle_tolerance"])
    if oracle_max > tolerance: raise RuntimeError(f"Oracle reinjection failed: maximum JS {oracle_max} > {tolerance}")
    by_cell = {}
    for row in rows: by_cell.setdefault((row["prefix_id"], row["fold"], row["candidate_index"]), {})[row["condition"]] = row
    recovery_rows = []
    for cell, conditions in by_cell.items():
        if "Oracle" not in conditions:
            continue
        d0, oracle = conditions["Rank-0"]["js"], conditions["Oracle"]["js"]
        denominator = d0 - oracle
        for condition in ("Local", "Matched", "Global"):
            if condition not in conditions:
                continue
            distance = conditions[condition]["js"]; recovery_rows.append({"prefix_id": cell[0], "fold": cell[1], "candidate_index": cell[2], "problem_id": conditions[condition]["problem_id"], "condition": condition, "d_oracle": oracle, "d_rank0": d0, "distance": distance, "gain": d0-distance, "normalized_recovery": (d0-distance)/denominator if denominator > float(config["functional"]["denominator_epsilon"]) else float("nan")})
    rows_path = root / "functional/distribution_rows.csv"; recovery_path = root / "functional/recovery_rows.csv"; _write_csv(rows_path, rows); _write_csv(recovery_path, recovery_rows)
    summary = {}
    for condition in ("Local", "Matched", "Global"):
        selected = [row for row in recovery_rows if row["condition"] == condition]; summary[f"G_{condition.lower()}"] = problem_bootstrap(np.asarray([row["gain"] for row in selected]), np.asarray([row["problem_id"] for row in selected]), replicates=int(config["statistics"]["bootstrap_replicates"]), seed=int(config["seed"]), ci=float(config["statistics"]["ci"]))
    recovery_by_cell = {}
    for row in recovery_rows:
        recovery_by_cell.setdefault((row["prefix_id"], row["fold"], row["candidate_index"]), {})[row["condition"]] = row
    for control in ("Matched", "Global"):
        paired = [(values["Local"]["gain"] - values[control]["gain"], values["Local"]["problem_id"]) for values in recovery_by_cell.values() if "Local" in values and control in values]
        summary[f"G_local_minus_{control.lower()}"] = problem_bootstrap(np.asarray([item[0] for item in paired]), np.asarray([item[1] for item in paired]), replicates=int(config["statistics"]["bootstrap_replicates"]), seed=int(config["seed"])+31, ci=float(config["statistics"]["ci"]))
    full_calibration_size = int(len(candidates["calibration_indices"]))
    if full_calibration_size not in set(map(int, config["functional"]["calibration_stability_sizes"])):
        raise ValueError("functional.calibration_stability_sizes must include the full calibration-set size")
    stability = {}
    for calibration_size in config["functional"]["calibration_stability_sizes"]:
        condition = f"Rank-0-M{int(calibration_size)}"; selected = [row for row in rows if row["condition"] == condition]
        stability[str(calibration_size)] = problem_aggregated_mean(selected)
    finite_stability = [value for value in stability.values() if np.isfinite(value)]
    relative_range = ((max(finite_stability)-min(finite_stability))/max(max(finite_stability), 1e-12)) if finite_stability else float("inf")
    anchor_difference = rank0_anchor_max_difference(rows, full_calibration_size)
    anchor_tolerance = float(config["functional"]["rank0_anchor_tolerance"])
    stability_pass = relative_range <= float(config["functional"]["rank0_stability_relative_tolerance"]) and anchor_difference <= anchor_tolerance
    valid_denominators = [row for row in recovery_rows if row["condition"] == "Local" and np.isfinite(row["normalized_recovery"])]
    total_denominators = sum(row["condition"] == "Local" for row in recovery_rows)
    test_rank0_rows = [row for row in rows if row["condition"] == "Rank-0"]
    dev_reference_rows = [row for row in rows if row["condition"] == "Rank-0-reference"]
    test_rank0_mean = problem_aggregated_mean(test_rank0_rows)
    dev_reference_mean = problem_aggregated_mean(dev_reference_rows)
    match_quality_pass = bool(match_diagnostics.get("prebranch_matching_gate_pass", False))
    global_claim_eligible = stability_pass and len(valid_denominators) > 0
    matched_claim_eligible = global_claim_eligible and match_quality_pass
    summary.update({"oracle_max_js": oracle_max, "oracle_pass": True, "rank0_mean_js": test_rank0_mean, "test_rank0_mean_js": test_rank0_mean, "dev_rank0_reference_mean_js": dev_reference_mean, "rank0_stability_mean_js_by_calibration_size": stability, "rank0_stability_same_cells": True, "rank0_stability_split": "analysis_dev", "rank0_test_split": "analysis_test", "rank0_stability_full_M_anchor_max_abs_difference": anchor_difference, "rank0_stability_anchor_tolerance": anchor_tolerance, "rank0_stability_relative_range": relative_range, "rank0_stability_pass": stability_pass, "match_quality_pass": match_quality_pass, "normalized_recovery_valid_cells": len(valid_denominators), "normalized_recovery_excluded_cells": total_denominators-len(valid_denominators), "rank0_indistinguishable_from_oracle": len(valid_denominators) == 0, "functional_global_claim_eligible": global_claim_eligible, "functional_matched_claim_eligible": matched_claim_eligible, "functional_claim_eligible": matched_claim_eligible})
    summary_path = root / "functional/summary.json"; atomic_json(summary_path, summary)
    atomic_json(manifest_path, {"complete": True, "skipped": False, "config_hash": stable_hash(config), **inputs, "rows": str(rows_path), "rows_sha256": file_sha256(rows_path), "recovery_rows": str(recovery_path), "summary": str(summary_path), "oracle_pass": True, "model_source": source, "precision": precision_name, "parallelism": "torch.nn.DataParallel" if len(device_ids)>1 else "single_gpu"})
    print(manifest_path)


if __name__ == "__main__": main()
