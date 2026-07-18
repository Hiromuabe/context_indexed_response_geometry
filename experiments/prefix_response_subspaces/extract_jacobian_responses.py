from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np

from prefix_displacement.extraction import resolve_decoder_layers
from experiments.prefix_successor_subspaces.src.hooks import hidden_tensor_from_output
from experiments.prefix_successor_subspaces.src.model import _decoder_module, _load_backbone_and_tokenizer

from .src.review_experiments import review_roots
from .src.utils import atomic_json, ensure_layout, file_sha256, load_config, read_json, read_jsonl, stable_hash, stage_is_complete


class _JacobianEndpointComplete(RuntimeError):
    def __init__(self, hidden):
        super().__init__("requested differentiable endpoint is complete")
        self.hidden = hidden


def install_differentiable_early_exit(backbone, layer: int) -> None:
    """Stop the decoder immediately after the raw requested block output."""
    import torch

    layers = resolve_decoder_layers(backbone)
    if not 0 <= int(layer) < len(layers):
        raise ValueError(f"Jacobian layer {layer} outside decoder depth {len(layers)}")
    inner = layers[int(layer)]

    class _DifferentiableEarlyExit(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.inner = module

        def forward(self, *args, **kwargs):
            output = self.inner(*args, **kwargs)
            raise _JacobianEndpointComplete(hidden_tensor_from_output(output))

    layers[int(layer)] = _DifferentiableEarlyExit(inner)


def embedding_eigensystem(embeddings: np.ndarray, components: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(embeddings, dtype=np.float64)
    centered = values - values.mean(axis=0, keepdims=True)
    left, singular, right = np.linalg.svd(centered, full_matrices=False)
    tolerance = singular[0] * max(centered.shape) * np.finfo(np.float64).eps if len(singular) else 0.0
    rank = int(np.sum(singular > tolerance))
    keep = rank if components is None else min(rank, int(components))
    if keep <= 0:
        raise ValueError("candidate embedding matrix has zero numerical rank")
    return left[:, :keep], singular[:keep], right[:keep]


def _jvp(endpoint, base, direction):
    import torch

    try:
        return torch.func.jvp(endpoint, (base,), (direction,))[1]
    except (NotImplementedError, RuntimeError) as first_error:
        try:
            return torch.autograd.functional.jvp(endpoint, base, direction, create_graph=False, strict=True)[1]
        except Exception as second_error:
            raise RuntimeError(
                "Jacobian-vector products failed. Use review_experiments.jacobian_attention_implementation='eager' "
                f"or reduce the model precision. torch.func error: {first_error}; fallback error: {second_error}"
            ) from second_error


def batched_jvps(endpoint, base, directions, batch_size: int):
    """Vectorize forward-mode directions in small memory-bounded batches."""
    import torch

    size = max(1, int(batch_size))
    if size == 1 or len(directions) == 1:
        return torch.stack([_jvp(endpoint, base, direction) for direction in directions]), "sequential"
    batches = []
    try:
        for start in range(0, len(directions), size):
            chunk = directions[start : start + size]

            def one(direction):
                return torch.func.jvp(endpoint, (base,), (direction,))[1]

            batches.append(torch.vmap(one)(chunk))
        return torch.cat(batches, dim=0), f"torch.vmap(torch.func.jvp), batch={size}"
    except (NotImplementedError, RuntimeError) as error:
        del batches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[jacobian] batched JVP unavailable; falling back to sequential JVP: {error}", flush=True)
        return torch.stack([_jvp(endpoint, base, direction) for direction in directions]), "sequential_fallback"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    source_root, root = review_roots(config)
    paths = {
        "hidden": source_root / "manifests/hidden_states.json",
        "candidates": source_root / "candidate_tokens/candidate_tokens.json",
        "geometry": source_root / "metrics/paper_geometry_summary.json",
    }
    inputs = {f"{key}_sha256": file_sha256(path) for key, path in paths.items()}
    inputs["implementation_version"] = "review_jacobian_v2_batched_raw_block_early_exit"
    manifest_path = root / "manifests/jacobian_responses.json"
    if not args.force and not args.preflight_only and stage_is_complete(manifest_path, config, inputs):
        print(manifest_path)
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for local-Jacobian extraction")
    review = config.get("review_experiments", {})
    hidden = read_json(paths["hidden"])
    candidates = read_json(paths["candidates"])
    geometry = read_json(paths["geometry"])
    prefixes = read_jsonl(hidden["prefix_snapshot"])
    layer = int(review.get("jacobian_layer", geometry["selected_layer"]))
    candidate_indices = list(map(int, candidates["analysis_indices"]))
    candidate_token_ids = [int(candidates["candidate_token_ids"][index]) for index in candidate_indices]
    model_config = copy.deepcopy(config)
    model_config["model"]["attention_implementation"] = str(review.get("jacobian_attention_implementation", "eager"))
    backbone, tokenizer, source, precision_name, _precision_dtype = _load_backbone_and_tokenizer(model_config, args.model_path)
    device = torch.device("cuda:0")
    backbone.to(device)
    backbone.eval()
    install_differentiable_early_exit(backbone, layer)
    decoder = _decoder_module(backbone)
    embedding_layer = backbone.get_input_embeddings()
    with torch.no_grad():
        candidate_embeddings = embedding_layer(torch.tensor(candidate_token_ids, device=device)).float().cpu().numpy()
    configured_components = review.get("jacobian_embedding_components", 64)
    components = None if configured_components in (None, "all") else int(configured_components)
    left, singular, directions = embedding_eigensystem(candidate_embeddings, components)
    total_energy = float(np.square(candidate_embeddings - candidate_embeddings.mean(axis=0, keepdims=True)).sum())
    retained_fraction = float(np.square(singular).sum() / total_energy) if total_energy > 0 else float("nan")
    target_limit = int(review.get("jacobian_target_contexts", 16))
    auxiliary_limit = int(review.get("jacobian_auxiliary_contexts", 8))
    auxiliary = sorted((row for row in prefixes if row["problem_group"] == "auxiliary"), key=lambda row: str(row["prefix_id"]))[:auxiliary_limit]
    target_groups = set(review.get("jacobian_target_groups", ["analysis_test"]))
    targets = sorted((row for row in prefixes if row["problem_group"] in target_groups), key=lambda row: str(row["prefix_id"]))[:target_limit]
    contexts = [("auxiliary", row) for row in auxiliary] + [("target", row) for row in targets]
    output_root = root / "hidden_states/jacobian_responses"
    output_root.mkdir(parents=True, exist_ok=True)
    embedding_path = output_root / "candidate_embedding_svd.npz"
    np.savez(
        embedding_path, left_vectors=left.astype(np.float32), singular_values=singular.astype(np.float32),
        right_vectors=directions.astype(np.float32), candidate_indices=np.asarray(candidate_indices),
        candidate_token_ids=np.asarray(candidate_token_ids),
    )
    checkpoint_path = output_root / "checkpoint.json"
    checkpoint_key = stable_hash({
        "version": 2, "config_hash": stable_hash(config), "inputs": inputs, "layer": layer,
        "context_ids": [row["prefix_id"] for _role, row in contexts], "candidate_token_ids": candidate_token_ids,
        "components": len(singular), "model_source": source,
    })
    checkpoint = read_json(checkpoint_path) if checkpoint_path.is_file() and not args.force else {}
    entries = list(checkpoint.get("entries", [])) if checkpoint.get("checkpoint_key") == checkpoint_key else []
    completed = {str(row["prefix_id"]) for row in entries if Path(row["path"]).is_file()}
    base_embedding = torch.tensor(candidate_embeddings.mean(axis=0), device=device, dtype=embedding_layer.weight.dtype)
    direction_tensor = torch.tensor(directions, device=device, dtype=embedding_layer.weight.dtype)
    jvp_batch_size = int(review.get("jacobian_jvp_batch_size", 4))
    jvp_modes: set[str] = {str(entry["jvp_mode"]) for entry in entries if entry.get("jvp_mode")}
    started = time.monotonic()
    for context_number, (role, row) in enumerate(contexts, start=1):
        prefix_id = str(row["prefix_id"])
        if prefix_id in completed:
            continue
        prefix_ids = torch.tensor([list(map(int, row["prefix_token_ids"]))], device=device, dtype=torch.long)
        with torch.no_grad():
            prefix_embeddings = embedding_layer(prefix_ids).detach()
        attention_mask = torch.ones((1, prefix_embeddings.shape[1] + 1), device=device, dtype=torch.long)

        def endpoint(candidate_embedding):
            full = torch.cat((prefix_embeddings, candidate_embedding[None, None, :].to(prefix_embeddings.dtype)), dim=1)
            try:
                decoder(
                    inputs_embeds=full, attention_mask=attention_mask, use_cache=False,
                    output_hidden_states=False, return_dict=True,
                )
            except _JacobianEndpointComplete as completed:
                return completed.hidden[0, -1].float()
            raise RuntimeError("decoder did not execute the requested differentiable layer")

        tangents, jvp_mode = batched_jvps(endpoint, base_embedding, direction_tensor, jvp_batch_size)
        jvp_modes.add(jvp_mode)
        weights = torch.tensor(singular, device=tangents.device, dtype=torch.float32)
        weighted = (tangents.detach().float() * weights[:, None]).cpu().numpy()
        output_path = output_root / f"context_{len(entries):04d}.npz"
        np.savez(output_path, weighted_responses=weighted)
        entry = {
            "prefix_id": prefix_id, "problem_id": str(row["problem_id"]), "problem_group": str(row["problem_group"]),
            "role": role, "path": str(output_path), "sha256": file_sha256(output_path), "shape": list(weighted.shape),
            "jvp_mode": jvp_mode,
        }
        entries.append(entry)
        atomic_json(checkpoint_path, {"checkpoint_key": checkpoint_key, "complete": False, "entries": entries})
        rate = (len(entries) - len(completed)) / max(time.monotonic() - started, 1e-9)
        print(f"[jacobian] {context_number}/{len(contexts)} prefix={prefix_id} components={len(singular)} rate={rate:.3f}/s", flush=True)
        if args.preflight_only:
            print(checkpoint_path)
            return
    atomic_json(checkpoint_path, {"checkpoint_key": checkpoint_key, "complete": True, "entries": entries})
    atomic_json(manifest_path, {
        "complete": True, "config_hash": stable_hash(config), **inputs,
        "layer": layer, "candidate_embedding_svd": str(embedding_path),
        "candidate_embedding_svd_sha256": file_sha256(embedding_path),
        "candidate_indices": candidate_indices, "candidate_token_ids": candidate_token_ids,
        "embedding_components": len(singular), "embedding_energy_retained_fraction": retained_fraction,
        "basepoint": "mean embedding of analysis candidates",
        "weighted_response_definition": "singular_value_k * J_i(mean_embedding) v_k; right singular span equals J_i E at full embedding rank",
        "auxiliary_context_count": len(auxiliary), "target_context_count": len(targets),
        "target_groups": sorted(target_groups),
        "jvp_execution": {"configured_batch_size": jvp_batch_size, "observed_modes": sorted(jvp_modes)},
        "decoder_early_exit_after_layer": layer,
        "model": {"model_source": source, "precision": precision_name, "attention_implementation": model_config["model"]["attention_implementation"]},
        "contexts": entries,
    })
    print(manifest_path)


if __name__ == "__main__":
    main()
