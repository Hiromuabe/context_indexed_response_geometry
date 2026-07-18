from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .analyze_paper_geometry import _fit_controls, _resolve_conditional_basis
from .src.shared_dictionary import (
    OptimizationConfig,
    SharedCovarianceFit,
    as_residual_tensor,
    context_projection_variances,
    covariance_subspace,
    fit_fixed_dictionary_weights,
    fit_shared_covariance_model,
    pooled_pca_basis,
    select_pooled_subspace,
)
from .src.storage import load_residual_entry
from .src.subspaces import explained_variance
from .src.utils import atomic_json, file_sha256, load_config, read_json, read_jsonl, stable_hash


METHOD_TARGET = "target_context_pca"
METHOD_MATCHED = "matched_common"
METHOD_WRONG = "wrong_context"
METHOD_POOLED = "pooled_pca_context_selection"
METHOD_CPC = "truncated_cpc"
METHOD_DICTIONARY = "shared_nonorthogonal_dictionary"


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _resolve_device(name: str) -> str:
    import torch

    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _stable_seed(base: int, *items: Any) -> int:
    digest = hashlib.sha256(json.dumps(items, sort_keys=True, default=str).encode("utf-8")).digest()
    return (int(base) + int.from_bytes(digest[:4], "little")) % (2**31 - 1)


def _residual_artifacts_exist(manifest: dict[str, Any]) -> bool:
    entries = manifest.get("entries", [])
    if not entries:
        return False
    for entry in entries:
        if entry.get("storage_format") == "npy_bundle":
            paths = (entry.get("train_path"), entry.get("evaluation_path"))
        else:
            paths = (entry.get("path"),)
        if any(not value or not Path(str(value)).is_file() for value in paths):
            return False
    return True


def _ensure_residuals(config_path: str, config: dict[str, Any], source_root: Path, *, build_missing: bool) -> Path:
    residual_path = source_root / "manifests/residuals.json"
    force_rebuild = False
    if residual_path.is_file():
        try:
            residual_manifest = read_json(residual_path)
        except (OSError, ValueError, json.JSONDecodeError):
            residual_manifest = {}
        if _residual_artifacts_exist(residual_manifest):
            return residual_path
        force_rebuild = True
    if not build_missing:
        reason = "references missing artifacts" if force_rebuild else "is missing"
        raise FileNotFoundError(f"residual manifest {reason}: {residual_path}")
    hidden_path = source_root / "manifests/hidden_states.json"
    candidates_path = source_root / "candidate_tokens/candidate_tokens.json"
    missing = [str(path) for path in (hidden_path, candidates_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Residuals cannot be regenerated because saved-state inputs are missing: " + ", ".join(missing)
        )
    hidden = read_json(hidden_path)
    missing_arrays = [str(row.get("successor_path")) for row in hidden.get("layers", []) if not Path(str(row.get("successor_path"))).is_file()]
    if missing_arrays:
        raise FileNotFoundError("Residuals cannot be regenerated because hidden-state arrays are missing: " + ", ".join(missing_arrays))
    build_config_path = config_path
    temporary: tempfile.TemporaryDirectory[str] | None = None
    configured_root = Path(str(config.get("results_root", source_root)))
    if configured_root.resolve() != source_root.resolve():
        temporary = tempfile.TemporaryDirectory(prefix="shared-dictionary-residual-config-")
        adjusted = dict(config)
        adjusted["results_root"] = str(source_root)
        generated = Path(temporary.name) / "config.json"
        generated.write_text(json.dumps(adjusted, indent=2), encoding="utf-8")
        build_config_path = str(generated)
    command = [sys.executable, "-m", "experiments.prefix_response_subspaces.compute_contrast_residuals", "--config", build_config_path]
    if force_rebuild:
        command.append("--force")
    reason = "incomplete" if force_rebuild else "missing"
    print(f"[shared_dictionary] residuals {reason}; rebuilding from saved hidden states", flush=True)
    print("[shared_dictionary] " + " ".join(command), flush=True)
    try:
        subprocess.run(command, check=True)
    finally:
        if temporary is not None:
            temporary.cleanup()
    if not residual_path.is_file() or not _residual_artifacts_exist(read_json(residual_path)):
        raise RuntimeError(f"residual generation finished without creating a complete artifact set: {residual_path}")
    return residual_path


def _wrong_map(path: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for row in read_jsonl(path):
        result[str(row["prefix_id"])] = list(map(str, row.get("wrong_prefix_ids", [])))
    return result


def _fit_checkpoint(path: Path, signature: str, fit: SharedCovarianceFit | None = None) -> SharedCovarianceFit | None:
    import torch

    if fit is None:
        if not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as bundle:
            observed = str(bundle["signature"].item())
            if observed != signature:
                raise RuntimeError(f"fit checkpoint signature mismatch: {path}")
            kind = str(bundle["kind"].item())
            basis = torch.as_tensor(bundle["basis"])
            weights = torch.as_tensor(bundle["weights"])
            loss = float(bundle["loss"].item())
            selected_restart = int(bundle["selected_restart"].item())
        diagnostics = read_json(path.with_suffix(".json"))
        return SharedCovarianceFit(kind, basis, weights, loss, selected_restart, diagnostics)  # type: ignore[arg-type]
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.stem + "-", suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez(
            temporary,
            signature=np.asarray(signature),
            kind=np.asarray(fit.kind),
            basis=fit.basis.detach().cpu().numpy(),
            weights=fit.weights.detach().cpu().numpy(),
            loss=np.asarray(fit.loss),
            selected_restart=np.asarray(fit.selected_restart),
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    atomic_json(path.with_suffix(".json"), fit.diagnostics)
    return fit


def _to_device(fit: SharedCovarianceFit, device: str, dtype: str) -> SharedCovarianceFit:
    from .src.shared_dictionary import torch_dtype

    return SharedCovarianceFit(
        fit.kind,
        fit.basis.to(device=device, dtype=torch_dtype(dtype)),
        fit.weights.to(device=device, dtype=torch_dtype(dtype)),
        fit.loss,
        fit.selected_restart,
        fit.diagnostics,
    )


def _fit_or_load(
    residuals,
    dictionary_size: int,
    kind: str,
    optimization: OptimizationConfig,
    checkpoint: Path,
    signature: str,
    device: str,
    dtype: str,
    pooled=None,
    data_parallel_device_ids: list[int] | None = None,
    restart_devices: list[str] | None = None,
) -> SharedCovarianceFit:
    loaded = _fit_checkpoint(checkpoint, signature)
    if loaded is not None:
        print(f"[shared_dictionary] REUSE {checkpoint}", flush=True)
        return _to_device(loaded, device, dtype)
    print(
        f"[shared_dictionary] FIT kind={kind} K={dictionary_size} beta={optimization.coherence_penalty:g} "
        f"contexts={residuals.shape[0]} candidates={residuals.shape[1]} restarts={optimization.restarts}",
        flush=True,
    )
    try:
        fit = fit_shared_covariance_model(
            residuals, dictionary_size, kind=kind, config=optimization, pooled_basis=pooled,
            data_parallel_device_ids=data_parallel_device_ids, restart_devices=restart_devices,
        )
    except Exception as exc:
        atomic_json(checkpoint.with_suffix(".failed.json"), {
            "signature": signature, "kind": kind, "dictionary_size": dictionary_size,
            "coherence_penalty": optimization.coherence_penalty,
            "failure_type": type(exc).__name__, "failure": str(exc),
        })
        raise
    _fit_checkpoint(checkpoint, signature, fit)
    return fit


def _baseline_rows(
    train_r: np.ndarray,
    eval_r: np.ndarray,
    prefixes: list[dict[str, Any]],
    nonaux: np.ndarray,
    wrong: dict[str, list[str]],
    ranks: list[int],
    layer: int,
    fold: int,
    evaluation_group: str,
) -> tuple[list[dict[str, Any]], list[int]]:
    maximum_rank = max(ranks)
    local, matched = _fit_controls(train_r, prefixes, nonaux, maximum_rank)
    rows: list[dict[str, Any]] = []
    evaluated_positions: list[int] = []
    for local_index, full_index in enumerate(nonaux):
        prefix = prefixes[int(full_index)]
        if prefix.get("problem_group") != evaluation_group:
            continue
        prefix_id = str(prefix["prefix_id"])
        target = np.asarray(eval_r[local_index], dtype=np.float64)
        target_basis = local.get(prefix_id)
        matched_basis, matched_exact, matched_distance, _ = _resolve_conditional_basis(matched, prefix)
        wrong_bases = [local[item] for item in wrong.get(prefix_id, []) if item in local]
        if target_basis is None or matched_basis is None or not wrong_bases:
            raise RuntimeError(f"required target/matched/wrong basis is missing for {prefix_id}")
        evaluated_positions.append(local_index)
        common = {
            "problem_id": str(prefix["problem_id"]), "prefix_id": prefix_id,
            "problem_group": str(prefix["problem_group"]), "layer": int(layer), "fold": int(fold),
            "dictionary_size": "", "coherence_penalty": "", "leave_one_context_out": False,
            "train_candidate_count": int(train_r.shape[1]), "evaluation_candidate_count": int(eval_r.shape[1]),
        }
        for rank in ranks:
            target_ev = explained_variance(target, target_basis[:, :rank])
            matched_ev = explained_variance(target, matched_basis[:, :rank])
            wrong_evs = [explained_variance(target, basis[:, :rank]) for basis in wrong_bases]
            rows.extend([
                {**common, "method": METHOD_TARGET, "rank": rank, "heldout_ev": target_ev, "effective_rank": rank},
                {**common, "method": METHOD_MATCHED, "rank": rank, "heldout_ev": matched_ev, "effective_rank": rank,
                 "matched_common_exact": bool(matched_exact), "matched_common_length_distance": matched_distance},
                {**common, "method": METHOD_WRONG, "rank": rank, "heldout_ev": float(np.mean(wrong_evs)),
                 "effective_rank": rank, "wrong_context_count": len(wrong_evs)},
            ])
    return rows, evaluated_positions


def _new_model_rows(
    *,
    train_tensor,
    eval_r: np.ndarray,
    fit_positions: list[int],
    evaluation_positions: list[int],
    prefixes: list[dict[str, Any]],
    nonaux: np.ndarray,
    pooled,
    cpc: SharedCovarianceFit,
    dictionaries: list[tuple[float, SharedCovarianceFit]],
    ranks: list[int],
    dictionary_size: int,
    layer: int,
    fold: int,
) -> list[dict[str, Any]]:
    fit_lookup = {position: index for index, position in enumerate(fit_positions)}
    rows: list[dict[str, Any]] = []
    for position in evaluation_positions:
        if position not in fit_lookup:
            raise RuntimeError("an evaluation context is absent from shared-context fitting")
        fit_index = fit_lookup[position]
        prefix = prefixes[int(nonaux[position])]
        target = np.asarray(eval_r[position], dtype=np.float64)
        common = {
            "problem_id": str(prefix["problem_id"]), "prefix_id": str(prefix["prefix_id"]),
            "problem_group": str(prefix["problem_group"]), "layer": int(layer), "fold": int(fold),
            "dictionary_size": int(dictionary_size), "leave_one_context_out": False,
            "train_candidate_count": int(train_tensor.shape[1]), "evaluation_candidate_count": int(eval_r.shape[1]),
        }
        for rank in ranks:
            if rank > dictionary_size:
                continue
            pooled_u, _ = select_pooled_subspace(train_tensor, pooled, fit_index, rank)
            cpc_u, _ = covariance_subspace(cpc.basis, cpc.weights[fit_index], rank, kind="cpc")
            rows.extend([
                {**common, "method": METHOD_POOLED, "rank": rank, "coherence_penalty": "",
                 "heldout_ev": explained_variance(target, pooled_u.detach().cpu().numpy()), "effective_rank": rank},
                {**common, "method": METHOD_CPC, "rank": rank, "coherence_penalty": "",
                 "heldout_ev": explained_variance(target, cpc_u.detach().cpu().numpy()), "effective_rank": rank},
            ])
            for beta, fit in dictionaries:
                dictionary_u, _ = covariance_subspace(fit.basis, fit.weights[fit_index], rank, kind="dictionary")
                rows.append({
                    **common, "method": METHOD_DICTIONARY, "rank": rank, "coherence_penalty": beta,
                    "heldout_ev": explained_variance(target, dictionary_u.detach().cpu().numpy()), "effective_rank": rank,
                })
    return rows


def _loco_rows(
    *,
    train_tensor,
    eval_r: np.ndarray,
    fit_positions: list[int],
    evaluation_positions: list[int],
    prefixes: list[dict[str, Any]],
    nonaux: np.ndarray,
    dictionary_sizes: list[int],
    ranks: list[int],
    betas: list[float],
    base_optimization: OptimizationConfig,
    checkpoint_root: Path,
    signature: str,
    device: str,
    dtype: str,
    data_parallel_device_ids: list[int] | None,
    restart_devices: list[str] | None,
    layer: int,
    fold: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import torch

    lookup = {position: index for index, position in enumerate(fit_positions)}
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for position in evaluation_positions:
        fit_index = lookup[position]
        keep = torch.as_tensor([index for index in range(len(fit_positions)) if index != fit_index], device=train_tensor.device)
        excluded = train_tensor.index_select(0, keep)
        target_train = train_tensor[fit_index : fit_index + 1]
        prefix = prefixes[int(nonaux[position])]
        target_eval = np.asarray(eval_r[position], dtype=np.float64)
        pooled_maximum = pooled_pca_basis(excluded, max(dictionary_sizes))
        for dictionary_size in dictionary_sizes:
            pooled = pooled_maximum[:, :dictionary_size].contiguous()
            cpc_config = OptimizationConfig(**{**base_optimization.__dict__, "coherence_penalty": 0.0,
                                               "seed": _stable_seed(base_optimization.seed, fold, position, dictionary_size, "cpc_loco")})
            cpc_path = checkpoint_root / f"fold_{fold}/loco_{prefix['prefix_id']}_cpc_K{dictionary_size}.npz"
            cpc_signature = stable_hash({"run": signature, "fold": fold, "prefix": prefix["prefix_id"], "K": dictionary_size, "kind": "cpc_loco"})
            cpc = _fit_or_load(
                excluded, dictionary_size, "cpc", cpc_config, cpc_path, cpc_signature,
                device, dtype, pooled, data_parallel_device_ids, restart_devices,
            )
            cpc_weights = context_projection_variances(target_train, cpc.basis)[0]
            diagnostics.append({"fold": fold, "excluded_prefix_id": prefix["prefix_id"], "kind": "cpc_loco", **cpc.diagnostics})
            dictionary_fits: list[tuple[float, SharedCovarianceFit, Any, dict[str, Any]]] = []
            for beta in betas:
                dictionary_config = OptimizationConfig(**{
                    **base_optimization.__dict__, "coherence_penalty": beta,
                    "seed": _stable_seed(base_optimization.seed, fold, position, dictionary_size, beta, "dictionary_loco"),
                })
                label = f"{beta:.0e}".replace("+", "")
                path = checkpoint_root / f"fold_{fold}/loco_{prefix['prefix_id']}_dictionary_K{dictionary_size}_beta{label}.npz"
                fit_signature = stable_hash({"run": signature, "fold": fold, "prefix": prefix["prefix_id"], "K": dictionary_size, "beta": beta, "kind": "dictionary_loco"})
                fit = _fit_or_load(
                    excluded, dictionary_size, "dictionary", dictionary_config, path, fit_signature,
                    device, dtype, pooled, data_parallel_device_ids, restart_devices,
                )
                heldout_weights, weight_diagnostics = fit_fixed_dictionary_weights(
                    target_train, fit.basis, learning_rate=base_optimization.learning_rate,
                    maximum_steps=base_optimization.maximum_steps, patience=base_optimization.patience,
                    seed=_stable_seed(base_optimization.seed, fold, position, dictionary_size, beta, "weights"),
                    epsilon=base_optimization.epsilon, full_loss_interval=base_optimization.full_loss_interval,
                )
                dictionary_fits.append((beta, fit, heldout_weights, weight_diagnostics))
                diagnostics.append({
                    "fold": fold, "excluded_prefix_id": prefix["prefix_id"], "kind": "dictionary_loco",
                    "heldout_weight_fit": weight_diagnostics, **fit.diagnostics,
                })
            common = {
                "problem_id": str(prefix["problem_id"]), "prefix_id": str(prefix["prefix_id"]),
                "problem_group": str(prefix["problem_group"]), "layer": layer, "fold": fold,
                "dictionary_size": dictionary_size, "leave_one_context_out": True,
                "train_candidate_count": int(train_tensor.shape[1]), "evaluation_candidate_count": int(eval_r.shape[1]),
            }
            for rank in ranks:
                if rank > dictionary_size:
                    continue
                cpc_u, _ = covariance_subspace(cpc.basis, cpc_weights, rank, kind="cpc")
                rows.append({
                    **common, "method": METHOD_CPC, "rank": rank, "coherence_penalty": "",
                    "heldout_ev": explained_variance(target_eval, cpc_u.detach().cpu().numpy()), "effective_rank": rank,
                })
                for beta, fit, heldout_weights, _ in dictionary_fits:
                    basis, _ = covariance_subspace(fit.basis, heldout_weights, rank, kind="dictionary")
                    rows.append({
                        **common, "method": METHOD_DICTIONARY, "rank": rank, "coherence_penalty": beta,
                        "heldout_ev": explained_variance(target_eval, basis.detach().cpu().numpy()), "effective_rank": rank,
                    })
    return rows, diagnostics


def fold_mean_by_problem(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ("problem_id", "method", "dictionary_size", "rank", "coherence_penalty", "leave_one_context_out")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key, "") for key in keys)].append(row)
    result = []
    for key, values in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        result.append({
            **dict(zip(keys, key)),
            "heldout_ev": float(np.mean([float(row["heldout_ev"]) for row in values])),
            "fold_count": len({int(row["fold"]) for row in values}),
            "context_fold_rows": len(values),
        })
    return result


def _bootstrap(values: np.ndarray, replicates: int, seed: int, ci: float) -> tuple[float, float, float, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan"), float("nan"), np.empty(0)
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(replicates, len(values)))].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    return float(values.mean()), float(np.quantile(samples, alpha)), float(np.quantile(samples, 1.0 - alpha)), samples


def paired_summaries(problem_rows: list[dict[str, Any]], *, replicates: int, seed: int, ci: float) -> list[dict[str, Any]]:
    target = {
        (str(row["problem_id"]), int(row["rank"])): float(row["heldout_ev"])
        for row in problem_rows
        if row["method"] == METHOD_TARGET and not bool(row["leave_one_context_out"])
    }
    group_keys = ("method", "dictionary_size", "rank", "coherence_penalty", "leave_one_context_out")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in problem_rows:
        groups[tuple(row.get(key, "") for key in group_keys)].append(row)
    output = []
    for group, rows in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        method, dictionary_size, rank, beta, loco = group
        paired = [
            (target[(str(row["problem_id"]), int(rank))], float(row["heldout_ev"]))
            for row in rows if (str(row["problem_id"]), int(rank)) in target
        ]
        if not paired:
            continue
        method_values = np.asarray([item[1] for item in paired])
        differences = np.asarray([item[0] - item[1] for item in paired])
        group_seed = _stable_seed(seed, *group)
        mean_ev, ev_low, ev_high, _ = _bootstrap(method_values, replicates, group_seed, ci)
        mean_delta, delta_low, delta_high, delta_samples = _bootstrap(differences, replicates, group_seed + 1, ci)
        if method == METHOD_TARGET:
            mean_delta = delta_low = delta_high = 0.0
            p_value = 1.0
        else:
            lower = (1 + int(np.sum(delta_samples <= 0.0))) / (len(delta_samples) + 1)
            upper = (1 + int(np.sum(delta_samples >= 0.0))) / (len(delta_samples) + 1)
            p_value = min(1.0, 2.0 * min(lower, upper))
        output.append({
            "method": method, "dictionary_size": dictionary_size, "rank": rank,
            "coherence_penalty": beta, "leave_one_context_out": loco,
            "heldout_ev_mean": mean_ev, "heldout_ev_ci_low": ev_low, "heldout_ev_ci_high": ev_high,
            "target_minus_method_mean": mean_delta, "target_minus_method_ci_low": delta_low,
            "target_minus_method_ci_high": delta_high, "two_sided_bootstrap_p": p_value,
            "problem_count": len(paired), "folds_averaged_before_bootstrap": True,
            "bootstrap_replicates": replicates, "bootstrap_seed": group_seed,
        })
    return output


def _latex(path: Path, summaries: list[dict[str, Any]]) -> None:
    ranks = [int(row["rank"]) for row in summaries]
    primary_rank = 64 if 64 in ranks else max(ranks)
    selected = [row for row in summaries if int(row["rank"]) == primary_rank and not bool(row["leave_one_context_out"])]
    order = {METHOD_TARGET: 0, METHOD_MATCHED: 1, METHOD_WRONG: 2, METHOD_POOLED: 3, METHOD_CPC: 4, METHOD_DICTIONARY: 5}
    selected.sort(key=lambda row: (order.get(str(row["method"]), 99), str(row["dictionary_size"]), str(row["coherence_penalty"])))
    names = {
        METHOD_TARGET: "Target-context PCA", METHOD_MATCHED: "Matched common", METHOD_WRONG: "Wrong context",
        METHOD_POOLED: "Pooled PCA with context selection", METHOD_CPC: "Truncated CPC",
        METHOD_DICTIONARY: "Shared non-orthogonal dictionary",
    }
    lines = [
        "\\begin{tabular}{llrrrr}", "\\toprule",
        "Method & Dictionary size & Rank & Held-out EV & Target minus method & 95\\% CI \\\\", "\\midrule",
    ]
    for row in selected:
        size = str(row["dictionary_size"] or "context-specific" if row["method"] == METHOD_TARGET else row["dictionary_size"] or "existing")
        if row["method"] == METHOD_DICTIONARY:
            size += f" ($\\beta={float(row['coherence_penalty']):g}$)"
        method_name = names.get(str(row["method"]), str(row["method"])).replace("_", "\\_")
        lines.append(
            f"{method_name} & {size} & {int(row['rank'])} & "
            f"{float(row['heldout_ev_mean']):.4f} & {float(row['target_minus_method_mean']):.4f} & "
            f"[{float(row['target_minus_method_ci_low']):.4f}, {float(row['target_minus_method_ci_high']):.4f}] \\\\" 
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _svg(path: Path, summaries: list[dict[str, Any]], value_key: str, title: str) -> None:
    numeric = [row for row in summaries if isinstance(row["dictionary_size"], int) and not bool(row["leave_one_context_out"])]
    ranks = sorted({int(row["rank"]) for row in numeric})
    if not ranks:
        path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"640\" height=\"80\"><text x=\"20\" y=\"40\">No numeric results</text></svg>\n", encoding="utf-8")
        return
    width, panel_height, margin = 920, 250, 58
    height = 45 + panel_height * len(ranks)
    colors = {METHOD_POOLED: "#4C78A8", METHOD_CPC: "#F58518", METHOD_DICTIONARY: "#54A24B"}
    values = np.asarray([float(row[value_key]) for row in numeric], dtype=np.float64)
    ymin, ymax = float(np.nanmin(values)), float(np.nanmax(values))
    if math.isclose(ymin, ymax):
        ymin -= 0.01; ymax += 0.01
    padding = 0.08 * (ymax - ymin)
    ymin -= padding; ymax += padding
    sizes = sorted({int(row["dictionary_size"]) for row in numeric})
    xmin, xmax = min(sizes), max(sizes)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="{width/2}" y="25" text-anchor="middle" font-family="sans-serif" font-size="17">{html.escape(title)}</text>']
    for panel, rank in enumerate(ranks):
        top = 40 + panel * panel_height
        left, right = 85, width - 40
        bottom = top + panel_height - 45
        parts.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#333"/>')
        parts.append(f'<line x1="{left}" y1="{top+10}" x2="{left}" y2="{bottom}" stroke="#333"/>')
        parts.append(f'<text x="{left+5}" y="{top+27}" font-family="sans-serif" font-size="14">rank {rank}</text>')
        for size in sizes:
            x = left + (size - xmin) / max(1, xmax - xmin) * (right - left)
            parts.append(f'<line x1="{x:.1f}" y1="{bottom}" x2="{x:.1f}" y2="{bottom+5}" stroke="#333"/>')
            parts.append(f'<text x="{x:.1f}" y="{bottom+20}" text-anchor="middle" font-family="sans-serif" font-size="11">{size}</text>')
        for tick in range(5):
            value = ymin + tick * (ymax - ymin) / 4
            y = bottom - (value - ymin) / (ymax - ymin) * (bottom - top - 10)
            parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#ddd"/>')
            parts.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="10">{value:.3f}</text>')
        series: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in numeric:
            if int(row["rank"]) == rank:
                series[(str(row["method"]), str(row["coherence_penalty"]))].append(row)
        for (method, beta), rows in sorted(series.items()):
            rows.sort(key=lambda row: int(row["dictionary_size"]))
            points = []
            for row in rows:
                x = left + (int(row["dictionary_size"]) - xmin) / max(1, xmax - xmin) * (right - left)
                y = bottom - (float(row[value_key]) - ymin) / (ymax - ymin) * (bottom - top - 10)
                points.append((x, y))
            color = colors.get(method, "#777")
            dash_length = 3 + int(hashlib.sha256(beta.encode("utf-8")).hexdigest()[:2], 16) % 7
            dash = "" if not beta else f' stroke-dasharray="{dash_length} 3"'
            parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2"{dash} points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in points) + '"/>')
            for x, y in points:
                parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}"/>')
    legend_y = height - 8
    parts.append(f'<text x="90" y="{legend_y}" font-family="sans-serif" font-size="11">Blue: pooled PCA; orange: CPC; green: non-orthogonal dictionary (dash by beta)</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    import torch

    config = load_config(args.config)
    if not args.smoke and str(config.get("model", {}).get("checkpoint")) != "Qwen/Qwen2.5-Math-1.5B":
        raise ValueError("the confirmatory shared-dictionary experiment requires Qwen/Qwen2.5-Math-1.5B")
    source_root = Path(args.source_results_root or config.get("source_results_root", config["results_root"]))
    residual_path = _ensure_residuals(args.config, config, source_root, build_missing=not args.no_build_residuals)
    if args.residuals_only:
        print(residual_path)
        return residual_path
    required = {
        "hidden": source_root / "manifests/hidden_states.json",
        "candidates": source_root / "candidate_tokens/candidate_tokens.json",
        "wrong": source_root / "controls/wrong_prefixes.jsonl",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("shared dictionary inputs are missing: " + ", ".join(missing))
    hidden = read_json(required["hidden"])
    prefixes = read_jsonl(hidden["prefix_snapshot"])
    residual_manifest = read_json(residual_path)
    geometry_path = source_root / "metrics/paper_geometry_summary.json"
    selected_paper_layer = int(read_json(geometry_path)["selected_layer"]) if geometry_path.is_file() else None
    if args.layer is None:
        if not geometry_path.is_file():
            raise FileNotFoundError("selected paper layer is unavailable; restore paper_geometry_summary.json or pass --layer")
        layer = int(selected_paper_layer)
    else:
        layer = int(args.layer)
        if not args.smoke and selected_paper_layer is not None and layer != selected_paper_layer:
            raise ValueError(f"--layer {layer} differs from the main experiment's selected layer {selected_paper_layer}")
    data_parallel_device_ids = list(map(int, args.data_parallel_device_ids or []))
    restart_parallel_device_ids = list(map(int, args.restart_parallel_device_ids or []))
    if data_parallel_device_ids and restart_parallel_device_ids:
        raise ValueError("choose either DataParallel or restart parallelism, not both")
    parallel_device_ids = data_parallel_device_ids or restart_parallel_device_ids
    if parallel_device_ids:
        flag = "--data-parallel-device-ids" if data_parallel_device_ids else "--restart-parallel-device-ids"
        if len(parallel_device_ids) < 2 or len(set(parallel_device_ids)) != len(parallel_device_ids):
            raise ValueError(f"{flag} requires at least two unique device IDs")
        if not torch.cuda.is_available():
            raise RuntimeError("multi-GPU optimization requires CUDA")
        visible = torch.cuda.device_count()
        if any(item < 0 or item >= visible for item in parallel_device_ids):
            raise ValueError(
                f"requested CUDA devices {parallel_device_ids}, but the container exposes {visible} CUDA devices"
            )
        device = f"cuda:{parallel_device_ids[0]}"
    else:
        device = _resolve_device(args.device)
    restart_devices = [f"cuda:{item}" for item in restart_parallel_device_ids] or None
    if args.allow_tf32:
        if args.dtype != "float32" or not str(device).startswith("cuda"):
            raise ValueError("--allow-tf32 requires --dtype float32 and a CUDA device")
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
    if args.loco_only and not args.leave_one_context_out:
        raise ValueError("--loco-only requires --leave-one-context-out")
    if args.smoke:
        dictionary_sizes = [min(args.dictionary_sizes)]
        ranks = [rank for rank in args.evaluation_ranks if rank <= dictionary_sizes[0]][:1]
        betas = [args.coherence_penalties[0]]
        restarts = min(2, args.restarts)
        maximum_steps = min(30, args.maximum_steps)
        patience = min(10, args.patience)
    else:
        dictionary_sizes = sorted(set(map(int, args.dictionary_sizes)))
        configured = set(map(int, config.get("analysis", {}).get("ranks", [])))
        ranks = sorted(configured | set(map(int, args.evaluation_ranks)))
        betas = sorted(set(map(float, args.coherence_penalties)))
        restarts = int(args.restarts)
        maximum_steps = int(args.maximum_steps)
        patience = int(args.patience)
        if restarts < 5:
            raise ValueError("confirmatory runs require at least five random restarts; use --smoke for a smaller check")
    if any(size <= 0 for size in dictionary_sizes) or any(rank <= 0 for rank in ranks):
        raise ValueError("dictionary sizes and ranks must be positive")
    if max(ranks) > max(dictionary_sizes):
        ranks = [rank for rank in ranks if rank <= max(dictionary_sizes)]
    base_optimization = OptimizationConfig(
        learning_rate=args.learning_rate, maximum_steps=maximum_steps, patience=patience,
        restarts=restarts, seed=args.optimization_seed, epsilon=args.epsilon,
        context_batch_size=args.context_batch_size, full_loss_interval=args.full_loss_interval,
    )
    run_spec = {
        "source_root": str(source_root), "residual_manifest_sha256": file_sha256(residual_path),
        "layer": layer, "dictionary_sizes": dictionary_sizes, "ranks": ranks, "coherence_penalties": betas,
        "learning_rate": args.learning_rate, "maximum_steps": maximum_steps, "patience": patience,
        "restarts": restarts, "optimization_seed": args.optimization_seed, "device": device, "dtype": args.dtype,
        "context_batch_size": args.context_batch_size, "full_loss_interval": args.full_loss_interval,
        "parallelism": (
            "torch.nn.DataParallel" if data_parallel_device_ids
            else "independent_restarts" if restart_parallel_device_ids else "single_device"
        ),
        "data_parallel_device_ids": data_parallel_device_ids,
        "restart_parallel_device_ids": restart_parallel_device_ids,
        "allow_tf32": bool(args.allow_tf32),
        "evaluation_group": args.evaluation_group, "shared_context_groups": args.shared_context_groups,
        "leave_one_context_out": args.leave_one_context_out, "loco_only": args.loco_only,
        "loco_context_limit": args.loco_context_limit,
        "smoke": bool(args.smoke), "model_checkpoint": config.get("model", {}).get("checkpoint"),
        "model_revision": config.get("model", {}).get("revision"),
        "saved_hidden_state_model": hidden.get("model", {}),
    }
    signature = stable_hash(run_spec)
    output_base = Path(args.output_root or source_root / "shared_dictionary_controls")
    output_root = output_base / f"run_{signature[:12]}"
    output_root.mkdir(parents=True, exist_ok=True)
    final_manifest = output_root / "manifest.json"
    required_review_outputs = [
        output_root / "paired_summary.csv",
        output_root / "optimization_diagnostics.json",
        output_root / "heldout_ev_all.csv",
    ]
    if final_manifest.is_file() and read_json(final_manifest).get("complete"):
        missing_review_outputs = [path for path in required_review_outputs if not path.is_file()]
        if not missing_review_outputs:
            print(final_manifest)
            return final_manifest
        print(
            "[shared_dictionary] completed run is missing reviewer artifacts; "
            "rebuilding summaries from saved fit checkpoints: "
            + ", ".join(map(str, missing_review_outputs)),
            flush=True,
        )
    atomic_json(output_root / "config_resolved.json", {"paper_config": config, "shared_dictionary": run_spec})
    wrong = _wrong_map(required["wrong"])
    candidates = read_json(required["candidates"])
    candidate_folds = {int(row["fold_id"]): row for row in candidates.get("folds", [])}
    all_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    entries = [row for row in residual_manifest["entries"] if int(row["layer"]) == layer]
    if len(entries) != int(config.get("candidates", {}).get("folds", len(entries))):
        raise RuntimeError(f"expected all candidate folds at layer {layer}; found {len(entries)}")
    for entry in sorted(entries, key=lambda row: int(row["fold"])):
        fold = int(entry["fold"])
        bundle = load_residual_entry(entry)
        train_r = np.asarray(bundle["train_residuals"])
        eval_r = np.asarray(bundle["evaluation_residuals"])
        nonaux = np.asarray(bundle["nonauxiliary_prefix_indices"], dtype=np.int64)
        if train_r.shape[1] != 192 or eval_r.shape[1] != 64:
            if not args.smoke:
                raise RuntimeError(f"fold {fold} has {train_r.shape[1]}/{eval_r.shape[1]} candidates, expected 192/64")
        if set(map(int, bundle["train_candidate_indices"])) & set(map(int, bundle["evaluation_candidate_indices"])):
            raise RuntimeError(f"candidate leakage in fold {fold}")
        expected_fold = candidate_folds.get(fold)
        if expected_fold is None:
            raise RuntimeError(f"fold {fold} is absent from the frozen candidate split")
        if list(map(int, bundle["train_candidate_indices"])) != list(map(int, expected_fold["train_indices"])):
            raise RuntimeError(f"residual training candidates differ from the frozen split in fold {fold}")
        if list(map(int, bundle["evaluation_candidate_indices"])) != list(map(int, expected_fold["evaluation_indices"])):
            raise RuntimeError(f"residual evaluation candidates differ from the frozen split in fold {fold}")
        baseline, evaluation_positions = _baseline_rows(
            train_r, eval_r, prefixes, nonaux, wrong, ranks, layer, fold, args.evaluation_group,
        )
        if args.smoke:
            evaluation_positions = evaluation_positions[: min(4, len(evaluation_positions))]
            keep_prefixes = {str(prefixes[int(nonaux[position])]["prefix_id"]) for position in evaluation_positions}
            baseline = [row for row in baseline if str(row["prefix_id"]) in keep_prefixes]
        all_rows.extend(baseline)
        fit_positions = [
            index for index, full_index in enumerate(nonaux)
            if not args.shared_context_groups or prefixes[int(full_index)].get("problem_group") in args.shared_context_groups
        ]
        if args.smoke:
            required_positions = list(dict.fromkeys(evaluation_positions + fit_positions))
            fit_positions = required_positions[: max(4, len(evaluation_positions))]
            for position in evaluation_positions:
                if position not in fit_positions:
                    fit_positions.append(position)
        train_tensor = as_residual_tensor(train_r[fit_positions], device=device, dtype=args.dtype)
        main_dictionary_sizes = [] if args.loco_only else dictionary_sizes
        pooled_maximum = pooled_pca_basis(train_tensor, max(main_dictionary_sizes)) if main_dictionary_sizes else None
        for dictionary_size in main_dictionary_sizes:
            if dictionary_size > train_tensor.shape[2] or dictionary_size > train_tensor.shape[0] * train_tensor.shape[1]:
                raise RuntimeError(f"K={dictionary_size} exceeds the available pooled rank")
            assert pooled_maximum is not None
            pooled = pooled_maximum[:, :dictionary_size].contiguous()
            cpc_config = OptimizationConfig(**{
                **base_optimization.__dict__, "coherence_penalty": 0.0,
                "seed": _stable_seed(args.optimization_seed, fold, dictionary_size, "cpc"),
            })
            cpc_signature = stable_hash({"run": signature, "fold": fold, "K": dictionary_size, "kind": "cpc"})
            cpc = _fit_or_load(
                train_tensor, dictionary_size, "cpc", cpc_config,
                output_root / f"checkpoints/fold_{fold}/cpc_K{dictionary_size}.npz",
                cpc_signature, device, args.dtype, pooled, data_parallel_device_ids, restart_devices,
            )
            diagnostics.append({"fold": fold, **cpc.diagnostics})
            dictionary_fits = []
            for beta in betas:
                dictionary_config = OptimizationConfig(**{
                    **base_optimization.__dict__, "coherence_penalty": beta,
                    "seed": _stable_seed(args.optimization_seed, fold, dictionary_size, beta, "dictionary"),
                })
                label = f"{beta:.0e}".replace("+", "")
                fit_signature = stable_hash({"run": signature, "fold": fold, "K": dictionary_size, "beta": beta, "kind": "dictionary"})
                fit = _fit_or_load(
                    train_tensor, dictionary_size, "dictionary", dictionary_config,
                    output_root / f"checkpoints/fold_{fold}/dictionary_K{dictionary_size}_beta{label}.npz",
                    fit_signature, device, args.dtype, pooled, data_parallel_device_ids, restart_devices,
                )
                diagnostics.append({"fold": fold, **fit.diagnostics})
                dictionary_fits.append((beta, fit))
            all_rows.extend(_new_model_rows(
                train_tensor=train_tensor, eval_r=eval_r, fit_positions=fit_positions,
                evaluation_positions=evaluation_positions, prefixes=prefixes, nonaux=nonaux,
                pooled=pooled, cpc=cpc, dictionaries=dictionary_fits, ranks=ranks,
                dictionary_size=dictionary_size, layer=layer, fold=fold,
            ))
        if args.leave_one_context_out:
            loco_positions = evaluation_positions[: args.loco_context_limit or None]
            loco_rows, loco_diagnostics = _loco_rows(
                train_tensor=train_tensor, eval_r=eval_r, fit_positions=fit_positions,
                evaluation_positions=loco_positions, prefixes=prefixes, nonaux=nonaux,
                dictionary_sizes=dictionary_sizes, ranks=ranks, betas=betas,
                base_optimization=base_optimization, checkpoint_root=output_root / "checkpoints",
                signature=signature, device=device, dtype=args.dtype, layer=layer, fold=fold,
                data_parallel_device_ids=data_parallel_device_ids, restart_devices=restart_devices,
            )
            all_rows.extend(loco_rows)
            diagnostics.extend(loco_diagnostics)
        del train_tensor
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    raw_path = output_root / "heldout_ev_all.csv"
    problem_path = output_root / "heldout_ev_problem_fold_mean.csv"
    summary_path = output_root / "paired_summary.csv"
    diagnostic_path = output_root / "optimization_diagnostics.json"
    table_path = output_root / "shared_dictionary_table.tex"
    _csv(raw_path, all_rows)
    problem_rows = fold_mean_by_problem(all_rows)
    _csv(problem_path, problem_rows)
    summary_rows = paired_summaries(
        problem_rows, replicates=int(config["statistics"]["bootstrap_replicates"]),
        seed=int(config["seed"]), ci=float(config["statistics"]["ci"]),
    )
    _csv(summary_path, summary_rows)
    atomic_json(diagnostic_path, {
        "run": run_spec, "fit_diagnostics": diagnostics,
        "failure_policy": "NaN, Inf, SVD, eigenspace, normalization, and QR failures are recorded per restart; all-restart failure aborts the run",
        "evaluation_data_used_for_fitting": False,
    })
    _latex(table_path, summary_rows)
    _svg(output_root / "heldout_ev_by_dictionary_size.svg", summary_rows, "heldout_ev_mean", "Held-out EV by dictionary size and rank")
    _svg(output_root / "target_gap_by_dictionary_size.svg", summary_rows, "target_minus_method_mean", "Target-context PCA minus shared model")
    manifest_inputs = {**required, "residuals": residual_path}
    if geometry_path.is_file():
        manifest_inputs["paper_geometry"] = geometry_path
    manifest = {
        "complete": True, "signature": signature, "run": run_spec,
        "inputs": {name: {"path": str(path), "sha256": file_sha256(path)} for name, path in manifest_inputs.items()},
        "outputs": {name: str(path) for name, path in {
            "heldout_ev_all": raw_path, "problem_fold_mean": problem_path, "paired_summary": summary_path,
            "optimization_diagnostics": diagnostic_path, "latex_table": table_path,
            "heldout_ev_figure": output_root / "heldout_ev_by_dictionary_size.svg",
            "target_gap_figure": output_root / "target_gap_by_dictionary_size.svg",
        }.items()},
        "row_count": len(all_rows), "problem_row_count": len(problem_rows), "summary_row_count": len(summary_rows),
        "forward_passes_executed": 0,
    }
    atomic_json(final_manifest, manifest)
    print(final_manifest)
    return final_manifest


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Evaluate shared direction dictionaries from saved candidate-response states")
    value.add_argument("--config", required=True)
    value.add_argument("--source-results-root")
    value.add_argument("--output-root")
    value.add_argument("--dictionary-sizes", nargs="+", type=int, default=[64, 96, 128, 160, 192, 256])
    value.add_argument("--evaluation-ranks", nargs="+", type=int, default=[8, 16, 32, 64])
    value.add_argument("--learning-rate", type=float, default=0.02)
    value.add_argument("--maximum-steps", type=int, default=2000)
    value.add_argument("--patience", type=int, default=200)
    value.add_argument("--restarts", type=int, default=5)
    value.add_argument("--optimization-seed", type=int, default=1729)
    value.add_argument("--context-batch-size", type=int, default=32)
    value.add_argument("--full-loss-interval", type=int, default=25)
    value.add_argument("--coherence-penalties", nargs="+", type=float, default=[0.0, 1e-4, 1e-3])
    value.add_argument("--device", default="auto")
    value.add_argument(
        "--data-parallel-device-ids", nargs="+", type=int,
        help="CUDA IDs visible inside the container, for example: 0 1",
    )
    value.add_argument(
        "--restart-parallel-device-ids", nargs="+", type=int,
        help="run independent optimization restarts concurrently on these container-visible CUDA IDs",
    )
    value.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    value.add_argument(
        "--allow-tf32", action="store_true",
        help="allow TensorFloat-32 CUDA matrix multiplications for faster approximate float32 optimization",
    )
    value.add_argument("--epsilon", type=float, default=1e-12)
    value.add_argument("--layer", type=int)
    value.add_argument("--evaluation-group", default="analysis_test")
    value.add_argument(
        "--shared-context-groups", nargs="*",
        default=["analysis_train", "analysis_dev", "analysis_test"],
    )
    value.add_argument("--leave-one-context-out", action="store_true")
    value.add_argument("--loco-only", action="store_true", help="run only the leave-one-context-out auxiliary fits")
    value.add_argument("--loco-context-limit", type=int)
    value.add_argument("--no-build-residuals", action="store_true")
    value.add_argument("--residuals-only", action="store_true")
    value.add_argument("--smoke", action="store_true")
    return value


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
