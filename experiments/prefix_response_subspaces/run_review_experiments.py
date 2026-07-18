from __future__ import annotations

import argparse
import copy
import os
import sys
import tempfile
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Iterator

from .src.utils import atomic_json, load_config, read_json

from . import (
    analyze_candidate_distribution_transfer,
    analyze_context_controls,
    analyze_jacobian_alignment,
    analyze_response_law_state,
    analyze_subspace_stability,
    build_review_candidate_sets,
    build_review_context_controls,
    extract_candidate_transfer_states,
    extract_context_control_states,
    extract_jacobian_responses,
)


STAGES = [
    build_review_candidate_sets,
    extract_candidate_transfer_states,
    analyze_candidate_distribution_transfer,
    build_review_context_controls,
    extract_context_control_states,
    analyze_context_controls,
    extract_jacobian_responses,
    analyze_jacobian_alignment,
    analyze_subspace_stability,
    analyze_response_law_state,
]
MODEL_STAGES = {
    build_review_candidate_sets, build_review_context_controls,
    extract_candidate_transfer_states, extract_context_control_states, extract_jacobian_responses,
    analyze_response_law_state,
}
FORWARD_STAGES = {
    extract_candidate_transfer_states, extract_context_control_states,
    extract_jacobian_responses, analyze_response_law_state,
}


@contextmanager
def quick_check_config(config_path: str) -> Iterator[str]:
    """Create an isolated, explicitly non-confirmatory pilot configuration."""
    config = copy.deepcopy(load_config(config_path))
    source_output = Path(str(config["results_root"]))
    config["profile"] = f"{config.get('profile', 'review')}_quick_check"
    config["results_root"] = str(source_output.with_name(f"{source_output.name}_quick_check"))
    config["quick_check"] = True
    config["candidates"]["proposal_top_k"] = min(int(config["candidates"]["proposal_top_k"]), 256)
    config["statistics"]["bootstrap_replicates"] = 50
    config["review_experiments"].update({
        "independent_candidate_set_size": 64,
        "candidate_selection_contexts_per_half": 8,
        "candidate_transfer_groups": ["auxiliary", "analysis_test"],
        "candidate_transfer_max_contexts_per_group": 8,
        "candidate_transfer_rank": 8,
        "candidate_transfer_equalize_fit_counts": True,
        "two_way_bootstrap_replicates": 50,
        "context_control_max_targets": 4,
        "context_control_auxiliary_contexts": 4,
        "context_control_candidate_limit": 64,
        "context_control_rank": 8,
        "stability_max_targets": 8,
        "stability_rank": 8,
        "subspace_candidate_bootstrap_replicates": 2,
        "subspace_candidate_bootstrap_targets": 4,
    })
    with tempfile.TemporaryDirectory(prefix="response-geometry-quick-config-") as temporary:
        output = Path(temporary) / "quick_check.json"
        atomic_json(output, config)
        print(
            f"[review_pipeline] QUICK_CHECK output_root={config['results_root']} "
            "candidate_sets=64 selection_contexts=8/half transfer_contexts=8/group "
            "control_targets=4 control_candidates=64 rank=8 bootstrap=50 jacobian=skipped",
            flush=True,
        )
        yield str(output)


@contextmanager
def formal_check_config(config_path: str) -> Iterator[str]:
    """Create an isolated medium-scale reviewer experiment configuration."""
    config = copy.deepcopy(load_config(config_path))
    source_output = Path(str(config["results_root"]))
    config["profile"] = f"{config.get('profile', 'review')}_formal_check"
    config["results_root"] = str(source_output.with_name(f"{source_output.name}_formal_check"))
    config["formal_check"] = True
    config["candidates"]["proposal_top_k"] = min(int(config["candidates"]["proposal_top_k"]), 512)
    config["statistics"]["bootstrap_replicates"] = 500
    config["review_experiments"].update({
        "independent_candidate_set_size": 160,
        "candidate_selection_contexts_per_half": 32,
        "candidate_transfer_groups": ["auxiliary", "analysis_test"],
        "candidate_transfer_max_contexts_per_group": 32,
        "candidate_transfer_rank": 16,
        "candidate_transfer_equalize_fit_counts": True,
        "two_way_bootstrap_replicates": 500,
        "context_control_max_targets": 16,
        "context_control_auxiliary_contexts": 32,
        "context_control_candidate_limit": 128,
        "context_control_rank": 16,
        "jacobian_embedding_components": 16,
        "jacobian_target_contexts": 4,
        "jacobian_auxiliary_contexts": 4,
        "jacobian_alignment_rank": 16,
        "stability_max_targets": 32,
        "stability_rank": 16,
        "subspace_candidate_bootstrap_replicates": 5,
        "subspace_candidate_bootstrap_targets": 16,
    })
    with tempfile.TemporaryDirectory(prefix="response-geometry-formal-config-") as temporary:
        output = Path(temporary) / "formal_check.json"
        atomic_json(output, config)
        print(
            f"[review_pipeline] FORMAL_CHECK output_root={config['results_root']} "
            "candidate_sets=160 selection_contexts=32/half transfer_contexts=32/group "
            "control_targets=16 control_candidates=128 rank=16 bootstrap=500 "
            "jacobian=4+4_contexts/16_components",
            flush=True,
        )
        yield str(output)


def print_check_summary(config_path: str, *, formal: bool = False) -> None:
    """Print only the directional pilot quantities needed for a go/no-go check."""
    config = load_config(config_path)
    root = Path(str(config["results_root"]))
    candidate_path = root / "metrics/candidate_distribution_transfer_summary.json"
    context_path = root / "metrics/context_control_summary.json"
    stability_path = root / "metrics/subspace_stability_summary.json"
    response_state_path = root / "metrics/response_law_state_summary.json"
    tag = "formal_result" if formal else "quick_result"
    status = "FORMAL_CHECK" if formal else "PILOT_ONLY not_for_paper_claims"
    print(f"[{tag}] {status}", flush=True)
    if candidate_path.is_file():
        candidate = read_json(candidate_path)
        for name in ("high_to_low", "low_to_high", "independent_A_to_B", "independent_B_to_A"):
            row = candidate.get("pairs", {}).get(name)
            if row:
                print(
                    f"[{tag}] candidate pair={name} "
                    f"transfer_fraction={row['transfer_fraction_of_target_reference']:.4f} "
                    f"delta={row['transfer_minus_reference_problem_bootstrap']['mean']:.4f}",
                    flush=True,
                )
    if context_path.is_file():
        context = read_json(context_path)
        for name, row in sorted(context.get("controls", {}).items()):
            delta = row.get("delta_target_minus_control_problem_bootstrap", {}).get("mean")
            if delta is not None:
                print(f"[{tag}] context control={name} target_minus_control={delta:.4f}", flush=True)
    if stability_path.is_file():
        stability = read_json(stability_path)
        delta = stability.get("between_minus_within_distance", {}).get("mean")
        if delta is not None:
            print(f"[{tag}] stability between_minus_within_distance={delta:.4f}", flush=True)
    if response_state_path.is_file():
        response_state = read_json(response_state_path)
        current = response_state.get("current_distribution_match", {}).get("mean")
        future = response_state.get("future_divergence", {}).get("mean")
        contrast = response_state.get("contrast_geometry", {})
        alignment = contrast.get("candidate_identity_alignment", {})
        cka_delta = alignment.get("observed_minus_permutation_mean")
        law_r = contrast.get("response_law_distance_correspondence", {}).get(
            "pearson_pair_bootstrap", {}
        ).get("mean")
        if None not in (current, future, cka_delta, law_r):
            print(
                f"[{tag}] response_state current_js={current:.4f} future_js={future:.4f} "
                f"candidate_cka_above_null={cka_delta:.4f} law_distance_r={law_r:.4f}",
                flush=True,
            )
    print(f"[{tag}] output_root={root}", flush=True)


@contextmanager
def temporary_model_download(config_path: str, *, tokenizer_only: bool = False) -> Iterator[str]:
    """Download into one process-scoped directory and remove it on exit."""
    config = load_config(config_path)
    model = config["model"]
    repository = str(model["checkpoint"])
    if Path(repository).is_dir():
        yield str(Path(repository).resolve())
        return
    review = config.get("review_experiments", {})
    workers = max(1, int(review.get("temporary_download_max_workers", 8)))
    timeout = max(10, int(review.get("temporary_download_timeout_seconds", 120)))
    # These directories all live under TemporaryDirectory, including the Hub
    # cache. Nothing is written to ~/.cache/huggingface by this mode.
    with tempfile.TemporaryDirectory(prefix="response-geometry-model-") as temporary:
        temporary_root = Path(temporary)
        model_dir = temporary_root / "model"
        cache_dir = temporary_root / "hub-cache"
        temporary_environment = {
            "HF_HUB_CACHE": str(cache_dir),
            "HUGGINGFACE_HUB_CACHE": str(cache_dir),
            "HF_XET_CACHE": str(temporary_root / "xet-cache"),
            "HF_ASSETS_CACHE": str(temporary_root / "assets-cache"),
            "TRANSFORMERS_CACHE": str(temporary_root / "transformers-cache"),
            "HF_HUB_DOWNLOAD_TIMEOUT": str(timeout),
            "HF_HUB_ETAG_TIMEOUT": str(min(timeout, 30)),
        }
        previous_environment = {name: os.environ.get(name) for name in temporary_environment}
        os.environ.update(temporary_environment)
        try:
            try:
                from huggingface_hub import snapshot_download
            except ImportError as exc:
                raise ImportError("huggingface_hub is required for --temporary-model-download") from exc
            if tokenizer_only:
                allow_patterns = [
                    "config.json", "tokenizer.json", "tokenizer_config.json",
                    "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt",
                ]
            else:
                # Skip repository prose and unrelated export formats. Include
                # both common weight formats so the mode remains checkpoint-safe.
                allow_patterns = [
                    "*.json", "*.safetensors", "*.bin", "*.model",
                    "*.tiktoken", "*.txt", "*.py",
                ]
            kind = "tokenizer files" if tokenizer_only else "model and tokenizer"
            print(
                f"[review_download] START repository={repository!r} kind={kind} "
                f"temporary_root={temporary_root} workers={workers}",
                flush=True,
            )
            snapshot_download(
                repo_id=repository,
                revision=model.get("revision", "main"),
                local_dir=str(model_dir),
                cache_dir=str(cache_dir),
                max_workers=workers,
                etag_timeout=min(timeout, 30),
                allow_patterns=allow_patterns,
                ignore_patterns=["*.md", ".gitattributes"],
            )
            if not (model_dir / "config.json").is_file():
                raise RuntimeError(f"temporary model download is incomplete: {model_dir / 'config.json'}")
            print(f"[review_download] DONE path={model_dir}", flush=True)
            yield str(model_dir)
        finally:
            print(f"[review_download] CLEANUP path={temporary_root}", flush=True)
            for name, previous in previous_environment.items():
                if previous is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = previous


def _run(module, config_path: str, model_path: str | None, force: bool) -> None:
    previous = sys.argv
    arguments = [module.__name__, "--config", config_path]
    if model_path and module in MODEL_STAGES:
        arguments.extend(("--model-path", model_path))
    if force:
        arguments.append("--force")
    sys.argv = arguments
    started = time.monotonic()
    name = module.__name__.split(".")[-1]
    print(f"[review_pipeline] START {name}", flush=True)
    try:
        module.main()
        print(f"[review_pipeline] DONE {name} elapsed={(time.monotonic()-started)/60:.1f}m", flush=True)
    finally:
        sys.argv = previous


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    model_source = parser.add_mutually_exclusive_group()
    model_source.add_argument("--model-path")
    model_source.add_argument(
        "--temporary-model-download",
        action="store_true",
        help="download the configured model once under /tmp and delete it when this pipeline exits",
    )
    parser.add_argument("--from-stage", choices=[module.__name__.split(".")[-1] for module in STAGES])
    parser.add_argument("--through-stage", choices=[module.__name__.split(".")[-1] for module in STAGES])
    parser.add_argument("--skip-jacobian", action="store_true")
    experiment_scale = parser.add_mutually_exclusive_group()
    experiment_scale.add_argument(
        "--quick-check", action="store_true",
        help="run a small isolated pilot; skip Jacobian and reduce contexts, candidates, ranks, and bootstraps",
    )
    experiment_scale.add_argument(
        "--formal-check", action="store_true",
        help="run an isolated medium-scale reviewer audit with equalized transfer fits and a small Jacobian check",
    )
    parser.add_argument("--skip-model-forward", action="store_true", help="run only analyses whose extracted inputs already exist")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.quick_check:
        config_context = quick_check_config(args.config)
    elif args.formal_check:
        config_context = formal_check_config(args.config)
    else:
        config_context = nullcontext(args.config)
    with config_context as resolved_config:
        skip_jacobian = args.skip_jacobian or args.quick_check
        active = [module for module in STAGES if not (skip_jacobian and module in {extract_jacobian_responses, analyze_jacobian_alignment})]
        if args.skip_model_forward:
            active = [module for module in active if module not in FORWARD_STAGES]
        names = [module.__name__.split(".")[-1] for module in active]
        start = names.index(args.from_stage) if args.from_stage else 0
        stop = names.index(args.through_stage) + 1 if args.through_stage else len(active)
        selected = active[start:stop]
        needs_model = any(module in MODEL_STAGES for module in selected)
        needs_weights = any(module in FORWARD_STAGES for module in selected)
        download = (
            temporary_model_download(resolved_config, tokenizer_only=not needs_weights)
            if args.temporary_model_download and needs_model
            else nullcontext(args.model_path)
        )
        with download as resolved_model_path:
            for module in selected:
                _run(module, resolved_config, resolved_model_path, args.force)
        if args.quick_check or args.formal_check:
            print_check_summary(resolved_config, formal=args.formal_check)


if __name__ == "__main__":
    main()
