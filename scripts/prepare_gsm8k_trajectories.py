from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from prefix_displacement.config import load_json_config
from prefix_displacement.model_loading import explain_model_load_failure, resolve_model_source
from prefix_displacement.runtime import (
    batch_size_metadata,
    gpu_memory_snapshot,
    prepare_data_parallel,
    resolve_precision,
    seed_everything,
    write_json_exclusive,
)
from prefix_displacement.schema import require_torch
from prefix_displacement.trajectory_generation import (
    GreedyNextTokenForward,
    extract_generated_answer,
    extract_reference_answer,
    first_distinct_answer_tokens,
    is_correct_answer,
    manual_greedy_decode,
    stable_problem_id,
)


def collate_prompts(rows, tokenizer):
    torch = require_torch()
    encoded = [tokenizer(row["prompt"], add_special_tokens=False)["input_ids"] for row in rows]
    max_length = max(map(len, encoded))
    input_ids = torch.full((len(rows), max_length), tokenizer.pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    # Left padding makes the last column the next-token position for every row.
    for index, token_ids in enumerate(encoded):
        input_ids[index, -len(token_ids):] = torch.tensor(token_ids, dtype=torch.long)
        attention_mask[index, -len(token_ids):] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask, "prompt_ids": encoded, "rows": rows}


def cached_greedy_decode(backbone, input_ids, attention_mask, *, max_new_tokens, eos_token_id, pad_token_id):
    """Deterministic greedy decoding with the checkpoint's KV cache."""
    torch = require_torch()
    with torch.no_grad():
        completed = backbone.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            num_beams=1,
            max_new_tokens=int(max_new_tokens),
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            use_cache=True,
        )
    generated = completed[:, input_ids.shape[1]:].tolist()
    cleaned = []
    for tokens in generated:
        values = []
        for token in tokens:
            if eos_token_id is not None and int(token) == int(eos_token_id):
                break
            values.append(int(token))
        cleaned.append(values)
    return completed, cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description="Download GSM8K and generate Qwen trajectories without generate().")
    parser.add_argument("--config", default="configs/trajectory_generation.json")
    parser.add_argument("--model-path", default=None)
    args = parser.parse_args()
    torch = require_torch()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    try:
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "The existing server environment must provide datasets and transformers; "
            "no installation was attempted"
        ) from exc

    config = load_json_config(args.config)
    model_config = load_json_config(config["model_config"])
    seed = int(model_config["project"]["seed"])
    seed_everything(seed)
    model_spec = model_config["model"]
    model_source, loading_kwargs = resolve_model_source(model_spec, args.model_path)
    precision_name, dtype = resolve_precision(config["runtime"].get("precision", "auto"))
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_source, **loading_kwargs)
        backbone = AutoModelForCausalLM.from_pretrained(
            model_source,
            dtype=dtype,
            attn_implementation=model_spec["attention_implementation"],
            **loading_kwargs,
        ).eval()
    except OSError as exc:
        raise explain_model_load_failure(model_source, exc) from exc
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    backbone.requires_grad_(False)
    generation_config = config["generation"]
    cached_generation = bool(generation_config.get("use_kv_cache", False))
    if cached_generation:
        device = torch.device("cuda:0"); backbone.to(device); device_ids = [0]; generation_model = None
    else:
        generation_model = GreedyNextTokenForward.build(backbone)
        generation_model, device, device_ids = prepare_data_parallel(generation_model)

    dataset_config = config["dataset"]
    dataset = load_dataset(
        dataset_config["name"],
        dataset_config["subset"],
        split=dataset_config["source_split"],
        revision=dataset_config.get("revision", "main"),
    )
    maximum = dataset_config.get("max_problems")
    if maximum is not None:
        dataset = dataset.select(range(min(int(maximum), len(dataset))))
    prompt_template = config["prompt"]["template"]
    rows = [
        {
            "source_index": index,
            "question": item["question"],
            "reference_answer_text": item["answer"],
            "prompt": prompt_template.format(question=item["question"]),
        }
        for index, item in enumerate(dataset)
    ]
    output_path = Path(config["output"]["trajectories_jsonl"])
    metadata_path = Path(config["output"]["metadata_json"])
    resume_count = 0
    existing_correct = 0
    if metadata_path.exists():
        raise FileExistsError(
            f"Completed trajectory metadata already exists: {metadata_path}. "
            "Use a new output path for a new immutable run."
        )
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as existing:
            for line_number, line in enumerate(existing, start=1):
                if not line.strip():
                    continue
                try:
                    prior_row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"Partial trajectory file has invalid JSON at line {line_number}; "
                        "refusing to guess a resume point"
                    ) from exc
                resume_count += 1
                existing_correct += int(bool(prior_row.get("correctness")))
        if resume_count > len(rows):
            raise RuntimeError("Partial output has more rows than the configured dataset")
        rows = rows[resume_count:]
    sizes = batch_size_metadata(int(config["runtime"]["per_device_batch_size"]))
    if cached_generation:
        sizes["global_batch_size"] = int(config["runtime"]["per_device_batch_size"])
    loader = torch.utils.data.DataLoader(
        rows,
        batch_size=sizes["global_batch_size"],
        shuffle=False,
        num_workers=0,
        collate_fn=functools.partial(collate_prompts, tokenizer=tokenizer),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    memory_path = Path(config["output"]["gpu_memory_jsonl"])
    written = resume_count
    correct = existing_correct
    output_mode = "a" if output_path.exists() else "x"
    memory_mode = "a" if memory_path.exists() else "x"
    remaining = len(rows); total_expected = resume_count + remaining; started = time.monotonic()
    print(f"[trajectory_generation] backend={'cached-greedy' if cached_generation else 'data-parallel-no-cache'} resume={resume_count} remaining={remaining} batch={sizes['global_batch_size']}",flush=True)
    try:
        with output_path.open(output_mode, encoding="utf-8") as output, memory_path.open(memory_mode, encoding="utf-8") as memory:
            for batch_index, batch in enumerate(loader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                if cached_generation:
                    _final_ids, generated_batches = cached_greedy_decode(backbone,input_ids,attention_mask,max_new_tokens=int(generation_config["max_new_tokens"]),eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id)
                else:
                    _final_ids, generated_batches = manual_greedy_decode(generation_model,input_ids,attention_mask,max_new_tokens=int(generation_config["max_new_tokens"]),eos_token_id=tokenizer.eos_token_id,stop_on_eos=bool(generation_config["stop_on_eos"]))
                for offset, (source, prompt_ids, generated_ids) in enumerate(
                    zip(batch["rows"], batch["prompt_ids"], generated_batches)
                ):
                    if not generated_ids:
                        raise RuntimeError(f"Empty generation for source index {source['source_index']}")
                    reference = extract_reference_answer(source["reference_answer_text"])
                    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                    generated_answer = extract_generated_answer(generated_text)
                    correctness = is_correct_answer(generated_answer, reference)
                    (
                        answer_shared_prefix_ids,
                        positive_id,
                        negative_id,
                        negative_text,
                    ) = first_distinct_answer_tokens(tokenizer, reference)
                    cue_ids = tokenizer(
                        generation_config["answer_cue"], add_special_tokens=False
                    )["input_ids"]
                    full_ids = (
                        list(prompt_ids)
                        + list(generated_ids)
                        + list(cue_ids)
                        + list(answer_shared_prefix_ids)
                    )
                    first_departure = len(prompt_ids) - 1
                    transition_positions = list(
                        range(first_departure, first_departure + len(generated_ids))
                    )
                    row = {
                        "problem_id": stable_problem_id(
                            dataset_config["source_split"], source["source_index"], source["question"]
                        ),
                        "trajectory_id": f"gsm8k-{dataset_config['source_split']}-{source['source_index']:05d}-greedy-0",
                        "input_ids": full_ids,
                        "transition_positions": transition_positions,
                        "generated_start_position": first_departure,
                        "evaluation_position": len(full_ids) - 1,
                        "positive_token_id": positive_id,
                        "negative_token_id": negative_id,
                        "correctness": correctness,
                        "reference_answer": reference,
                        "generated_answer": generated_answer,
                        "negative_answer_candidate": negative_text,
                        "margin_definition": (
                            "first_divergent_correct_vs_numeric_distractor_token_"
                            "after_teacher_forced_shared_prefix"
                            if negative_text != "VOCABULARY_NON_SPECIAL_FALLBACK"
                            else
                            "correct_first_answer_token_vs_deterministic_non_special_"
                            "vocabulary_fallback"
                        ),
                        "answer_shared_prefix_ids": answer_shared_prefix_ids,
                        "generation_strategy": "greedy_manual_forward",
                    }
                    output.write(json.dumps(row, sort_keys=True) + "\n")
                    output.flush()
                    written += 1
                    correct += int(correctness)
                memory.write(json.dumps(gpu_memory_snapshot(batch_index), sort_keys=True) + "\n")
                memory.flush()
                elapsed=max(time.monotonic()-started,1e-9); completed=written-resume_count; rate=completed/elapsed; eta=(remaining-completed)/rate if rate>0 else float("inf")
                print(f"[trajectory_generation] {written}/{total_expected} problems rate={rate:.3f}/s eta={eta/60:.1f}m",flush=True)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise RuntimeError(
                "CUDA OOM; batch size, max_new_tokens, sequence length, and data count "
                "were not changed automatically"
            ) from exc
        raise
    write_json_exclusive(
        metadata_path,
        {
            "config": config,
            "model_source": model_source,
            "resolved_revision": getattr(backbone.config, "_commit_hash", None)
            or "UNKNOWN",
            "precision": precision_name,
            "device_ids": device_ids,
            **sizes,
            "num_trajectories": written,
            "num_correct": correct,
            "resumed_from_rows": resume_count,
            "accuracy": correct / max(written, 1),
            "margin_definition": (
                "row-level first-divergent correct-vs-numeric distractor; "
                "deterministic non-special vocabulary fallback when numeric "
                "candidates do not diverge"
            ),
        },
    )
    print(output_path)


if __name__ == "__main__":
    main()
