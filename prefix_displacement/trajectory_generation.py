from __future__ import annotations

import hashlib
import re
import unicodedata
from decimal import Decimal, InvalidOperation
from typing import Any

from .schema import require_torch


FINAL_ANSWER_PATTERN = re.compile(r"####\s*([^\n]+)")
NUMBER_PATTERN = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def stable_problem_id(source_split: str, index: int, question: str) -> str:
    digest = hashlib.sha256(question.encode("utf-8")).hexdigest()[:12]
    return f"gsm8k-{source_split}-{index:05d}-{digest}"


def extract_reference_answer(answer_text: str) -> str:
    matches = FINAL_ANSWER_PATTERN.findall(answer_text)
    if not matches:
        raise ValueError("GSM8K answer does not contain a #### final answer marker")
    return matches[-1].strip().replace(",", "")


def extract_generated_answer(text: str) -> str | None:
    matches = NUMBER_PATTERN.findall(text)
    return matches[-1].replace(",", "") if matches else None


def normalized_number(value: str | None) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(value.strip())
    except (InvalidOperation, ValueError):
        return None


def is_correct_answer(generated: str | None, reference: str) -> bool:
    generated_value = normalized_number(generated)
    reference_value = normalized_number(reference)
    return generated_value is not None and reference_value is not None and generated_value == reference_value


def first_distinct_answer_tokens(
    tokenizer: Any, reference: str
) -> tuple[list[int], int, int, str]:
    """Find the first token divergence after a teacher-forced shared prefix."""
    positive_ids = tokenizer(" " + reference, add_special_tokens=False)["input_ids"]
    if not positive_ids:
        raise ValueError("Reference answer produced no tokens")
    positive = int(positive_ids[0])
    reference_value = normalized_number(reference)
    candidates = []
    if reference_value is not None:
        candidates.extend((str(reference_value + 1), str(reference_value - 1)))
    candidates.extend(("0", "1", "2", "10"))
    for candidate in candidates:
        negative_ids = tokenizer(" " + candidate, add_special_tokens=False)["input_ids"]
        if not negative_ids:
            continue
        shared = 0
        while (
            shared < len(positive_ids)
            and shared < len(negative_ids)
            and int(positive_ids[shared]) == int(negative_ids[shared])
        ):
            shared += 1
        if shared < len(positive_ids) and shared < len(negative_ids):
            return (
                [int(value) for value in positive_ids[:shared]],
                int(positive_ids[shared]),
                int(negative_ids[shared]),
                candidate,
            )
    # Defensive vocabulary-level fallback. Search deterministically for a
    # non-special, decodable token instead of assuming the adjacent vocabulary
    # ID is a valid model input.
    special_ids = set(map(int, getattr(tokenizer, "all_special_ids", [])))
    for offset in range(1, len(tokenizer)):
        negative = (positive + offset) % len(tokenizer)
        if negative == positive or negative in special_ids:
            continue
        decoded = tokenizer.decode(
            [negative],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if not decoded or "\ufffd" in decoded or "\x00" in decoded:
            continue
        if any(
            unicodedata.category(character) in {"Cc", "Cs"}
            and character not in "\n\t"
            for character in decoded
        ):
            continue
        return [], positive, negative, "VOCABULARY_NON_SPECIAL_FALLBACK"
    raise ValueError("Could not construct a valid distinct answer-token distractor")


class GreedyNextTokenForward:
    @staticmethod
    def build(backbone: Any) -> Any:
        torch = require_torch()

        class _GreedyNextTokenForward(torch.nn.Module):
            def __init__(self, model: Any) -> None:
                super().__init__()
                self.backbone = model

            def forward(self, input_ids: Any, attention_mask: Any) -> Any:
                outputs = self.backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
                return outputs.logits[:, -1, :].float().argmax(dim=-1)

        return _GreedyNextTokenForward(backbone)


def manual_greedy_decode(
    model: Any,
    input_ids: Any,
    attention_mask: Any,
    *,
    max_new_tokens: int,
    eos_token_id: int | None,
    stop_on_eos: bool,
) -> tuple[Any, list[list[int]]]:
    torch = require_torch()
    generated: list[list[int]] = [[] for _ in range(input_ids.shape[0])]
    finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
    for _step in range(max_new_tokens):
        with torch.no_grad():
            next_token = model(input_ids=input_ids, attention_mask=attention_mask)
        for index, token in enumerate(next_token.tolist()):
            if not bool(finished[index]):
                generated[index].append(int(token))
        if stop_on_eos and eos_token_id is not None:
            finished |= next_token.eq(eos_token_id)
        append_token = next_token.clone()
        if eos_token_id is not None:
            append_token = torch.where(finished, torch.full_like(append_token, eos_token_id), append_token)
        input_ids = torch.cat((input_ids, append_token[:, None]), dim=1)
        attention_mask = torch.cat(
            (attention_mask, (~finished).to(attention_mask.dtype)[:, None]), dim=1
        )
        if bool(finished.all()):
            break
    if eos_token_id is not None:
        generated = [
            tokens[:-1] if tokens and tokens[-1] == eos_token_id else tokens
            for tokens in generated
        ]
    return input_ids, generated
