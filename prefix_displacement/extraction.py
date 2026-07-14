from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .schema import require_torch


class TrajectoryFormatError(ValueError):
    """Raised when an extraction input would require a scientific guess."""


def resolve_decoder_layers(backbone: Any) -> Any:
    candidates = (
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    )
    for parent_name, layers_name in candidates:
        parent = getattr(backbone, parent_name, None)
        layers = getattr(parent, layers_name, None) if parent is not None else None
        if layers is not None:
            return layers
    raise AttributeError(
        "Could not locate decoder blocks; expected model.layers, transformer.h, "
        "or gpt_neox.layers"
    )


class HiddenGradientForward:
    """Factory namespace; call ``build`` after PyTorch is importable."""

    @staticmethod
    def build(backbone: Any, layer_index: int) -> Any:
        torch = require_torch()

        class _HiddenGradientForward(torch.nn.Module):
            def __init__(self, model: Any, index: int) -> None:
                super().__init__()
                self.backbone = model
                self.layer_index = index

            def forward(
                self,
                input_ids: Any,
                attention_mask: Any,
                evaluation_position: Any,
                positive_token_id: Any,
                negative_token_id: Any,
                transition_positions: Any,
                sample_index: Any,
            ) -> tuple[Any, Any, Any, Any, Any, Any]:
                layers = resolve_decoder_layers(self.backbone)
                if not 0 <= self.layer_index < len(layers):
                    raise IndexError(
                        f"layer_index={self.layer_index} outside [0, {len(layers)})"
                    )
                captured: dict[str, Any] = {}

                def capture_hook(_module: Any, _inputs: Any, output: Any):
                    hidden = output[0] if isinstance(output, tuple) else output
                    # Cut the graph below the target layer. Only this leaf needs a gradient.
                    hidden_leaf = hidden.detach().requires_grad_(True)
                    hidden_leaf.retain_grad()
                    captured["hidden"] = hidden_leaf
                    if isinstance(output, tuple):
                        return (hidden_leaf, *output[1:])
                    return hidden_leaf

                handle = layers[self.layer_index].register_forward_hook(capture_hook)
                try:
                    outputs = self.backbone(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True,
                    )
                finally:
                    handle.remove()
                hidden = captured.get("hidden")
                if hidden is None:
                    raise RuntimeError("Target layer hook did not run")

                batch = torch.arange(input_ids.shape[0], device=input_ids.device)
                eval_logits = outputs.logits[batch, evaluation_position].float()
                positive = eval_logits.gather(1, positive_token_id[:, None]).squeeze(1)
                negative = eval_logits.gather(1, negative_token_id[:, None]).squeeze(1)
                margin = positive - negative
                gradient = torch.autograd.grad(
                    margin.sum(), hidden, retain_graph=False, create_graph=False
                )[0]

                valid_positions = transition_positions.clamp_min(0)
                batch_index = batch[:, None].expand_as(valid_positions)
                departure = hidden[batch_index, valid_positions]
                arrival = hidden[batch_index, valid_positions + 1]
                arrival_gradient = gradient[batch_index, valid_positions + 1]

                log_normalizer = torch.logsumexp(outputs.logits.float(), dim=-1)
                observed_next = input_ids[batch_index, valid_positions + 1]
                observed_logit = outputs.logits[batch_index, valid_positions, observed_next].float()
                surprisal = log_normalizer[batch_index, valid_positions] - observed_logit

                return (
                    departure.detach(),
                    arrival.detach(),
                    arrival_gradient.detach(),
                    margin.detach(),
                    surprisal.detach(),
                    sample_index.detach(),
                )

        return _HiddenGradientForward(backbone, layer_index)


class JsonlTrajectoryDataset:
    """Map-style JSONL dataset using byte offsets instead of loading all tensors."""

    def __init__(self, path: str | Path, tokenizer: Any, max_sequence_length: int) -> None:
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.offsets: list[int] = []
        self.problem_ids: list[str] = []
        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                row = json.loads(line)
                problem_id = str(row.get("problem_id", ""))
                if not problem_id:
                    raise TrajectoryFormatError(f"Missing problem_id at byte offset {offset}")
                self.offsets.append(offset)
                self.problem_ids.append(problem_id)

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        with self.path.open("rb") as handle:
            handle.seek(self.offsets[index])
            row = json.loads(handle.readline())
        return normalize_trajectory_row(
            row,
            tokenizer=self.tokenizer,
            sample_index=index,
            max_sequence_length=self.max_sequence_length,
        )


def _single_token_id(row: Mapping[str, Any], id_name: str, text_name: str, tokenizer: Any) -> int:
    if id_name in row:
        value = row[id_name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise TrajectoryFormatError(f"{id_name} must be an integer")
        return value
    text = row.get(text_name)
    if not isinstance(text, str):
        raise TrajectoryFormatError(f"Provide {id_name} or {text_name}")
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(token_ids) != 1:
        raise TrajectoryFormatError(
            f"{text_name} must tokenize to exactly one token, got {len(token_ids)}"
        )
    return int(token_ids[0])


def normalize_trajectory_row(
    row: Mapping[str, Any],
    *,
    tokenizer: Any,
    sample_index: int,
    max_sequence_length: int,
) -> dict[str, Any]:
    input_ids = row.get("input_ids")
    if input_ids is None:
        text = row.get("text")
        if not isinstance(text, str):
            raise TrajectoryFormatError("Provide input_ids or text")
        input_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not isinstance(input_ids, list) or not input_ids:
        raise TrajectoryFormatError("input_ids must be a non-empty list")
    if len(input_ids) > max_sequence_length:
        raise TrajectoryFormatError(
            f"Sequence length {len(input_ids)} exceeds configured maximum "
            f"{max_sequence_length}; automatic truncation is disabled"
        )

    positions = row.get("transition_positions")
    if not isinstance(positions, list) or not positions:
        raise TrajectoryFormatError("transition_positions must be a non-empty list")
    positions = [int(position) for position in positions]
    if positions != sorted(set(positions)):
        raise TrajectoryFormatError("transition_positions must be sorted and unique")
    if positions[0] < 0 or positions[-1] + 1 >= len(input_ids):
        raise TrajectoryFormatError("transition_positions contain an invalid adjacent pair")

    evaluation_position = row.get("evaluation_position")
    if isinstance(evaluation_position, bool) or not isinstance(evaluation_position, int):
        raise TrajectoryFormatError("evaluation_position must be an integer")
    if not 0 <= evaluation_position < len(input_ids):
        raise TrajectoryFormatError("evaluation_position is outside the input sequence")
    correctness = row.get("correctness")
    if not isinstance(correctness, bool):
        raise TrajectoryFormatError("correctness must be boolean")

    return {
        "sample_index": sample_index,
        "problem_id": str(row["problem_id"]),
        "trajectory_id": str(row["trajectory_id"]),
        "input_ids": [int(value) for value in input_ids],
        "transition_positions": positions,
        "generated_start_position": int(row.get("generated_start_position", positions[0])),
        "evaluation_position": evaluation_position,
        "positive_token_id": _single_token_id(
            row, "positive_token_id", "positive_token", tokenizer
        ),
        "negative_token_id": _single_token_id(
            row, "negative_token_id", "negative_token", tokenizer
        ),
        "correctness": correctness,
    }


def collate_trajectories(rows: Sequence[Mapping[str, Any]], pad_token_id: int) -> dict[str, Any]:
    torch = require_torch()
    batch_size = len(rows)
    max_length = max(len(row["input_ids"]) for row in rows)
    max_transitions = max(len(row["transition_positions"]) for row in rows)
    input_ids = torch.full((batch_size, max_length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
    transition_positions = torch.full(
        (batch_size, max_transitions), -1, dtype=torch.long
    )
    transition_mask = torch.zeros((batch_size, max_transitions), dtype=torch.bool)
    for index, row in enumerate(rows):
        length = len(row["input_ids"])
        count = len(row["transition_positions"])
        input_ids[index, :length] = torch.tensor(row["input_ids"], dtype=torch.long)
        attention_mask[index, :length] = 1
        transition_positions[index, :count] = torch.tensor(
            row["transition_positions"], dtype=torch.long
        )
        transition_mask[index, :count] = True
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "transition_positions": transition_positions,
        "transition_mask": transition_mask,
        "evaluation_position": torch.tensor(
            [row["evaluation_position"] for row in rows], dtype=torch.long
        ),
        "positive_token_id": torch.tensor(
            [row["positive_token_id"] for row in rows], dtype=torch.long
        ),
        "negative_token_id": torch.tensor(
            [row["negative_token_id"] for row in rows], dtype=torch.long
        ),
        "sample_index": torch.tensor(
            [row["sample_index"] for row in rows], dtype=torch.long
        ),
        "metadata": list(rows),
    }


def assert_gathered_batch_order(expected: Any, observed: Any) -> None:
    torch = require_torch()
    try:
        torch.testing.assert_close(observed.cpu(), expected.cpu(), atol=0, rtol=0)
    except AssertionError as exc:
        raise RuntimeError(
            "DataParallel output order does not match the input batch order"
        ) from exc
