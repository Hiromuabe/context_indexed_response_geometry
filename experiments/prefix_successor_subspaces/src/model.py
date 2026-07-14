from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from prefix_displacement.extraction import resolve_decoder_layers
from prefix_displacement.model_loading import (
    explain_model_load_failure,
    resolve_model_source,
)
from prefix_displacement.runtime import prepare_data_parallel, resolve_precision
from prefix_displacement.schema import require_torch

from .hooks import make_replica_local_layer_controller


def _decoder_module(backbone: Any) -> Any:
    """Return the decoder without the full-sequence causal-LM vocabulary head."""

    getter = getattr(backbone, "get_decoder", None)
    if callable(getter):
        try:
            decoder = getter()
        except (AttributeError, NotImplementedError):
            decoder = None
        if decoder is not None:
            return decoder
    for attribute in ("model", "transformer"):
        decoder = getattr(backbone, attribute, None)
        if decoder is not None:
            return decoder
    raise AttributeError("Causal LM does not expose get_decoder(), model, or transformer")


def _decoder_hidden(
    backbone: Any, *, input_ids: Any, attention_mask: Any
) -> Any:
    output = _decoder_module(backbone)(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        output_hidden_states=False,
        return_dict=True,
    )
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is None and isinstance(output, (tuple, list)) and output:
        hidden = output[0]
    if hidden is None:
        raise RuntimeError("Decoder did not return last_hidden_state")
    return hidden


def _endpoint_logits(backbone: Any, final_hidden: Any, endpoint_positions: Any) -> Any:
    torch = require_torch()
    output_embeddings = backbone.get_output_embeddings()
    if output_embeddings is None:
        raise RuntimeError("Causal LM does not expose output embeddings")
    endpoint_positions = endpoint_positions.to(
        device=final_hidden.device, dtype=torch.long
    )
    batch = torch.arange(final_hidden.shape[0], device=final_hidden.device)
    endpoint_hidden = final_hidden[batch, endpoint_positions]
    # Materialize [B,V], never the prohibitively large [B,S,V] tensor.
    return output_embeddings(endpoint_hidden).float()


@dataclass
class LoadedForwardModel:
    model: Any
    tokenizer: Any
    device: Any
    device_ids: list[int]
    precision_name: str
    precision_dtype: Any
    model_source: str
    resolved_revision: str | None
    hidden_size: int
    num_layers: int
    layer_ids: list[int]

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "model_source": self.model_source,
            "resolved_revision": self.resolved_revision or "UNKNOWN",
            "precision": self.precision_name,
            "hidden_size": self.hidden_size,
            "num_decoder_layers": self.num_layers,
            "layer_ids": self.layer_ids,
            "device_ids": self.device_ids,
            "primary_device": str(self.device),
            "parallelism": (
                "torch.nn.DataParallel" if len(self.device_ids) > 1 else "single_gpu"
            ),
        }


class NextTokenLogitsForward:
    """Factory for a DataParallel-safe endpoint-logit wrapper."""

    @staticmethod
    def build(backbone: Any) -> Any:
        torch = require_torch()

        class _NextTokenLogitsForward(torch.nn.Module):
            def __init__(self, model: Any) -> None:
                super().__init__()
                self.backbone = model

            def forward(
                self,
                input_ids: Any,
                attention_mask: Any,
                endpoint_positions: Any,
                sample_index: Any,
            ) -> tuple[Any, Any]:
                final_hidden = _decoder_hidden(
                    self.backbone,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                return (
                    _endpoint_logits(self.backbone, final_hidden, endpoint_positions),
                    sample_index,
                )

        return _NextTokenLogitsForward(backbone)


class MultiLayerEndpointForward:
    """Factory extracting raw decoder-block outputs at exact endpoint positions.

    Target blocks are wrapped before DataParallel replication.  Each replica then
    owns its capture state, so only endpoint vectors (never full activations) are
    gathered onto cuda:0 and batch order follows DataParallel's normal gather.
    """

    @staticmethod
    def build(backbone: Any, layer_ids: Sequence[int]) -> Any:
        torch = require_torch()
        requested = tuple(map(int, layer_ids))
        if len(set(requested)) != len(requested):
            raise ValueError("Target layer IDs must be unique")
        decoder_layers = resolve_decoder_layers(backbone)
        for layer_id in requested:
            if not 0 <= layer_id < len(decoder_layers):
                raise ValueError(
                    f"Layer {layer_id} outside decoder depth {len(decoder_layers)}"
                )
            decoder_layers[layer_id] = make_replica_local_layer_controller(
                decoder_layers[layer_id]
            )

        class _MultiLayerEndpointForward(torch.nn.Module):
            def __init__(self, model: Any, requested_layers: Sequence[int]) -> None:
                super().__init__()
                self.backbone = model
                self.layer_ids = tuple(map(int, requested_layers))

            def forward(
                self,
                input_ids: Any,
                attention_mask: Any,
                endpoint_positions: Any,
                sample_index: Any,
                candidate_token_ids: Any | None = None,
            ) -> tuple[Any, ...]:
                layers = resolve_decoder_layers(self.backbone)
                controllers = [layers[index] for index in self.layer_ids]
                for controller in controllers:
                    controller.capture_at(endpoint_positions)
                try:
                    final_hidden = _decoder_hidden(
                        self.backbone,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    endpoints = torch.stack(
                        [controller.take_captured() for controller in controllers],
                        dim=1,
                    )
                finally:
                    for controller in controllers:
                        controller.clear()
                if candidate_token_ids is None:
                    return endpoints.detach(), sample_index.detach()

                batch = torch.arange(input_ids.shape[0], device=input_ids.device)
                logits = _endpoint_logits(
                    self.backbone, final_hidden, endpoint_positions
                )
                if candidate_token_ids.ndim != 2 or candidate_token_ids.shape[0] != len(batch):
                    raise ValueError("candidate_token_ids must have shape [batch, candidates]")
                normalizer = torch.logsumexp(logits, dim=-1)
                selected = logits.gather(1, candidate_token_ids)
                candidate_logprob = selected - normalizer[:, None]
                probabilities = torch.softmax(logits, dim=-1)
                entropy = normalizer - (probabilities * logits).sum(dim=-1)
                return (
                    endpoints.detach(),
                    candidate_logprob.detach(),
                    entropy.detach(),
                    sample_index.detach(),
                )

        return _MultiLayerEndpointForward(backbone, requested)


def resolve_target_layers(model_config: Mapping[str, Any], num_layers: int) -> list[int]:
    """Resolve zero-based block IDs from explicit IDs or normalized depths."""

    if bool(model_config.get("all_layers", False)):
        return list(range(num_layers))
    raw = model_config.get("target_layers", model_config.get("layers"))
    if raw is None and "layer" in model_config:
        raw = [model_config["layer"]]
    if raw is None:
        raw = model_config.get(
            "normalized_depths",
            model_config.get("layer_fractions", [0.25, 0.50, 0.75, 1.0]),
        )
    if not isinstance(raw, list) or not raw:
        raise ValueError("model target_layers/layer_fractions must be a non-empty list")
    resolved: list[int] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Invalid target layer specification: {value!r}")
        if isinstance(value, float):
            if not 0 < value <= 1:
                raise ValueError("Normalized layer depths must lie in (0, 1]")
            layer_id = math.ceil(value * num_layers) - 1
        else:
            layer_id = int(value)
        if not 0 <= layer_id < num_layers:
            raise ValueError(f"Layer {layer_id} outside decoder depth {num_layers}")
        if layer_id not in resolved:
            resolved.append(layer_id)
    additional = model_config.get("additional_target_layers", [])
    if not isinstance(additional, list):
        raise ValueError("model.additional_target_layers must be a list")
    for value in additional:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                "model.additional_target_layers must contain zero-based integers"
            )
        if not 0 <= value < num_layers:
            raise ValueError(f"Layer {value} outside decoder depth {num_layers}")
        if value not in resolved:
            resolved.append(value)
    return resolved


def _load_backbone_and_tokenizer(
    config: Mapping[str, Any], model_path: str | None = None
) -> tuple[Any, Any, str, str, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required in the existing GPU environment; no package "
            "installation was attempted"
        ) from exc
    model_config = config.get("model")
    if not isinstance(model_config, Mapping):
        raise ValueError("Missing model config mapping")
    source, loading_kwargs = resolve_model_source(model_config, model_path)
    requested_precision = str(
        model_config.get("precision", model_config.get("dtype", "auto"))
    )
    precision_name, precision_dtype = resolve_precision(requested_precision)
    try:
        tokenizer = AutoTokenizer.from_pretrained(source, **loading_kwargs)
        backbone = AutoModelForCausalLM.from_pretrained(
            source,
            dtype=precision_dtype,
            attn_implementation=model_config.get("attention_implementation", "sdpa"),
            **loading_kwargs,
        )
    except OSError as exc:
        raise explain_model_load_failure(source, exc) from exc
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer has neither pad_token_id nor eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    backbone.eval()
    backbone.requires_grad_(False)
    return backbone, tokenizer, source, precision_name, precision_dtype


def load_next_token_model(
    config: Mapping[str, Any], model_path: str | None = None
) -> LoadedForwardModel:
    backbone, tokenizer, source, precision_name, precision_dtype = (
        _load_backbone_and_tokenizer(config, model_path)
    )
    layers = resolve_decoder_layers(backbone)
    hidden_size = int(backbone.config.hidden_size)
    revision = getattr(backbone.config, "_commit_hash", None)
    wrapped = NextTokenLogitsForward.build(backbone)
    model, device, device_ids = prepare_data_parallel(wrapped)
    model.eval()
    return LoadedForwardModel(
        model=model,
        tokenizer=tokenizer,
        device=device,
        device_ids=device_ids,
        precision_name=precision_name,
        precision_dtype=precision_dtype,
        model_source=source,
        resolved_revision=revision,
        hidden_size=hidden_size,
        num_layers=len(layers),
        layer_ids=[],
    )


def load_endpoint_model(
    config: Mapping[str, Any], model_path: str | None = None
) -> LoadedForwardModel:
    backbone, tokenizer, source, precision_name, precision_dtype = (
        _load_backbone_and_tokenizer(config, model_path)
    )
    decoder_layers = resolve_decoder_layers(backbone)
    model_config = config["model"]
    layer_ids = resolve_target_layers(model_config, len(decoder_layers))
    hidden_size = int(backbone.config.hidden_size)
    revision = getattr(backbone.config, "_commit_hash", None)
    wrapped = MultiLayerEndpointForward.build(backbone, layer_ids)
    model, device, device_ids = prepare_data_parallel(wrapped)
    model.eval()
    return LoadedForwardModel(
        model=model,
        tokenizer=tokenizer,
        device=device,
        device_ids=device_ids,
        precision_name=precision_name,
        precision_dtype=precision_dtype,
        model_source=source,
        resolved_revision=revision,
        hidden_size=hidden_size,
        num_layers=len(decoder_layers),
        layer_ids=layer_ids,
    )


def assert_output_order(expected: Any, observed: Any) -> None:
    torch = require_torch()
    try:
        torch.testing.assert_close(expected.cpu(), observed.cpu(), atol=0, rtol=0)
    except AssertionError as exc:
        raise RuntimeError(
            "DataParallel gathered endpoint outputs in an unexpected batch order"
        ) from exc
