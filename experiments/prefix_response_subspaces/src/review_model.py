from __future__ import annotations

from typing import Any, Sequence

from prefix_displacement.extraction import resolve_decoder_layers
from prefix_displacement.runtime import prepare_data_parallel
from prefix_displacement.schema import require_torch
from experiments.prefix_successor_subspaces.src.hooks import make_replica_local_layer_controller
from experiments.prefix_successor_subspaces.src.model import (
    LoadedForwardModel,
    _decoder_hidden,
    _load_backbone_and_tokenizer,
    resolve_target_layers,
)


class _EndpointComplete(RuntimeError):
    """Replica-local control flow after the deepest requested block finishes."""


def _early_exit_layer(controller: Any) -> Any:
    torch = require_torch()

    class _EarlyExitLayer(torch.nn.Module):
        def __init__(self, inner: Any) -> None:
            super().__init__()
            self.controller = inner

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            self.controller(*args, **kwargs)
            raise _EndpointComplete

    return _EarlyExitLayer(controller)


class EarlyExitEndpointForward:
    """Capture requested block outputs and skip every deeper decoder block."""

    @staticmethod
    def build(backbone: Any, layer_ids: Sequence[int]) -> Any:
        torch = require_torch()
        requested = tuple(map(int, layer_ids))
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("requested layer IDs must be non-empty and unique")
        layers = resolve_decoder_layers(backbone)
        controllers = {}
        for layer_id in requested:
            if not 0 <= layer_id < len(layers):
                raise ValueError(f"layer {layer_id} outside decoder depth {len(layers)}")
            controller = make_replica_local_layer_controller(layers[layer_id])
            controllers[layer_id] = controller
            layers[layer_id] = controller
        deepest = max(requested)
        layers[deepest] = _early_exit_layer(controllers[deepest])

        class _EarlyExitEndpointForward(torch.nn.Module):
            def __init__(self, model: Any) -> None:
                super().__init__()
                self.backbone = model
                self.layer_ids = requested
                self.deepest_layer = deepest

            def _controllers(self) -> list[Any]:
                decoder_layers = resolve_decoder_layers(self.backbone)
                result = []
                for layer_id in self.layer_ids:
                    layer = decoder_layers[layer_id]
                    result.append(layer.controller if layer_id == self.deepest_layer else layer)
                return result

            def forward(self, input_ids: Any, attention_mask: Any, endpoint_positions: Any, sample_index: Any):
                active = self._controllers()
                for controller in active:
                    controller.capture_at(endpoint_positions)
                try:
                    try:
                        _decoder_hidden(self.backbone, input_ids=input_ids, attention_mask=attention_mask)
                    except _EndpointComplete:
                        pass
                    endpoints = torch.stack([controller.take_captured() for controller in active], dim=1)
                finally:
                    for controller in active:
                        controller.clear()
                return endpoints.detach(), sample_index.detach()

        return _EarlyExitEndpointForward(backbone)


def load_early_exit_endpoint_model(config: dict[str, Any], model_path: str | None = None) -> LoadedForwardModel:
    backbone, tokenizer, source, precision_name, precision_dtype = _load_backbone_and_tokenizer(config, model_path)
    decoder_layers = resolve_decoder_layers(backbone)
    layer_ids = resolve_target_layers(config["model"], len(decoder_layers))
    hidden_size = int(backbone.config.hidden_size)
    revision = getattr(backbone.config, "_commit_hash", None)
    wrapped = EarlyExitEndpointForward.build(backbone, layer_ids)
    model, device, device_ids = prepare_data_parallel(wrapped)
    model.eval()
    return LoadedForwardModel(
        model=model, tokenizer=tokenizer, device=device, device_ids=device_ids,
        precision_name=precision_name, precision_dtype=precision_dtype,
        model_source=source, resolved_revision=revision, hidden_size=hidden_size,
        num_layers=len(decoder_layers), layer_ids=layer_ids,
    )
