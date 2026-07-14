from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterable, Iterator, Mapping, MutableMapping, Sequence


class HookContractError(RuntimeError):
    """Raised when a model block violates the expected hidden-state contract."""


def hidden_tensor_from_output(output: Any) -> Any:
    """Extract a block hidden tensor from Tensor, tuple, or list output."""

    try:
        from prefix_displacement.schema import require_torch

        torch = require_torch()
    except ImportError as exc:
        raise HookContractError("PyTorch is required for transformer hooks") from exc
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise HookContractError(
        f"Expected block output Tensor or tuple/list beginning with Tensor, got {type(output)!r}"
    )


def output_with_hidden_tensor(output: Any, hidden: Any) -> Any:
    """Replace only the first hidden tensor while preserving output container type."""

    from prefix_displacement.schema import require_torch

    torch = require_torch()
    if torch.is_tensor(output):
        return hidden
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    raise HookContractError(f"Unsupported block output type: {type(output)!r}")


def gather_sequence_positions(hidden: Any, positions: Any) -> Any:
    from prefix_displacement.schema import require_torch

    torch = require_torch()
    if hidden.ndim != 3:
        raise HookContractError("Hidden states must have shape [batch, sequence, hidden]")
    if positions.ndim != 1 or positions.shape[0] != hidden.shape[0]:
        raise HookContractError("positions must have shape [batch]")
    # DataParallel scatters tensor arguments to each replica.  Keep this helper
    # defensive as it is also used by hooks/controllers whose closure state is
    # not scattered automatically.
    positions = positions.to(device=hidden.device, dtype=torch.long)
    if bool((positions < 0).any()) or bool((positions >= hidden.shape[1]).any()):
        raise HookContractError("A requested sequence position is outside the hidden tensor")
    batch = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[batch, positions]


def first_unique_batch_rows(input_ids: Any, attention_mask: Any) -> tuple[Any, Any]:
    """Return first local row per sequence and an inverse row mapping.

    This runs inside a DataParallel replica, so uniqueness is computed only for
    that replica's local batch.  The inverse mapping restores the exact input
    order before DataParallel gathers outputs on cuda:0.
    """

    from prefix_displacement.schema import require_torch

    torch = require_torch()
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise HookContractError(
            "input_ids and attention_mask must have identical [batch, sequence] shapes"
        )
    signature = torch.cat(
        (input_ids.to(dtype=torch.long), attention_mask.to(dtype=torch.long)), dim=1
    )
    _, inverse = torch.unique(
        signature, dim=0, sorted=True, return_inverse=True
    )
    group_count = int(inverse.max().item()) + 1 if len(inverse) else 0
    first = torch.stack(
        [torch.nonzero(inverse == group, as_tuple=False)[0, 0]
         for group in range(group_count)]
    ) if group_count else torch.empty(0, dtype=torch.long, device=input_ids.device)
    return first, inverse


def make_replica_local_layer_controller(layer: Any) -> Any:
    """Wrap one decoder block with replica-local capture/intervention state.

    ``torch.nn.DataParallel`` shallow-copies a module's hook registries while it
    creates replicas.  Registering temporary hooks inside ``forward`` can
    therefore make two replica threads run hooks that close over tensors from
    different GPUs.  This controller is itself a decoder child module, so each
    replica receives an independent controller and independent operation state.
    """

    from prefix_displacement.schema import require_torch

    torch = require_torch()

    class _ReplicaLocalLayerController(torch.nn.Module):
        def __init__(self, inner: Any) -> None:
            super().__init__()
            self.inner = inner
            self._operation: str | None = None
            self._positions: Any | None = None
            self._replacement: Any | None = None
            self._captured: Any | None = None

        def capture_at(self, positions: Any) -> None:
            self._configure("capture", positions, None)

        def replace_at(self, positions: Any, replacement: Any) -> None:
            self._configure("replace", positions, replacement)

        def gradient_leaf_at(self, positions: Any) -> None:
            self._configure("gradient_leaf", positions, None)

        def _configure(
            self, operation: str, positions: Any, replacement: Any | None
        ) -> None:
            if self._operation is not None:
                raise HookContractError(
                    f"Layer controller already configured for {self._operation}"
                )
            self._operation = operation
            self._positions = positions
            self._replacement = replacement
            self._captured = None

        def take_captured(self) -> Any:
            captured = self._captured
            self._captured = None
            if captured is None:
                raise HookContractError("Controlled decoder layer did not run")
            return captured

        def clear(self) -> None:
            self._operation = None
            self._positions = None
            self._replacement = None
            self._captured = None

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            output = self.inner(*args, **kwargs)
            operation = self._operation
            if operation is None:
                return output
            hidden = hidden_tensor_from_output(output)
            positions = self._positions
            if positions is None:
                raise HookContractError("Controlled layer has no sequence positions")
            if operation == "capture":
                self._captured = gather_sequence_positions(hidden, positions)
                return output
            if operation == "replace":
                replacement = self._replacement
                if replacement is None:
                    raise HookContractError("Replacement operation has no tensor")
                modified = replace_hidden_at_positions(hidden, positions, replacement)
                return output_with_hidden_tensor(output, modified)
            if operation == "gradient_leaf":
                endpoint_leaf = (
                    gather_sequence_positions(hidden.detach(), positions)
                    .clone()
                    .requires_grad_(True)
                )
                self._captured = endpoint_leaf
                modified = replace_hidden_at_positions(
                    hidden.detach(), positions, endpoint_leaf
                )
                return output_with_hidden_tensor(output, modified)
            raise HookContractError(f"Unknown layer-controller operation: {operation}")

    return _ReplicaLocalLayerController(layer)


@contextmanager
def capture_layer_outputs(
    layers: Sequence[Any], layer_ids: Iterable[int], positions: Any | None = None
) -> Iterator[MutableMapping[int, Any]]:
    """Capture block outputs for a non-DataParallel module or hook unit test.

    Production DataParallel paths use ``make_replica_local_layer_controller``;
    temporary hook registration inside concurrent replica forwards is unsafe.
    """

    requested = tuple(int(layer_id) for layer_id in layer_ids)
    if len(set(requested)) != len(requested):
        raise HookContractError("Target layer IDs must be unique")
    captured: MutableMapping[int, Any] = {}
    handles = []
    for layer_id in requested:
        if not 0 <= layer_id < len(layers):
            raise HookContractError(
                f"Layer {layer_id} outside decoder depth {len(layers)}"
            )

        def hook(_module: Any, _inputs: Any, output: Any, *, index: int = layer_id) -> None:
            hidden = hidden_tensor_from_output(output)
            captured[index] = (
                hidden if positions is None else gather_sequence_positions(hidden, positions)
            )

        handles.append(layers[layer_id].register_forward_hook(hook))
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def replace_hidden_at_positions(hidden: Any, positions: Any, replacement: Any) -> Any:
    """Clone ``hidden`` and replace exactly one sequence position per batch row."""

    from prefix_displacement.schema import require_torch

    torch = require_torch()
    if hidden.ndim != 3 or replacement.ndim != 2:
        raise HookContractError(
            "hidden and replacement must have shapes [B,S,H] and [B,H]"
        )
    if replacement.shape != (hidden.shape[0], hidden.shape[2]):
        raise HookContractError("Replacement shape does not match batch and hidden size")
    if positions.shape != (hidden.shape[0],):
        raise HookContractError("positions must have shape [batch]")
    positions = positions.to(device=hidden.device)
    if bool((positions < 0).any()) or bool((positions >= hidden.shape[1]).any()):
        raise HookContractError("Replacement position is outside sequence bounds")
    modified = hidden.clone()
    batch = torch.arange(hidden.shape[0], device=hidden.device)
    modified[batch, positions] = replacement.to(device=hidden.device, dtype=hidden.dtype)
    return modified


def make_position_replacement_hook(positions: Any, replacement: Any):
    """Build an intervention hook safe for Tensor and tuple block outputs."""

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = hidden_tensor_from_output(output)
        modified = replace_hidden_at_positions(hidden, positions, replacement)
        return output_with_hidden_tensor(output, modified)

    return hook
