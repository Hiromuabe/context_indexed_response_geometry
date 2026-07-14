from __future__ import annotations

from typing import Any

import numpy as np

from prefix_displacement.extraction import resolve_decoder_layers
from prefix_displacement.runtime import prepare_data_parallel
from prefix_displacement.schema import require_torch
from experiments.prefix_successor_subspaces.src.hooks import gather_sequence_positions, hidden_tensor_from_output
from experiments.prefix_successor_subspaces.src.model import LoadedForwardModel, _decoder_hidden, _load_backbone_and_tokenizer, _endpoint_logits

SITE_NAMES = ("pre_attention", "post_attention", "post_mlp")


class _FirstLayerEarlyExit(RuntimeError):
    """Internal control flow used to skip decoder blocks after layer zero."""


def _hidden_argument(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    if args:
        return args[0]
    if "hidden_states" in kwargs:
        return kwargs["hidden_states"]
    raise RuntimeError("decoder layer did not receive hidden_states")


def make_replica_local_input_capture(module: Any) -> Any:
    """Wrap a submodule and capture its input at replica-local positions."""
    torch = require_torch()

    class _InputCapture(torch.nn.Module):
        def __init__(self, inner: Any) -> None:
            super().__init__()
            self.inner = inner
            self._positions = None
            self._captured = None

        def capture_input_at(self, positions: Any) -> None:
            if self._positions is not None:
                raise RuntimeError("input capture already configured")
            self._positions = positions
            self._captured = None

        def take_captured(self) -> Any:
            value = self._captured
            self._captured = None
            self._positions = None
            if value is None:
                raise RuntimeError("captured submodule did not run")
            return value

        def clear(self) -> None:
            self._positions = None
            self._captured = None

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            hidden = _hidden_argument(args, kwargs)
            if self._positions is not None:
                self._captured = gather_sequence_positions(hidden, self._positions)
            return self.inner(*args, **kwargs)

    return _InputCapture(module)


def make_first_layer_site_controller(layer: Any) -> Any:
    """Capture pre-attention, post-attention-residual, and block output sites."""
    torch = require_torch()
    if not hasattr(layer, "post_attention_layernorm"):
        raise TypeError("first-layer mechanism extraction requires post_attention_layernorm")
    layer.post_attention_layernorm = make_replica_local_input_capture(layer.post_attention_layernorm)

    class _FirstLayerSiteController(torch.nn.Module):
        def __init__(self, inner: Any) -> None:
            super().__init__()
            self.inner = inner
            self._positions = None
            self._pre = None
            self._post_mlp = None
            self._capture_sequence = False
            self._sequence_input = None

        def capture_at(self, positions: Any) -> None:
            if self._positions is not None or self._capture_sequence:
                raise RuntimeError("first-layer capture already configured")
            self._positions = positions
            self._pre = None
            self._post_mlp = None
            self.inner.post_attention_layernorm.capture_input_at(positions)

        def take_captured(self) -> tuple[Any, Any, Any]:
            if self._pre is None or self._post_mlp is None:
                raise RuntimeError("first decoder layer did not run")
            post_attention = self.inner.post_attention_layernorm.take_captured()
            result = self._pre, post_attention, self._post_mlp
            self._positions = None
            self._pre = None
            self._post_mlp = None
            return result

        def capture_sequence_input(self) -> None:
            if self._positions is not None or self._capture_sequence:
                raise RuntimeError("first-layer capture already configured")
            self._capture_sequence = True
            self._sequence_input = None

        def take_sequence_input(self) -> Any:
            value = self._sequence_input
            self._capture_sequence = False
            self._sequence_input = None
            if value is None:
                raise RuntimeError("first decoder layer did not run")
            return value

        def clear(self) -> None:
            self._positions = None
            self._pre = None
            self._post_mlp = None
            self._capture_sequence = False
            self._sequence_input = None
            self.inner.post_attention_layernorm.clear()

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            hidden = _hidden_argument(args, kwargs)
            if self._capture_sequence:
                self._sequence_input = hidden
                raise _FirstLayerEarlyExit
            if self._positions is not None:
                self._pre = gather_sequence_positions(hidden, self._positions)
            output = self.inner(*args, **kwargs)
            if self._positions is not None:
                self._post_mlp = gather_sequence_positions(hidden_tensor_from_output(output), self._positions)
                # The requested representation is now complete.  The wrapper
                # catches this inside each DataParallel replica, avoiding all
                # later decoder blocks without changing the model's masking or
                # first-layer call contract.
                raise _FirstLayerEarlyExit
            return output

    return _FirstLayerSiteController(layer)


class FirstLayerSitesForward:
    @staticmethod
    def build(backbone: Any, layer_id: int = 0) -> Any:
        torch = require_torch()
        layers = resolve_decoder_layers(backbone)
        if not 0 <= int(layer_id) < len(layers):
            raise ValueError("first-layer site is outside decoder depth")
        layers[int(layer_id)] = make_first_layer_site_controller(layers[int(layer_id)])

        class _FirstLayerSitesForward(torch.nn.Module):
            def __init__(self, model: Any, target_layer: int) -> None:
                super().__init__()
                self.backbone = model
                self.target_layer = int(target_layer)

            def forward(self, input_ids: Any, attention_mask: Any, endpoint_positions: Any, sample_index: Any) -> tuple[Any, Any]:
                controller = resolve_decoder_layers(self.backbone)[self.target_layer]
                controller.capture_at(endpoint_positions)
                try:
                    try:
                        _decoder_hidden(self.backbone, input_ids=input_ids, attention_mask=attention_mask)
                    except _FirstLayerEarlyExit:
                        pass
                    sites = controller.take_captured()
                finally:
                    controller.clear()
                return torch.stack(sites, dim=1).detach(), sample_index.detach()

        return _FirstLayerSitesForward(backbone, int(layer_id))


def load_first_layer_mechanism_model(config: dict[str, Any], model_path: str | None = None) -> LoadedForwardModel:
    backbone, tokenizer, source, precision_name, precision_dtype = _load_backbone_and_tokenizer(config, model_path)
    layers = resolve_decoder_layers(backbone)
    hidden_size = int(backbone.config.hidden_size)
    revision = getattr(backbone.config, "_commit_hash", None)
    wrapped = FirstLayerSitesForward.build(backbone, 0)
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
        layer_ids=[0],
    )


def _unwrapped_backbone(loaded: LoadedForwardModel) -> Any:
    wrapped = loaded.model.module if hasattr(loaded.model, "module") else loaded.model
    return wrapped.backbone


def first_layer_value_output_basis(loaded: LoadedForwardModel, prefix_token_ids: list[int], rank: int, excluded_positions: set[int] | None = None) -> tuple[np.ndarray, dict[str, int]]:
    """Construct span{W_O E_h W_V^(h) h_t} from first-layer prefix values."""
    torch = require_torch()
    backbone = _unwrapped_backbone(loaded)
    controller = resolve_decoder_layers(backbone)[0]
    if not hasattr(controller, "capture_sequence_input"):
        raise TypeError("first decoder layer is not a mechanism capture controller")
    ids = torch.tensor([list(map(int, prefix_token_ids))], dtype=torch.long, device=loaded.device)
    mask = torch.ones_like(ids)
    controller.capture_sequence_input()
    try:
        try:
            _decoder_hidden(backbone, input_ids=ids, attention_mask=mask)
        except _FirstLayerEarlyExit:
            pass
        pre = controller.take_sequence_input()[0]
    finally:
        controller.clear()
    layer = controller.inner
    attention = getattr(layer, "self_attn", None)
    if attention is None or not all(hasattr(attention, name) for name in ("v_proj", "o_proj")):
        raise TypeError("first-layer value-space extraction requires self_attn.v_proj and o_proj")
    normalized = layer.input_layernorm(pre)
    # The mechanistic comparison is the linear value/output span in the paper
    # definition.  Exclude learned affine biases even if a future compatible
    # checkpoint enables them.
    values = torch.nn.functional.linear(normalized, attention.v_proj.weight, None)
    attention_config = getattr(attention, "config", getattr(backbone, "config", None))
    num_heads_value = getattr(attention, "num_heads", None)
    if num_heads_value is None and attention_config is not None:
        num_heads_value = getattr(attention_config, "num_attention_heads", None)
    key_value_heads_value = getattr(attention, "num_key_value_heads", None)
    if key_value_heads_value is None and attention_config is not None:
        key_value_heads_value = getattr(attention_config, "num_key_value_heads", None)
    if num_heads_value is None or key_value_heads_value is None:
        raise TypeError("attention module does not expose attention-head counts")
    num_heads = int(num_heads_value)
    num_key_value_heads = int(key_value_heads_value)
    head_dim = int(getattr(attention, "head_dim", values.shape[-1] // num_key_value_heads))
    if values.shape[-1] != num_key_value_heads * head_dim:
        raise RuntimeError("unexpected first-layer value projection width")
    if num_heads % num_key_value_heads:
        raise RuntimeError("attention heads are not divisible by key/value heads")
    groups = num_heads // num_key_value_heads
    values = values.reshape(values.shape[0], num_key_value_heads, head_dim)
    positions = [index for index in range(values.shape[0]) if index not in set(excluded_positions or ())]
    if not positions:
        raise ValueError("value-space construction excluded every prefix position")
    value_rows = values[positions]
    isolated = torch.zeros((len(positions) * num_heads, num_heads * head_dim), dtype=value_rows.dtype, device=value_rows.device)
    for head in range(num_heads):
        key_value_head = head // groups
        row_slice = slice(head * len(positions), (head + 1) * len(positions))
        column_slice = slice(head * head_dim, (head + 1) * head_dim)
        isolated[row_slice, column_slice] = value_rows[:, key_value_head]
    output_vectors = torch.nn.functional.linear(isolated, attention.o_proj.weight, None).detach().float()
    effective_rank = min(int(rank), int(min(output_vectors.shape)))
    q = min(int(min(output_vectors.shape)), effective_rank + 16)
    if effective_rank <= 0:
        raise ValueError("value/output space has zero estimable rank")
    # A complete SVD for every prefix is unnecessarily expensive.  Four power
    # iterations accurately resolve the leading rank-64 span while keeping the
    # new mechanism stage much cheaper than the original branching run.
    _u, _s, right = torch.pca_lowrank(output_vectors, q=q, center=False, niter=4)
    basis = right[:, :effective_rank].detach().float().cpu().numpy()
    return basis, {
        "prefix_positions": len(positions),
        "attention_heads": num_heads,
        "key_value_heads": num_key_value_heads,
        "value_vectors": int(output_vectors.shape[0]),
        "effective_rank": int(basis.shape[1]),
        "reduction_solver": "torch.pca_lowrank(q=rank+16,niter=4,center=False)",
    }


def first_layer_value_output_span(loaded: LoadedForwardModel, prefix_token_ids: list[int], excluded_positions: set[int] | None = None) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Exact linear value/output span; ``None`` denotes the full hidden space.

    Unlike :func:`first_layer_value_output_basis`, this does not choose leading
    value PCs.  It first finds each KV head's row span, embeds that span in every
    associated query-head block, applies W_O, and orthonormalizes the result.
    """
    torch = require_torch()
    backbone = _unwrapped_backbone(loaded)
    controller = resolve_decoder_layers(backbone)[0]
    ids = torch.tensor([list(map(int, prefix_token_ids))], dtype=torch.long, device=loaded.device)
    controller.capture_sequence_input()
    try:
        try:
            _decoder_hidden(backbone, input_ids=ids, attention_mask=torch.ones_like(ids))
        except _FirstLayerEarlyExit:
            pass
        pre = controller.take_sequence_input()[0]
    finally:
        controller.clear()
    layer = controller.inner
    attention = layer.self_attn
    normalized = layer.input_layernorm(pre)
    values = torch.nn.functional.linear(normalized, attention.v_proj.weight, None)
    cfg = getattr(attention, "config", backbone.config)
    n_heads = int(getattr(attention, "num_heads", getattr(cfg, "num_attention_heads")))
    n_kv = int(getattr(attention, "num_key_value_heads", getattr(cfg, "num_key_value_heads")))
    head_dim = int(getattr(attention, "head_dim", values.shape[-1] // n_kv))
    if n_heads % n_kv or values.shape[-1] != n_kv * head_dim:
        raise RuntimeError("unsupported grouped-query attention dimensions")
    keep = [i for i in range(len(pre)) if i not in set(excluded_positions or ())]
    if not keep:
        raise ValueError("value-space construction excluded every prefix position")
    values = values[keep].reshape(len(keep), n_kv, head_dim).float()
    kv_bases, kv_ranks = [], []
    for kv in range(n_kv):
        _u, singular, vh = torch.linalg.svd(values[:, kv], full_matrices=False)
        tol = float(singular[0]) * max(values[:, kv].shape) * torch.finfo(torch.float32).eps if len(singular) else 0.0
        r = int((singular > tol).sum().item())
        kv_bases.append(vh[:r].T)
        kv_ranks.append(r)
    dimension = sum(kv_ranks[h * n_kv // n_heads] for h in range(n_heads))
    hidden = n_heads * head_dim
    input_basis = torch.zeros((hidden, dimension), device=values.device, dtype=torch.float32)
    offset = 0
    for head in range(n_heads):
        kv = head * n_kv // n_heads
        r = kv_ranks[kv]
        input_basis[head * head_dim:(head + 1) * head_dim, offset:offset + r] = kv_bases[kv]
        offset += r
    # W_O is shared by every prefix.  Prove its numerical rank once.  If it is
    # full rank, an independent d-dimensional input span remains d-dimensional;
    # in particular a full input span needs no per-prefix hidden x hidden QR.
    weight_rank = getattr(attention, "_value_output_weight_numerical_rank", None)
    if weight_rank is None:
        weight_rank = int(torch.linalg.matrix_rank(attention.o_proj.weight.float()).item())
        attention._value_output_weight_numerical_rank = weight_rank
    if weight_rank == hidden and dimension == hidden:
        return None, {
            "prefix_positions": len(keep), "attention_heads": n_heads,
            "key_value_heads": n_kv, "kv_head_ranks": kv_ranks,
            "input_span_rank": dimension, "output_projection_rank": weight_rank,
            "output_span_rank": hidden, "hidden_size": hidden,
            "full_hidden_space": True,
            "definition": "orth(W_O blockdiag_h rowspan_t(W_V^kv LN(h_i,t)))",
            "solver": "full-input-span shortcut after one-time W_O matrix-rank audit",
        }
    transformed = attention.o_proj.weight.float() @ input_basis
    if weight_rank == hidden:
        q = torch.linalg.qr(transformed, mode="reduced")[0]
        output_rank = dimension
    else:
        left, singular, _vh = torch.linalg.svd(transformed, full_matrices=False)
        tol = float(singular[0]) * max(transformed.shape) * torch.finfo(torch.float32).eps if len(singular) else 0.0
        output_rank = int((singular > tol).sum().item())
        q = left[:, :output_rank]
    full = output_rank == hidden
    basis = None if full else q[:, :output_rank].detach().cpu().numpy().astype(np.float32)
    return basis, {
        "prefix_positions": len(keep), "attention_heads": n_heads,
        "key_value_heads": n_kv, "kv_head_ranks": kv_ranks,
        "input_span_rank": dimension, "output_span_rank": output_rank,
        "hidden_size": hidden, "output_projection_rank": weight_rank,
        "full_hidden_space": full,
        "definition": "orth(W_O blockdiag_h rowspan_t(W_V^kv LN(h_i,t)))",
        "solver": "QR after one-time W_O matrix-rank audit" if weight_rank == hidden else "SVD for rank-deficient W_O",
    }
