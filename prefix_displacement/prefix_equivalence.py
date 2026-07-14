from __future__ import annotations

from typing import Any, Callable

from .schema import require_torch


class PrefixEquivalenceError(AssertionError):
    """Raised when full-sequence and isolated-prefix endpoints disagree."""


def assert_prefix_endpoint_equivalence(
    input_ids: Any,
    hidden_state_forward: Callable[[Any], Any],
    *,
    atol: float,
    rtol: float,
) -> dict[str, float | int]:
    """Compare every full-forward position with its isolated-prefix endpoint.

    ``hidden_state_forward`` must return a tensor shaped ``[batch, sequence, d]``
    at one fixed layer/hook. The helper intentionally does not know model-specific
    hook conventions; those must be resolved before a real-model test is run.
    """
    torch = require_torch()
    input_ids = torch.as_tensor(input_ids)
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("input_ids must have shape [1, sequence]")
    if input_ids.shape[1] < 2:
        raise ValueError("At least two tokens are required")

    with torch.no_grad():
        full_hidden = hidden_state_forward(input_ids)
        if full_hidden.ndim != 3 or full_hidden.shape[:2] != input_ids.shape:
            raise ValueError(
                "hidden_state_forward must return [batch, sequence, d], got "
                f"{tuple(full_hidden.shape)}"
            )
        prefix_endpoints = []
        for end in range(1, input_ids.shape[1] + 1):
            prefix_hidden = hidden_state_forward(input_ids[:, :end])
            if prefix_hidden.ndim != 3 or prefix_hidden.shape[0] != 1:
                raise ValueError("Prefix hidden states must have shape [1, prefix, d]")
            prefix_endpoints.append(prefix_hidden[:, -1, :])
        prefix_hidden = torch.stack(prefix_endpoints, dim=1)

    absolute_error = (full_hidden.float() - prefix_hidden.float()).abs()
    max_abs_error = float(absolute_error.max().item())
    try:
        torch.testing.assert_close(
            full_hidden.float(), prefix_hidden.float(), atol=atol, rtol=rtol
        )
    except AssertionError as exc:
        raise PrefixEquivalenceError(
            "Full-sequence positions do not match isolated-prefix endpoints; "
            f"max_abs_error={max_abs_error:.9g}, atol={atol}, rtol={rtol}"
        ) from exc

    full_delta = full_hidden[:, 1:, :] - full_hidden[:, :-1, :]
    prefix_delta = prefix_hidden[:, 1:, :] - prefix_hidden[:, :-1, :]
    delta_max_abs_error = float(
        (full_delta.float() - prefix_delta.float()).abs().max().item()
    )
    try:
        torch.testing.assert_close(
            full_delta.float(), prefix_delta.float(), atol=atol, rtol=rtol
        )
    except AssertionError as exc:
        raise PrefixEquivalenceError(
            "Adjacent-position displacement does not match adjacent-prefix endpoint "
            f"displacement; max_abs_error={delta_max_abs_error:.9g}"
        ) from exc

    return {
        "num_tokens": int(input_ids.shape[1]),
        "hidden_dimension": int(full_hidden.shape[-1]),
        "endpoint_max_abs_error": max_abs_error,
        "delta_max_abs_error": delta_max_abs_error,
    }
