"""Re-export the already-tested replica-safe hook implementation."""
from experiments.prefix_successor_subspaces.src.hooks import (  # noqa: F401
    HookContractError,
    hidden_tensor_from_output,
    make_position_replacement_hook,
    output_with_hidden_tensor,
    replace_hidden_at_positions,
)

