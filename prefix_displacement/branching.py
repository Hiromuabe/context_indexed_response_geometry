from __future__ import annotations

from typing import Any

from .extraction import resolve_decoder_layers
from .schema import require_torch


class BranchForward:
    @staticmethod
    def build(backbone: Any, layer_index: int) -> Any:
        torch = require_torch()

        class _BranchForward(torch.nn.Module):
            def __init__(self, model: Any, index: int) -> None:
                super().__init__()
                self.backbone = model
                self.layer_index = index

            def forward(
                self,
                input_ids: Any,
                attention_mask: Any,
                endpoint_position: Any,
                candidate_pool_size: int,
            ):
                layers = resolve_decoder_layers(self.backbone)
                captured: dict[str, Any] = {}

                def hook(_module: Any, _inputs: Any, output: Any):
                    captured["hidden"] = output[0] if isinstance(output, tuple) else output

                handle = layers[self.layer_index].register_forward_hook(hook)
                try:
                    outputs = self.backbone(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True,
                    )
                finally:
                    handle.remove()
                batch = torch.arange(input_ids.shape[0], device=input_ids.device)
                hidden = captured["hidden"][batch, endpoint_position]
                logits = outputs.logits[batch, endpoint_position]
                candidate_ids = logits.float().topk(candidate_pool_size, dim=-1).indices
                return hidden.detach(), candidate_ids.detach()

        return _BranchForward(backbone, layer_index)


def held_out_local_svd(
    branch_delta: Any,
    *,
    estimation_indices: Any,
    held_out_indices: Any,
    rank: int,
) -> dict[str, float]:
    torch = require_torch()
    estimation = branch_delta.index_select(0, estimation_indices).float()
    held_out = branch_delta.index_select(0, held_out_indices).float()
    _u, _s, vh = torch.linalg.svd(estimation, full_matrices=False)
    basis = vh[:rank].transpose(0, 1)
    reconstruction = (held_out @ basis) @ basis.transpose(0, 1)
    cosine = torch.nn.functional.cosine_similarity(held_out, reconstruction, dim=-1)
    retained = reconstruction.square().sum(dim=-1) / held_out.square().sum(dim=-1).clamp_min(1e-12)
    return {
        "mean_reconstruction_cosine": float(cosine.mean()),
        "mean_retained_energy": float(retained.mean()),
    }
