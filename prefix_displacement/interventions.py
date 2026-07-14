from __future__ import annotations

from typing import Any

from .schema import require_torch
from .extraction import resolve_decoder_layers


def random_orthogonal_matrix(rank: int, seed: int, device: Any = None) -> Any:
    torch = require_torch()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    matrix = torch.randn(rank, rank, generator=generator)
    q, r = torch.linalg.qr(matrix)
    signs = torch.diagonal(r).sign().clamp_min(1.0)
    return (q * signs).to(device=device) if device is not None else q * signs


def rotate_basis(basis: Any, rotation: Any) -> Any:
    return basis @ rotation


def coordinate_statistics(basis: Any, delta: Any, gradient: Any):
    coefficients = basis.transpose(-1, -2) @ delta.unsqueeze(-1)
    gradient_coordinates = basis.transpose(-1, -2) @ gradient.unsqueeze(-1)
    coefficients = coefficients.squeeze(-1)
    gradient_coordinates = gradient_coordinates.squeeze(-1)
    return coefficients, gradient_coordinates, coefficients * gradient_coordinates


def deletion_intervention(
    basis: Any,
    delta: Any,
    gradient: Any,
    *,
    k: int,
    alpha: float,
    selector: str,
    seed: int = 0,
) -> tuple[Any, Any]:
    torch = require_torch()
    coefficients, gradient_coordinates, contribution = coordinate_statistics(
        basis, delta, gradient
    )
    rank = basis.shape[-1]
    if not 0 <= k <= rank:
        raise ValueError(f"k must be in [0, {rank}]")
    if k == 0 or alpha == 0:
        zero = torch.zeros_like(delta)
        return zero, torch.zeros(delta.shape[0], device=delta.device)
    if selector == "most_negative_q_times_a":
        indices = contribution.topk(k, dim=-1, largest=False).indices
    elif selector == "most_positive_q_times_a":
        indices = contribution.topk(k, dim=-1, largest=True).indices
    elif selector == "largest_abs_a":
        indices = coefficients.abs().topk(k, dim=-1).indices
    elif selector == "largest_abs_q":
        indices = gradient_coordinates.abs().topk(k, dim=-1).indices
    elif selector == "random":
        generator = torch.Generator(device="cpu").manual_seed(seed)
        permutations = [torch.randperm(rank, generator=generator)[:k] for _ in range(delta.shape[0])]
        indices = torch.stack(permutations).to(delta.device)
    else:
        raise ValueError(f"Unknown coordinate selector: {selector}")
    selected_a = coefficients.gather(1, indices)
    selected_basis = basis.gather(
        2, indices[:, None, :].expand(-1, basis.shape[1], -1)
    )
    removed = (selected_basis * selected_a[:, None, :]).sum(dim=-1)
    intervention = -float(alpha) * removed
    predicted_effect = (gradient * intervention).sum(dim=-1)
    return intervention, predicted_effect


def projected_gradient_intervention(
    basis: Any,
    gradient: Any,
    reference_intervention: Any,
    *,
    mode: str,
) -> Any:
    projected_gradient = (
        basis @ (basis.transpose(-1, -2) @ gradient.unsqueeze(-1))
    ).squeeze(-1)
    if mode == "norm_matched":
        scale = reference_intervention.norm(dim=-1) / projected_gradient.norm(dim=-1).clamp_min(1e-12)
    elif mode == "predicted_effect_matched":
        target = (gradient * reference_intervention).sum(dim=-1)
        denominator = (gradient * projected_gradient).sum(dim=-1).clamp_min(1e-12)
        scale = target / denominator
    else:
        raise ValueError("mode must be norm_matched or predicted_effect_matched")
    return projected_gradient * scale[:, None]


class InterventionPairForward:
    """Baseline/intervened forward pair; vocabulary tensors stay on each replica."""

    @staticmethod
    def build(backbone: Any, layer_index: int) -> Any:
        torch = require_torch()

        class _InterventionPairForward(torch.nn.Module):
            def __init__(self, model: Any, index: int) -> None:
                super().__init__()
                self.backbone = model
                self.layer_index = index

            def _margin(self, logits: Any, evaluation_position: Any, positive: Any, negative: Any):
                batch = torch.arange(logits.shape[0], device=logits.device)
                selected = logits[batch, evaluation_position].float()
                return (
                    selected.gather(1, positive[:, None]).squeeze(1)
                    - selected.gather(1, negative[:, None]).squeeze(1)
                ), selected

            def forward(
                self,
                input_ids: Any,
                attention_mask: Any,
                evaluation_position: Any,
                positive_token_id: Any,
                negative_token_id: Any,
                intervention_position: Any,
                intervention_vector: Any,
                sample_index: Any,
            ):
                with torch.no_grad():
                    baseline_outputs = self.backbone(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True,
                    )
                    baseline_margin, baseline_logits = self._margin(
                        baseline_outputs.logits,
                        evaluation_position,
                        positive_token_id,
                        negative_token_id,
                    )

                    layers = resolve_decoder_layers(self.backbone)

                    def intervention_hook(_module: Any, _inputs: Any, output: Any):
                        hidden = output[0] if isinstance(output, tuple) else output
                        modified = hidden.clone()
                        batch = torch.arange(hidden.shape[0], device=hidden.device)
                        modified[batch, intervention_position] += intervention_vector.to(hidden.dtype)
                        if isinstance(output, tuple):
                            return (modified, *output[1:])
                        return modified

                    handle = layers[self.layer_index].register_forward_hook(intervention_hook)
                    try:
                        intervened_outputs = self.backbone(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            use_cache=False,
                            return_dict=True,
                        )
                    finally:
                        handle.remove()
                    intervened_margin, intervened_logits = self._margin(
                        intervened_outputs.logits,
                        evaluation_position,
                        positive_token_id,
                        negative_token_id,
                    )
                    baseline_logprob = baseline_logits.log_softmax(dim=-1)
                    intervened_logprob = intervened_logits.log_softmax(dim=-1)
                    baseline_prob = baseline_logprob.exp()
                    intervened_prob = intervened_logprob.exp()
                    kl = (baseline_prob * (baseline_logprob - intervened_logprob)).sum(dim=-1)
                    mixture = 0.5 * (baseline_prob + intervened_prob)
                    mixture_log = mixture.clamp_min(1e-30).log()
                    js = 0.5 * (
                        (baseline_prob * (baseline_logprob - mixture_log)).sum(dim=-1)
                        + (intervened_prob * (intervened_logprob - mixture_log)).sum(dim=-1)
                    )
                return (
                    baseline_margin,
                    intervened_margin,
                    kl,
                    js,
                    sample_index,
                )

        return _InterventionPairForward(backbone, layer_index)
