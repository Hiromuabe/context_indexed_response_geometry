from __future__ import annotations

from typing import Any, Mapping

from .schema import require_torch


def orthonormalize(raw_basis: Any) -> Any:
    torch = require_torch()
    q, _ = torch.linalg.qr(raw_basis.float(), mode="reduced")
    return q


def project_with_basis(delta: Any, basis: Any) -> Any:
    coefficients = basis.transpose(-1, -2) @ delta.float().unsqueeze(-1)
    return (basis @ coefficients).squeeze(-1)


def build_projector(method: str, hidden_dimension: int, config: Mapping[str, Any]) -> Any:
    torch = require_torch()
    rank = int(config["rank"])
    if not 0 < rank <= hidden_dimension:
        raise ValueError(f"rank must be in [1, {hidden_dimension}]")

    class GlobalProjector(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.raw_basis = torch.nn.Parameter(torch.randn(hidden_dimension, rank) * 0.02)

        def forward(self, basepoint: Any, arrival: Any, delta: Any, **_kwargs: Any):
            basis = orthonormalize(self.raw_basis)
            expanded = basis.unsqueeze(0).expand(delta.shape[0], -1, -1)
            return project_with_basis(delta, expanded), expanded, delta.new_zeros(delta.shape[0])

    class RandomProjector(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            generator = torch.Generator().manual_seed(int(config.get("seed", 0)))
            basis = orthonormalize(
                torch.randn(hidden_dimension, rank, generator=generator)
            )
            self.register_buffer("fixed_basis", basis)

        def forward(self, basepoint: Any, arrival: Any, delta: Any, **_kwargs: Any):
            expanded = self.fixed_basis.unsqueeze(0).expand(delta.shape[0], -1, -1)
            return project_with_basis(delta, expanded), expanded, delta.new_zeros(delta.shape[0])

    class StateConditionedProjector(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            experts = int(config.get("num_experts", 16))
            router_hidden = int(config.get("router_hidden", 512))
            self.router = torch.nn.Sequential(
                torch.nn.LayerNorm(hidden_dimension),
                torch.nn.Linear(hidden_dimension, router_hidden),
                torch.nn.GELU(),
                torch.nn.Linear(router_hidden, experts),
            )
            self.experts = torch.nn.Parameter(
                torch.randn(experts, hidden_dimension, rank) * 0.02
            )

        def forward(self, basepoint: Any, arrival: Any, delta: Any, **_kwargs: Any):
            weights = self.router(basepoint.float()).softmax(dim=-1)
            raw = torch.einsum("be,edr->bdr", weights, self.experts)
            basis = orthonormalize(raw)
            return project_with_basis(delta, basis), basis, delta.new_zeros(delta.shape[0])

    class LexicalMetadataProjector(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            vocabulary_size = int(config["vocabulary_size"])
            embedding_dim = int(config.get("token_embedding_dim", 128))
            experts = int(config.get("num_experts", 16))
            router_hidden = int(config.get("router_hidden", 512))
            self.current_embedding = torch.nn.Embedding(vocabulary_size, embedding_dim)
            self.next_embedding = torch.nn.Embedding(vocabulary_size, embedding_dim)
            self.use_current = bool(config.get("use_current_token", True))
            self.use_next = bool(config.get("use_next_token", True))
            self.use_metadata = bool(config.get("use_metadata", True))
            input_dim = embedding_dim * (self.use_current + self.use_next)
            input_dim += 2 if self.use_metadata else 0
            if input_dim == 0:
                raise ValueError("Lexical/metadata router has no enabled inputs")
            self.router = torch.nn.Sequential(
                torch.nn.Linear(input_dim, router_hidden),
                torch.nn.GELU(),
                torch.nn.Linear(router_hidden, experts),
            )
            self.experts = torch.nn.Parameter(
                torch.randn(experts, hidden_dimension, rank) * 0.02
            )

        def forward(
            self,
            basepoint: Any,
            arrival: Any,
            delta: Any,
            current_token_id: Any,
            next_token_id: Any,
            position: Any,
            surprisal: Any,
            **_kwargs: Any,
        ):
            features = []
            if self.use_current:
                features.append(self.current_embedding(current_token_id))
            if self.use_next:
                features.append(self.next_embedding(next_token_id))
            if self.use_metadata:
                metadata = torch.stack((position, surprisal), dim=-1).float()
                metadata = metadata / metadata.abs().mean(dim=0, keepdim=True).clamp_min(1.0)
                features.append(metadata)
            weights = self.router(torch.cat(features, dim=-1)).softmax(dim=-1)
            basis = orthonormalize(torch.einsum("be,edr->bdr", weights, self.experts))
            return project_with_basis(delta, basis), basis, delta.new_zeros(delta.shape[0])

    class DirectTransitionAutoencoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Linear(hidden_dimension, rank, bias=False)
            self.decoder = torch.nn.Linear(rank, hidden_dimension, bias=False)

        def forward(self, basepoint: Any, arrival: Any, delta: Any, **_kwargs: Any):
            latent = self.encoder(delta.float())
            reconstruction = self.decoder(latent)
            empty = delta.new_empty((delta.shape[0], hidden_dimension, 0))
            return reconstruction, empty, latent.abs().mean(dim=-1)

    class PairedEndpointModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Linear(hidden_dimension * 2, rank)
            self.decoder = torch.nn.Linear(rank, hidden_dimension)

        def forward(self, basepoint: Any, arrival: Any, delta: Any, **_kwargs: Any):
            latent = self.encoder(torch.cat((basepoint.float(), arrival.float()), dim=-1))
            reconstruction = self.decoder(latent)
            empty = delta.new_empty((delta.shape[0], hidden_dimension, 0))
            return reconstruction, empty, delta.new_zeros(delta.shape[0])

    class HiddenSaeDifference(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Linear(hidden_dimension, rank, bias=False)
            self.decoder = torch.nn.Linear(rank, hidden_dimension, bias=False)

        def reconstruct(self, hidden: Any) -> Any:
            return self.decoder(self.encoder(hidden.float()))

        def forward(self, basepoint: Any, arrival: Any, delta: Any, **_kwargs: Any):
            departure_latent = self.encoder(basepoint.float())
            arrival_latent = self.encoder(arrival.float())
            reconstruction = self.decoder(arrival_latent) - self.decoder(departure_latent)
            empty = delta.new_empty((delta.shape[0], hidden_dimension, 0))
            sparsity = 0.5 * (
                departure_latent.abs().mean(dim=-1) + arrival_latent.abs().mean(dim=-1)
            )
            return reconstruction, empty, sparsity

    methods = {
        "global": GlobalProjector,
        "random": RandomProjector,
        "bclp": StateConditionedProjector,
        "lexical_metadata": LexicalMetadataProjector,
        "direct_transition_sae": DirectTransitionAutoencoder,
        "paired_endpoint": PairedEndpointModel,
        "h_sae_difference": HiddenSaeDifference,
    }
    if method not in methods:
        raise ValueError(f"Unknown projector method: {method}")
    return methods[method]()


def reconstruction_loss(delta: Any, delta_hat: Any, *, normalized: bool, epsilon: float):
    residual_energy = (delta.float() - delta_hat.float()).square().sum(dim=-1)
    if normalized:
        residual_energy = residual_energy / delta.float().square().sum(dim=-1).clamp_min(
            epsilon
        )
    return residual_energy.mean()


def model_inputs(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    return {
        "basepoint": batch["h_departure"].to(device, non_blocking=True),
        "arrival": batch["h_arrival"].to(device, non_blocking=True),
        "delta": batch["delta"].to(device, non_blocking=True),
        "current_token_id": batch["current_token_id"].to(device, non_blocking=True),
        "next_token_id": batch["next_token_id"].to(device, non_blocking=True),
        "position": batch["position"].to(device, non_blocking=True),
        "surprisal": batch["surprisal"].to(device, non_blocking=True),
    }
