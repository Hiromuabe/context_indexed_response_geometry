from __future__ import annotations

import unittest

import torch

from prefix_displacement.prefix_equivalence import (
    PrefixEquivalenceError,
    assert_prefix_endpoint_equivalence,
)


class ToyCausalModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 6)
        self.projection = torch.nn.Linear(6, 6, bias=False)
        generator = torch.Generator().manual_seed(11)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.randn(32, 6, generator=generator))
            self.projection.weight.copy_(torch.randn(6, 6, generator=generator))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        cumulative = torch.cumsum(self.embedding(input_ids), dim=1)
        return torch.tanh(self.projection(cumulative))


class PrefixEquivalenceTest(unittest.TestCase):
    def test_single_forward_matches_individual_prefix_endpoints(self) -> None:
        model = ToyCausalModel().eval()
        result = assert_prefix_endpoint_equivalence(
            torch.tensor([[1, 7, 4, 9, 2]], dtype=torch.long),
            model,
            atol=1e-6,
            rtol=1e-6,
        )
        self.assertLessEqual(result["endpoint_max_abs_error"], 1e-6)
        self.assertLessEqual(result["delta_max_abs_error"], 1e-6)

    def test_noncausal_forward_is_rejected(self) -> None:
        def noncausal_forward(input_ids: torch.Tensor) -> torch.Tensor:
            values = input_ids.float().unsqueeze(-1)
            return values + values.mean(dim=1, keepdim=True)

        with self.assertRaises(PrefixEquivalenceError):
            assert_prefix_endpoint_equivalence(
                torch.tensor([[1, 2, 3]], dtype=torch.long),
                noncausal_forward,
                atol=0.0,
                rtol=0.0,
            )


if __name__ == "__main__":
    unittest.main()
