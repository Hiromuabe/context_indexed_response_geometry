from __future__ import annotations

import unittest

import torch

from prefix_displacement.interventions import (
    deletion_intervention,
    random_orthogonal_matrix,
    rotate_basis,
)


class InterventionTest(unittest.TestCase):
    def test_rotation_preserves_projection(self) -> None:
        torch.manual_seed(0)
        basis, _ = torch.linalg.qr(torch.randn(2, 8, 4))
        delta = torch.randn(2, 8)
        rotation = random_orthogonal_matrix(4, seed=3)
        rotated = rotate_basis(basis, rotation)
        original_projection = basis @ (basis.transpose(-1, -2) @ delta.unsqueeze(-1))
        rotated_projection = rotated @ (rotated.transpose(-1, -2) @ delta.unsqueeze(-1))
        torch.testing.assert_close(original_projection, rotated_projection)

    def test_k_zero_is_identity_intervention(self) -> None:
        basis, _ = torch.linalg.qr(torch.randn(2, 8, 4))
        delta, gradient = torch.randn(2, 8), torch.randn(2, 8)
        intervention, predicted = deletion_intervention(
            basis, delta, gradient, k=0, alpha=1.0,
            selector="most_negative_q_times_a"
        )
        torch.testing.assert_close(intervention, torch.zeros_like(delta))
        torch.testing.assert_close(predicted, torch.zeros(2))


if __name__ == "__main__":
    unittest.main()
