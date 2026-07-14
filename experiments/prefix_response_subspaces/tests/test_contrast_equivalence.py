import unittest
import numpy as np

from experiments.prefix_response_subspaces.src.residualization import centered_residual_ev, explicit_contrast_ev
from experiments.prefix_response_subspaces.src.subspaces import top_svd


class ContrastEquivalenceTest(unittest.TestCase):
    def test_explicit_pairwise_contrast_equals_centered_residual_ev(self):
        rng = np.random.default_rng(12)
        residuals = rng.normal(size=(17, 11)); residuals -= residuals.mean(axis=0)
        basis = top_svd(rng.normal(size=(30, 11)), 4)
        self.assertAlmostEqual(explicit_contrast_ev(residuals, basis), centered_residual_ev(residuals, basis), places=12)
        pair_energy = np.square(residuals[:, None] - residuals[None, :]).sum()
        centered_energy = 2 * len(residuals) * np.square(residuals).sum()
        self.assertAlmostEqual(pair_energy, centered_energy, places=9)


if __name__ == "__main__": unittest.main()
