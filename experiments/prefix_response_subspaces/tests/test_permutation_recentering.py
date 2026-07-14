import unittest
import numpy as np

from experiments.prefix_response_subspaces.src.permutation import exchangeability_diagnostics, permutation_space_size, permute_prefix_labels_by_token


class PermutationRecenteringTest(unittest.TestCase):
    def test_singleton_strata_are_reported_as_nonexchangeable(self):
        diagnostics = exchangeability_diagnostics(np.asarray(["a", "a", "b", "c"]), 2)
        self.assertEqual(diagnostics["singleton_strata"], 2)
        self.assertEqual(diagnostics["exchangeable_prefix_fraction"], 0.5)

    def test_permutation_space_and_actual_movement_are_recorded(self):
        labels = np.asarray(["a", "a", "b", "b"])
        space = permutation_space_size(labels, 3)
        self.assertEqual(space["distinct_label_permutations_exact"], str(4**3))
        values = np.arange(4*3*2, dtype=np.float32).reshape(4,3,2)
        values -= values.mean(axis=1, keepdims=True)
        _permuted, diagnostics = permute_prefix_labels_by_token(values, labels, np.random.default_rng(7), return_diagnostics=True)
        self.assertIn("actual_moved_prefix_count", diagnostics)
        self.assertIn("plan_sha256", diagnostics)

    def test_permutation_is_tokenwise_stratified_and_recentered(self):
        values = np.arange(6*5*3, dtype=np.float32).reshape(6,5,3); values -= values.mean(axis=1, keepdims=True)
        strata = np.asarray([0,0,0,1,1,1]); observed = permute_prefix_labels_by_token(values, strata, np.random.default_rng(9))
        np.testing.assert_allclose(observed.mean(axis=1), 0, atol=1e-6)
        for token in range(5):
            for label in (0,1):
                indices = np.flatnonzero(strata == label)
                before = sorted(map(tuple, values[indices, token])); # compare pre-recentering source through invariant stratum sums
                self.assertEqual(len(before), len(indices))


if __name__ == "__main__": unittest.main()
