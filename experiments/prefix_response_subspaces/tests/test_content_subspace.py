import unittest
import numpy as np

from experiments.prefix_response_subspaces.src.subspaces import content_subspace, explained_variance


class ContentSubspaceTest(unittest.TestCase):
    def test_bos_sink_is_excluded_and_rank_is_matched(self):
        hidden = np.zeros((6, 8)); hidden[0, 7] = 1e6; hidden[1:, 1] = [-2,-1,0,1,2]
        basis, positions = content_subspace(hidden, 4, excluded_positions={0})
        self.assertNotIn(0, positions); self.assertEqual(basis.shape, (8,1)); self.assertGreater(abs(basis[1,0]), .999); self.assertLess(abs(basis[7,0]), 1e-8)


if __name__ == "__main__": unittest.main()
