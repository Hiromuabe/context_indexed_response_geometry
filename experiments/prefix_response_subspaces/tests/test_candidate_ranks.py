import unittest

import numpy as np

from experiments.prefix_response_subspaces.analyze_geometry import candidate_vocabulary_ranks


class CandidateVocabularyRankTest(unittest.TestCase):
    def test_ranks_use_tokenizer_vocabulary_and_ignore_reserved_lm_rows(self):
        logits = np.asarray([
            [9.0, 7.0, 5.0, 3.0, 1000.0, 999.0],
            [1.0, 4.0, 3.0, 2.0, 1000.0, 999.0],
        ], dtype=np.float32)
        ranks = candidate_vocabulary_ranks(logits, [0, 2, 3], 4)
        np.testing.assert_array_equal(ranks, [[1, 3, 4], [4, 2, 3]])


if __name__ == "__main__":
    unittest.main()
