import unittest

import numpy as np

from experiments.prefix_response_subspaces.match_prefixes import assess_match_quality
from experiments.prefix_response_subspaces.analyze_geometry import continuous_relationship
from experiments.prefix_response_subspaces.src.matching import match_prefixes, paired_js_from_logits, softmax32
from experiments.prefix_response_subspaces.src.metrics import normalized_js_distance


class MatchingQualityTest(unittest.TestCase):
    def test_prebranch_match_uses_top1_then_overlap_then_js(self):
        records = [
            {"prefix_id": "q/p", "problem_id": "q", "prefix_length_bin": 1, "reasoning_progress_bin": 2},
            {"prefix_id": "a/p", "problem_id": "a", "prefix_length_bin": 1, "reasoning_progress_bin": 2},
            {"prefix_id": "b/p", "problem_id": "b", "prefix_length_bin": 1, "reasoning_progress_bin": 2},
        ]
        top = np.asarray([
            list(range(20)),
            list(range(20)),
            [0] + list(range(20, 39)),
        ])
        logits = np.zeros((3, 40), dtype=np.float32)
        logits[0, 0], logits[1, 0], logits[2, 0] = 5.0, 3.0, 5.0
        rows = match_prefixes(records, query_indices=np.asarray([0]), candidate_indices=np.asarray([1, 2]), logits=logits, top_token_ids=top, tokenizer_vocabulary_size=40)
        self.assertEqual(rows[0]["matched_prefix_id"], "a/p")
        self.assertEqual(rows[0]["top20_overlap"], 20)

    def test_prebranch_match_can_report_unmatched_without_blocking_other_controls(self):
        records = [
            {"prefix_id": "q/p", "problem_id": "q", "prefix_length_bin": 1, "reasoning_progress_bin": 2},
            {"prefix_id": "a/p", "problem_id": "a", "prefix_length_bin": 1, "reasoning_progress_bin": 2},
        ]
        top = np.asarray([list(range(20)), list(range(1, 21))])
        rows = match_prefixes(records, query_indices=np.asarray([0]), candidate_indices=np.asarray([1]), logits=np.zeros((2, 32), dtype=np.float32), top_token_ids=top, tokenizer_vocabulary_size=32)
        self.assertFalse(rows[0]["matched"])
        self.assertIsNone(rows[0]["matched_prefix_id"])

    def test_normalized_js_is_js_divided_by_log_two(self):
        self.assertEqual(normalized_js_distance(np.log(2.0)), 1.0)
        self.assertEqual(normalized_js_distance(0.0), 0.0)

    def test_continuous_match_relationship_records_slope_and_correlations(self):
        result = continuous_relationship([0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1])
        self.assertAlmostEqual(result["pearson"], -1.0)
        self.assertAlmostEqual(result["spearman"], -1.0)
        self.assertLess(result["ols_slope_delta_per_normalized_js"], 0.0)

    def test_matching_softmax_keeps_float64_probability_tails(self):
        probabilities = softmax32(np.asarray([[0.0, -100.0]], dtype=np.float32))
        self.assertEqual(probabilities.dtype, np.float64)
        self.assertGreater(probabilities[0, 1], 0.0)
        distance = paired_js_from_logits(
            np.asarray([[20.0, 0.0]], dtype=np.float32),
            np.asarray([[0.0, 20.0]], dtype=np.float32),
        )[0]
        self.assertLess(distance, np.log(2.0))

    def test_near_maximum_js_cannot_be_called_a_good_match(self):
        maximum = float(np.log(2.0))
        result = assess_match_quality(
            [maximum, maximum, maximum, maximum],
            [maximum, maximum, maximum, maximum],
            [maximum] * 32,
            development_quantile=0.75,
            maximum_normalized_js=0.25,
            minimum_random_median_improvement=0.25,
            minimum_good_evaluation_matches=2,
        )
        self.assertFalse(result["match_quality_pass"])
        self.assertEqual(result["good_evaluation_count"], 0)
        self.assertAlmostEqual(result["good_match_threshold"], 0.25 * maximum)

    def test_absolute_and_random_baselines_must_both_pass(self):
        result = assess_match_quality(
            [0.03, 0.04, 0.05, 0.06],
            [0.04, 0.05, 0.06, 0.07],
            [0.40] * 32,
            development_quantile=0.75,
            maximum_normalized_js=0.25,
            minimum_random_median_improvement=0.25,
            minimum_good_evaluation_matches=2,
        )
        self.assertTrue(result["match_quality_pass"])
        self.assertEqual(result["good_evaluation_count"], 2)


if __name__ == "__main__":
    unittest.main()
