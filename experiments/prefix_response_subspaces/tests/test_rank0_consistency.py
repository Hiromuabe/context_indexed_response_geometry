import unittest

from experiments.prefix_response_subspaces.analyze_functional_recovery import (
    problem_aggregated_mean,
    rank0_anchor_max_difference,
)


class RankZeroConsistencyTest(unittest.TestCase):
    def test_full_M_anchor_uses_identical_development_cells(self):
        rows = []
        for problem, value in (("q0", 0.1), ("q1", 0.2)):
            for candidate in (3, 7):
                base = {"problem_id": problem, "prefix_id": problem + "/p0", "fold": 0, "candidate_index": candidate}
                rows.append({**base, "condition": "Rank-0-reference", "js": value})
                rows.append({**base, "condition": "Rank-0-M8", "js": value})
        self.assertEqual(rank0_anchor_max_difference(rows, 8), 0.0)

    def test_aggregation_means_within_problem_before_across_problems(self):
        rows = [
            {"problem_id": "q0", "js": 0.0},
            {"problem_id": "q0", "js": 0.2},
            {"problem_id": "q1", "js": 0.8},
        ]
        self.assertAlmostEqual(problem_aggregated_mean(rows), 0.45)


if __name__ == "__main__":
    unittest.main()
