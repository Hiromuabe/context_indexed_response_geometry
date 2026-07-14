import unittest

from experiments.prefix_response_subspaces.select_wrong_prefixes import select_wrong_prefix_controls


class WrongPrefixControlTest(unittest.TestCase):
    def test_controls_are_problem_disjoint_and_prefer_same_last_token(self):
        target = {"prefix_id": "t", "problem_id": "qt", "problem_group": "analysis_test", "prefix_length_bin": 1, "reasoning_progress_bin": 2, "last_token_id": 7, "prefix_length": 20, "prefix_position_fraction": .5}
        donors = [{"prefix_id": f"d{i}", "problem_id": f"q{i}", "problem_group": "analysis_train", "prefix_length_bin": 1, "reasoning_progress_bin": 2, "last_token_id": 7 if i < 2 else 9, "prefix_length": 20+i, "prefix_position_fraction": .5} for i in range(6)]
        row = select_wrong_prefix_controls([target, *donors], 5)[0]
        self.assertEqual(len(row["wrong_prefix_ids"]), 5)
        self.assertEqual(row["wrong_prefix_ids"][:2], ["d0", "d1"])
        self.assertNotIn("qt", row["wrong_problem_ids"])

    def test_empty_primary_bin_uses_exact_bin_evaluation_safe_fallback(self):
        target={"prefix_id":"t","problem_id":"qt","problem_group":"analysis_test","prefix_length_bin":3,"reasoning_progress_bin":2,"last_token_id":7,"prefix_length":99,"prefix_position_fraction":.8}
        fallback=[{"prefix_id":f"f{i}","problem_id":f"qf{i}","problem_group":"analysis_dev" if i<3 else "analysis_test","prefix_length_bin":3,"reasoning_progress_bin":2,"last_token_id":7,"prefix_length":100+i,"prefix_position_fraction":.8} for i in range(5)]
        row=select_wrong_prefix_controls([target,*fallback],5)[0]
        self.assertTrue(row["complete"]); self.assertEqual(row["fallback_wrong_prefixes"],5); self.assertEqual(row["relaxed_length_wrong_prefixes"],0); self.assertEqual(row["primary_eligible_pool_size"],0)

    def test_structurally_isolated_exact_bin_uses_recorded_nearest_length_bin(self):
        target={"prefix_id":"t","problem_id":"qt","problem_group":"analysis_test","prefix_length_bin":3,"reasoning_progress_bin":2,"last_token_id":7,"prefix_length":99,"prefix_position_fraction":.8}
        donors=[{"prefix_id":f"d{i}","problem_id":f"qd{i}","problem_group":"analysis_train","prefix_length_bin":2,"reasoning_progress_bin":2,"last_token_id":7,"prefix_length":90+i,"prefix_position_fraction":.8} for i in range(5)]
        row=select_wrong_prefix_controls([target,*donors],5)[0]
        self.assertTrue(row["complete"]); self.assertEqual(row["fallback_wrong_prefixes"],0); self.assertEqual(row["relaxed_length_wrong_prefixes"],5); self.assertEqual(row["maximum_length_bin_distance"],1)


if __name__ == "__main__": unittest.main()
