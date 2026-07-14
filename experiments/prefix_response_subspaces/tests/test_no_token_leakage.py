import unittest

from experiments.prefix_response_subspaces.build_candidate_tokens import logit_scored_indices, stratified_nested_order, stratified_partition
from experiments.prefix_response_subspaces.src.data import assign_groups


class NoTokenLeakageTest(unittest.TestCase):
    def test_paper_logits_skip_auxiliary_global_and_wrong_donor_rows(self):
        groups=["candidate_selection","auxiliary","analysis_train","analysis_dev","analysis_test","matching_pool"]
        rows=[{"problem_group":group} for group in groups]
        self.assertEqual(logit_scored_indices(rows,True),[0,3,4])
        self.assertEqual(logit_scored_indices(rows,False),list(range(6)))

    def test_problem_group_assignment_is_one_to_one(self):
        config = {"data": {"candidate_selection_prefixes": 4, "auxiliary_prefixes": 4, "analysis_dev_prefixes": 4, "evaluation_prefixes": 4, "analysis_train_prefixes": 4}}
        groups = assign_groups([f"q{i}" for i in range(32)], config, 5)
        self.assertEqual(len(groups), 32)
        self.assertEqual(sum(value == "analysis_test" for value in groups.values()), 4)
        self.assertEqual(sum(value == "analysis_train" for value in groups.values()), 4)

    def test_calibration_train_and_evaluation_are_disjoint(self):
        rows = [{"probability_band": i%4, "coverage_band": (i//4)%4, "category": ["number","word","operator","subword"][i%4]} for i in range(40)]
        calibration, heldouts = stratified_partition(rows, 8, 4, 17)
        self.assertEqual(len(calibration), 8); self.assertEqual(sorted(calibration + [x for fold in heldouts for x in fold]), list(range(40)))
        self.assertEqual(len(set(calibration)), 8)
        analysis = set(x for fold in heldouts for x in fold)
        self.assertFalse(set(calibration) & analysis)
        for heldout in heldouts:
            train = analysis - set(heldout)
            self.assertEqual(len(heldout), 8); self.assertEqual(len(train), 24); self.assertFalse(train & set(heldout)); self.assertFalse(train & set(calibration))

    def test_rank0_calibration_order_preserves_membership_and_interleaves_strata(self):
        rows = [{"probability_band": i % 2, "coverage_band": 0, "category": "number" if i % 2 == 0 else "word"} for i in range(8)]
        order = stratified_nested_order(rows, list(range(8)), 13)
        self.assertEqual(sorted(order), list(range(8)))
        first_four_strata = {(rows[i]["probability_band"], rows[i]["category"]) for i in order[:4]}
        self.assertEqual(len(first_four_strata), 2)


if __name__ == "__main__": unittest.main()
