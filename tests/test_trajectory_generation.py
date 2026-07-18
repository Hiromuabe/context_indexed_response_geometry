from __future__ import annotations

import unittest

from scripts.prepare_gsm8k_trajectories import select_source_rows
from prefix_displacement.trajectory_generation import (
    extract_generated_answer,
    extract_reference_answer,
    first_distinct_answer_tokens,
    is_correct_answer,
    stable_problem_id,
)


class TrajectoryGenerationTest(unittest.TestCase):
    def test_balanced_multi_subset_selection_round_robins_subjects(self) -> None:
        selected = select_source_rows(
            [("algebra", [{"x": 1}, {"x": 2}]), ("geometry", [{"x": 3}, {"x": 4}])],
            3, seed=0, balanced_subsets=True,
        )
        self.assertEqual([row["source_subset"] for row in selected], ["algebra", "geometry", "algebra"])

    def test_gsm8k_reference_and_generated_answer(self) -> None:
        self.assertEqual(extract_reference_answer("work\n#### 1,234"), "1234")
        generated = extract_generated_answer("reasoning 12 then final 1,234")
        self.assertEqual(generated, "1234")
        self.assertTrue(is_correct_answer(generated, "1234"))

    def test_math_reference_uses_last_balanced_boxed_answer(self) -> None:
        solution = r"First \boxed{1}. Finally \boxed{\frac{3}{\sqrt{5}}}."
        expected = r"\frac{3}{\sqrt{5}}"
        self.assertEqual(extract_reference_answer(solution, "math_boxed"), expected)
        generated = extract_generated_answer(r"Thus \boxed{\frac{3}{\sqrt{5}}}", "math_boxed")
        self.assertEqual(generated, expected)
        self.assertTrue(is_correct_answer(generated, expected, "math_boxed"))

    def test_multiple_choice_answer_uses_explicit_final_label(self) -> None:
        reference = extract_reference_answer("d", "choice_label")
        generated = extract_generated_answer("A is tempting, but the final answer is (D).", "choice_label")
        self.assertEqual(reference, "D")
        self.assertEqual(generated, "D")
        self.assertTrue(is_correct_answer(generated, reference, "choice_label"))
        self.assertEqual(extract_generated_answer("Reasoning complete.\nD\n", "choice_label"), "D")

    def test_problem_id_is_stable(self) -> None:
        self.assertEqual(
            stable_problem_id("train", 2, "question"),
            stable_problem_id("train", 2, "question"),
        )
        self.assertTrue(stable_problem_id("train", 2, "question", "MATH/algebra").startswith("math-algebra-"))

    def test_answer_margin_uses_first_divergence_after_shared_prefix(self) -> None:
        class Tokenizer:
            def __call__(self, text, add_special_tokens=False):
                mapping = {" 12": [99, 1, 2], " 13": [99, 1, 3], " 11": [99, 1, 1]}
                return {"input_ids": mapping.get(text, [99, 8])}

            def __len__(self):
                return 128

        shared, positive, negative, _candidate = first_distinct_answer_tokens(
            Tokenizer(), "12"
        )
        self.assertEqual(shared, [99, 1])
        self.assertEqual(positive, 2)
        self.assertNotEqual(positive, negative)

    def test_answer_margin_fallback_avoids_special_tokens(self) -> None:
        class Tokenizer:
            all_special_ids = [6]

            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": [5]}

            def __len__(self):
                return 8

            def decode(self, token_ids, **_kwargs):
                return {6: "<special>", 7: "safe"}.get(token_ids[0], "x")

        shared, positive, negative, candidate = first_distinct_answer_tokens(
            Tokenizer(), "12"
        )
        self.assertEqual(shared, [])
        self.assertEqual(positive, 5)
        self.assertEqual(negative, 7)
        self.assertEqual(candidate, "VOCABULARY_NON_SPECIAL_FALLBACK")


if __name__ == "__main__":
    unittest.main()
