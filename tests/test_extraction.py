from __future__ import annotations

import unittest

import torch

from prefix_displacement.extraction import (
    TrajectoryFormatError,
    assert_gathered_batch_order,
    collate_trajectories,
    normalize_trajectory_row,
)


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(character) % 31 for character in text]}


class ExtractionTest(unittest.TestCase):
    def test_collate_preserves_sample_and_metadata_order(self) -> None:
        tokenizer = FakeTokenizer()
        rows = []
        for index, length in enumerate((4, 6, 5)):
            rows.append(normalize_trajectory_row(
                {
                    "problem_id": f"p{index}", "trajectory_id": f"t{index}",
                    "input_ids": list(range(length)), "transition_positions": [1, 2],
                    "evaluation_position": length - 1, "positive_token_id": 3,
                    "negative_token_id": 4, "correctness": True,
                },
                tokenizer=tokenizer, sample_index=10 + index, max_sequence_length=20
            ))
        batch = collate_trajectories(rows, pad_token_id=99)
        self.assertEqual(batch["sample_index"].tolist(), [10, 11, 12])
        self.assertEqual([row["trajectory_id"] for row in batch["metadata"]], ["t0", "t1", "t2"])
        assert_gathered_batch_order(batch["sample_index"], torch.tensor([10, 11, 12]))

    def test_no_automatic_truncation(self) -> None:
        with self.assertRaisesRegex(TrajectoryFormatError, "automatic truncation is disabled"):
            normalize_trajectory_row(
                {
                    "problem_id": "p", "trajectory_id": "t", "input_ids": list(range(10)),
                    "transition_positions": [1], "evaluation_position": 9,
                    "positive_token_id": 1, "negative_token_id": 2, "correctness": False,
                },
                tokenizer=FakeTokenizer(), sample_index=0, max_sequence_length=5
            )


if __name__ == "__main__":
    unittest.main()
