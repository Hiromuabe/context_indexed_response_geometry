from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from prefix_displacement.split_registry import (
    SplitLeakageError,
    SplitRatios,
    assert_rows_respect_registry,
    create_split_registry,
    load_split_registry,
    validate_split_registry,
    write_split_registry,
)


class SplitRegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.problem_ids = [f"gsm8k-{index:03d}" for index in range(20)]
        self.ratios = SplitRatios(train=0.6, dev=0.2, test=0.2)

    def test_deterministic_problem_level_split(self) -> None:
        first = create_split_registry(self.problem_ids, ratios=self.ratios, seed=17)
        second = create_split_registry(reversed(self.problem_ids), ratios=self.ratios, seed=17)
        self.assertEqual(first, second)
        self.assertEqual(first["counts"], {"train": 12, "dev": 4, "test": 4})
        validate_split_registry(first, expected_problem_ids=self.problem_ids)

    def test_all_trajectories_and_transitions_follow_problem_assignment(self) -> None:
        registry = create_split_registry(self.problem_ids, ratios=self.ratios, seed=17)
        rows = []
        for problem_id in self.problem_ids:
            split = registry["problem_to_split"][problem_id]
            for trajectory_index in range(2):
                trajectory_id = f"{problem_id}-trajectory-{trajectory_index}"
                for transition_index in range(3):
                    rows.append(
                        {
                            "problem_id": problem_id,
                            "trajectory_id": trajectory_id,
                            "transition_id": str(transition_index),
                            "split": split,
                        }
                    )
        assert_rows_respect_registry(rows, registry)

    def test_row_split_mismatch_fails_immediately(self) -> None:
        registry = create_split_registry(self.problem_ids, ratios=self.ratios, seed=17)
        problem_id = self.problem_ids[0]
        expected = registry["problem_to_split"][problem_id]
        wrong = next(split for split in ("train", "dev", "test") if split != expected)
        row = {
            "problem_id": problem_id,
            "trajectory_id": "trajectory-0",
            "transition_id": "transition-0",
            "split": wrong,
        }
        with self.assertRaisesRegex(SplitLeakageError, "expected"):
            assert_rows_respect_registry([row], registry)

    def test_problem_overlap_fails(self) -> None:
        registry = create_split_registry(self.problem_ids, ratios=self.ratios, seed=17)
        leaked = {**registry, "splits": {key: list(value) for key, value in registry["splits"].items()}}
        leaked["splits"]["dev"].append(leaked["splits"]["train"][0])
        with self.assertRaisesRegex(SplitLeakageError, "leakage"):
            validate_split_registry(leaked)

    def test_trajectory_ids_may_be_problem_scoped(self) -> None:
        registry = create_split_registry(self.problem_ids, ratios=self.ratios, seed=17)
        rows = []
        for problem_id in self.problem_ids[:2]:
            rows.append(
                {
                    "problem_id": problem_id,
                    "trajectory_id": "shared-trajectory",
                    "transition_id": problem_id,
                    "split": registry["problem_to_split"][problem_id],
                }
            )
        assert_rows_respect_registry(rows, registry)

    def test_registry_is_immutable(self) -> None:
        registry = create_split_registry(self.problem_ids, ratios=self.ratios, seed=17)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            write_split_registry(registry, path)
            self.assertEqual(load_split_registry(path), registry)
            with self.assertRaises(FileExistsError):
                write_split_registry(registry, path)


if __name__ == "__main__":
    unittest.main()
