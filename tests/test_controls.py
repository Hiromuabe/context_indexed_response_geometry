from __future__ import annotations

import unittest

from prefix_displacement.controls import build_wrong_basepoint_map, validate_wrong_basepoint_map


class ControlsTest(unittest.TestCase):
    def test_wrong_basepoints_always_use_different_problem(self) -> None:
        rows = [
            {
                "problem_id": f"p{index}", "current_token_id": 1, "next_token_id": 2,
                "absolute_position": 10, "boundary_class": "ordinary", "surprisal": 1.2,
            }
            for index in range(4)
        ]
        mapping, diagnostics = build_wrong_basepoint_map(
            rows, mode="conditional", seed=0
        )
        validate_wrong_basepoint_map(rows, mapping)
        self.assertEqual(diagnostics["matched"], 4)


if __name__ == "__main__":
    unittest.main()
