from __future__ import annotations

import unittest

import torch

from prefix_displacement.runtime import load_model_state, state_dict_without_parallel_prefix


class RuntimeTest(unittest.TestCase):
    def test_module_prefix_is_accepted(self) -> None:
        source = torch.nn.Linear(3, 2)
        target = torch.nn.Linear(3, 2)
        prefixed = {f"module.{key}": value.clone() for key, value in source.state_dict().items()}
        load_model_state(target, prefixed)
        for left, right in zip(source.parameters(), target.parameters()):
            torch.testing.assert_close(left, right)

    def test_unwrapped_state_dict_has_no_module_prefix(self) -> None:
        model = torch.nn.DataParallel(torch.nn.Linear(3, 2))
        self.assertTrue(all(not key.startswith("module.") for key in state_dict_without_parallel_prefix(model)))


if __name__ == "__main__":
    unittest.main()
