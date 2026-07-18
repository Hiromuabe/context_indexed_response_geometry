from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from prefix_displacement.config import ConfigError, load_json_config, parse_split_config


class ConfigTest(unittest.TestCase):
    def test_production_config_refuses_unresolved_split(self) -> None:
        config = load_json_config("configs/task1_cache.json")
        with self.assertRaisesRegex(ConfigError, "split.seed is unresolved"):
            parse_split_config(config)

    def test_resolved_split_is_config_driven(self) -> None:
        payload = {
            "split": {
                "unit": "gsm8k_problem_id",
                "seed": 73,
                "ratios": {"train": 0.6, "dev": 0.2, "test": 0.2},
                "registry_path": "registry.json",
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            parsed = parse_split_config(load_json_config(path))
        self.assertEqual(parsed.seed, 73)
        self.assertEqual(parsed.train_ratio, 0.6)
        self.assertEqual(parsed.unit, "gsm8k_problem_id")


if __name__ == "__main__":
    unittest.main()
