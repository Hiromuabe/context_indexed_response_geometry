from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from prefix_displacement.model_loading import resolve_model_source


class ModelLoadingTest(unittest.TestCase):
    def test_cli_local_path_overrides_hub_id_and_revision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, kwargs = resolve_model_source(
                {"checkpoint": "hub/model", "revision": "main"}, directory
            )
        self.assertEqual(source, directory)
        self.assertNotIn("revision", kwargs)
        self.assertTrue(kwargs["local_files_only"])

    def test_environment_override(self) -> None:
        with patch.dict(os.environ, {"RESPONSE_GEOMETRY_MODEL_PATH": "/models/qwen"}):
            source, _kwargs = resolve_model_source(
                {"checkpoint": "hub/model", "revision": "main"}
            )
        self.assertEqual(source, "/models/qwen")


if __name__ == "__main__":
    unittest.main()
