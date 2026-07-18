import csv
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.prefix_response_subspaces.prepare_shared_dictionary_artifacts import (
    REQUIRED_FILES,
    audit_run,
    main,
)


class SharedDictionaryArtifactTest(unittest.TestCase):
    def _fixture(self, root: Path) -> Path:
        run_dir = root / "run_test"
        run_dir.mkdir()
        run = {
            "dictionary_sizes": [2],
            "ranks": [1, 2],
            "coherence_penalties": [0.0],
            "leave_one_context_out": False,
        }
        settings = []
        for rank in (1, 2):
            settings.extend([
                ("target_context_pca", "", rank, "", False),
                ("matched_common", "", rank, "", False),
                ("wrong_context", "", rank, "", False),
                ("pooled_pca_context_selection", 2, rank, "", False),
                ("truncated_cpc", 2, rank, "", False),
                ("shared_nonorthogonal_dictionary", 2, rank, 0.0, False),
            ])
        summary_fields = [
            "method", "dictionary_size", "rank", "coherence_penalty", "leave_one_context_out",
            "heldout_ev_mean", "problem_count",
        ]
        with (run_dir / "paired_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=summary_fields)
            writer.writeheader()
            for method, size, rank, beta, loco in settings:
                writer.writerow({
                    "method": method, "dictionary_size": size, "rank": rank,
                    "coherence_penalty": beta, "leave_one_context_out": loco,
                    "heldout_ev_mean": 0.5, "problem_count": 2,
                })
        raw_fields = [
            "problem_id", "prefix_id", "fold", "method", "dictionary_size", "rank",
            "coherence_penalty", "leave_one_context_out", "heldout_ev",
        ]
        with (run_dir / "heldout_ev_all.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=raw_fields)
            writer.writeheader()
            for problem in ("p0", "p1"):
                for fold in (0, 1):
                    for method, size, rank, beta, loco in settings:
                        writer.writerow({
                            "problem_id": problem, "prefix_id": problem, "fold": fold,
                            "method": method, "dictionary_size": size, "rank": rank,
                            "coherence_penalty": beta, "leave_one_context_out": loco,
                            "heldout_ev": 0.5,
                        })
        diagnostics = {
            "run": run,
            "fit_diagnostics": [
                {
                    "fold": 0, "kind": "cpc", "dictionary_size": 2,
                    "optimization": {"maximum_steps": 20, "coherence_penalty": 0.0},
                    "selected_restart": 0, "orthogonality_error_frobenius": 1e-7,
                    "restarts": [
                        {"restart": 0, "status": "success", "steps_executed": 15},
                        {"restart": 1, "status": "failed", "failure_type": "RuntimeError", "failure": "test"},
                    ],
                },
                {
                    "fold": 0, "kind": "dictionary", "dictionary_size": 2,
                    "optimization": {"maximum_steps": 20, "coherence_penalty": 0.0},
                    "selected_restart": 1, "maximum_column_coherence": 0.25,
                    "restarts": [
                        {"restart": 0, "status": "success", "steps_executed": 20},
                        {"restart": 1, "status": "success", "steps_executed": 12},
                    ],
                },
            ],
        }
        (run_dir / "optimization_diagnostics.json").write_text(
            json.dumps(diagnostics), encoding="utf-8",
        )
        (run_dir / "config_resolved.json").write_text(
            json.dumps({"paper_config": {"candidates": {"folds": 2}}}), encoding="utf-8",
        )
        return run_dir

    def test_audit_passes_and_reports_requested_diagnostics(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = self._fixture(Path(temporary))
            audit = audit_run(run_dir)
            self.assertEqual(audit["status"], "PASS")
            self.assertEqual(audit["optimization_audit"]["failed_restart_count"], 1)
            self.assertEqual(audit["optimization_audit"]["maximum_step_restart_count"], 1)
            self.assertEqual(audit["optimization_audit"]["cpc_orthogonality_error_max"], 1e-7)
            self.assertEqual(audit["optimization_audit"]["dictionary_coherence_max"], 0.25)
            self.assertEqual(audit["heldout_ev_audit"]["missing_fold_count"], 0)
            self.assertEqual(audit["heldout_ev_audit"]["nonfinite_heldout_ev_count"], 0)

    def test_cli_archive_contains_only_required_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = self._fixture(Path(temporary))
            archive_path = run_dir / "review.tar.gz"
            with patch("sys.argv", [
                "prepare_shared_dictionary_artifacts", "--run-dir", str(run_dir),
                "--archive", str(archive_path),
            ]):
                with self.assertRaises(SystemExit) as exit_context:
                    main()
            self.assertEqual(exit_context.exception.code, 0)
            with tarfile.open(archive_path, "r:gz") as archive:
                self.assertEqual(sorted(archive.getnames()), sorted(REQUIRED_FILES))

    def test_nonfinite_ev_and_missing_fold_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = self._fixture(Path(temporary))
            path = run_dir / "heldout_ev_all.csv"
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
                fields = list(rows[0])
            rows[0]["heldout_ev"] = "nan"
            rows.pop()
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            audit = audit_run(run_dir)
            self.assertEqual(audit["status"], "FAIL")
            self.assertEqual(audit["heldout_ev_audit"]["nonfinite_heldout_ev_count"], 1)
            self.assertEqual(audit["heldout_ev_audit"]["missing_fold_count"], 1)


if __name__ == "__main__":
    unittest.main()
