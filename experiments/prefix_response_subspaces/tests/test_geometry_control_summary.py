from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.prefix_response_subspaces.analyze_paper_geometry import (
    _rotation_rows,
    _rotation_rows_for_ranks,
)
from experiments.prefix_response_subspaces.refresh_paper_geometry import refresh_multirank_rotation
from experiments.prefix_response_subspaces.summarize_geometry_controls import (
    _conditional_global_is_exact,
    _wrong_control_is_exact,
    _write_cross_dataset_curve_csv,
    _write_cross_dataset_curve_svg,
    _write_cross_dataset_distance_curve_svg,
    _write_cross_dataset_ev_curve_svg,
    summarize,
)


class GeometryControlSummaryTest(unittest.TestCase):
    def test_legacy_gsm8k_rows_recover_exact_bin_status(self) -> None:
        self.assertTrue(_conditional_global_is_exact({}))
        self.assertTrue(_conditional_global_is_exact({"conditional_global_length_bin_distance": "0"}))
        self.assertFalse(_conditional_global_is_exact({"conditional_global_length_bin_distance": "1"}))
        self.assertTrue(_wrong_control_is_exact({"prefix_id": "p0"}, {"p0": True}))
        self.assertFalse(_wrong_control_is_exact({"prefix_id": "p0"}, {"p0": False}))

    def test_multirank_rotation_reuses_maximum_basis_without_changing_max_rank(self) -> None:
        prefixes = [
            {"prefix_id": "train0", "problem_id": "train0", "problem_group": "analysis_train", "prefix_length_bin": 0, "reasoning_progress_bin": 0},
            {"prefix_id": "train1", "problem_id": "train1", "problem_group": "analysis_train", "prefix_length_bin": 0, "reasoning_progress_bin": 0},
            {"prefix_id": "test0", "problem_id": "test0", "problem_group": "analysis_test", "prefix_length_bin": 0, "reasoning_progress_bin": 0},
        ]
        residuals = np.random.default_rng(3).normal(size=(3, 12, 6))
        nonaux = np.arange(3, dtype=np.int64)
        wrong = {"test0": ["train0"]}
        single = _rotation_rows(residuals, prefixes, nonaux, wrong, 3, 0, 0, 0)
        multiple = _rotation_rows_for_ranks(residuals, prefixes, nonaux, wrong, [2, 3], 0, 0, 0)
        maximum = [row for row in multiple if row["rank"] == 3]
        self.assertEqual(len(multiple), 2)
        self.assertAlmostEqual(single[0]["R_within"], maximum[0]["R_within"])
        self.assertAlmostEqual(single[0]["R_between"], maximum[0]["R_between"])

    def test_refresh_multirank_rotation_uses_saved_residuals_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "results"
            for relative in ("metrics", "manifests", "controls"):
                (root / relative).mkdir(parents=True, exist_ok=True)
            prefixes = [
                {"prefix_id": "train0", "problem_id": "train0", "problem_group": "analysis_train", "prefix_length_bin": 0, "reasoning_progress_bin": 0},
                {"prefix_id": "train1", "problem_id": "train1", "problem_group": "analysis_train", "prefix_length_bin": 0, "reasoning_progress_bin": 0},
                {"prefix_id": "test0", "problem_id": "test0", "problem_group": "analysis_test", "prefix_length_bin": 0, "reasoning_progress_bin": 0},
            ]
            prefix_path = root / "prefixes.jsonl"
            prefix_path.write_text("".join(json.dumps(row) + "\n" for row in prefixes), encoding="utf-8")
            (root / "metrics/paper_geometry_summary.json").write_text(
                json.dumps({"selected_layer": 0, "selected_rank": 3}), encoding="utf-8",
            )
            (root / "manifests/hidden_states.json").write_text(
                json.dumps({"prefix_snapshot": str(prefix_path)}), encoding="utf-8",
            )
            (root / "controls/wrong_prefixes.jsonl").write_text(
                json.dumps({"prefix_id": "test0", "wrong_prefix_ids": ["train0"], "relaxed_length_wrong_prefixes": 0}) + "\n",
                encoding="utf-8",
            )
            residual_path = root / "residuals.npz"
            residuals = np.random.default_rng(5).normal(size=(3, 12, 6))
            np.savez(
                residual_path,
                train_residuals=residuals,
                evaluation_residuals=residuals[:, :4],
                nonauxiliary_prefix_indices=np.arange(3, dtype=np.int64),
                train_candidate_indices=np.arange(12, dtype=np.int64),
                evaluation_candidate_indices=np.arange(4, dtype=np.int64),
            )
            (root / "manifests/residuals.json").write_text(
                json.dumps({"entries": [{"layer": 0, "fold": 0, "path": str(residual_path)}]}),
                encoding="utf-8",
            )
            (root / "manifests/paper_geometry.json").write_text("{}", encoding="utf-8")
            config = {
                "seed": 0,
                "analysis": {"ranks": [2, 3]},
                "data": {"evaluation_prefixes": 1},
                "candidates": {"folds": 1},
            }
            output = refresh_multirank_rotation(config, root)
            with output.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([int(row["rank"]) for row in rows], [2, 3])
            self.assertTrue(all(row["wrong_control_exact_length_bin"] == "True" for row in rows))
            manifest = json.loads((root / "manifests/paper_geometry.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["multirank_rotation_refreshed_from_saved_residuals"])

    def test_reports_positive_signs_at_every_rank(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "results"
            metrics = root / "metrics"
            metrics.mkdir(parents=True)
            config = {
                "seed": 0,
                "model": {"checkpoint": "test-model"},
                "data": {"trajectories_jsonl": "test.jsonl"},
                "candidates": {"analysis": 8, "folds": 2},
                "controls": {"wrong_prefixes_per_target": 2},
                "analysis": {"ranks": [2, 4]},
                "statistics": {"bootstrap_replicates": 20, "ci": 0.95},
                "results_root": str(root),
            }
            config_path = Path(directory) / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            (metrics / "paper_geometry_summary.json").write_text(
                json.dumps({"selected_layer": 0, "rotation_distance_definition": "distance"}),
                encoding="utf-8",
            )
            ev_rows = []
            rotation_rows = []
            for rank in (2, 4):
                for problem in ("p0", "p1"):
                    ev_rows.append({
                        "problem_id": problem, "split": "evaluation", "layer": 0,
                        "rank": rank, "conditional_global_exact_length_bin": True,
                        "wrong_control_exact_length_bin": True, "wrong_prefix_count": 2,
                        "ev_local": 0.6, "ev_conditional_global": 0.4,
                        "ev_wrong_mean": 0.3,
                    })
                    rotation_rows.append({
                        "problem_id": problem, "layer": 0, "rank": rank,
                        "wrong_control_exact_length_bin": True, "wrong_prefix_count": 2,
                        "R_within": 0.1, "R_between": 0.5,
                        "R_between_minus_within": 0.4,
                    })
            for name, rows in (
                ("paper_geometry_rows.csv", ev_rows),
                ("paper_rotation_rank_rows.csv", rotation_rows),
            ):
                with (metrics / name).open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            output_path, result = summarize(str(config_path))
            self.assertTrue(output_path.is_file())
            self.assertEqual(len(result["ranks"]), 2)
            self.assertTrue(all(result["all_ranks_same_positive_sign"].values()))
            figure_path = Path(result["rank_curve_figure_svg"])
            self.assertTrue(figure_path.is_file())
            figure = figure_path.read_text(encoding="utf-8")
            self.assertIn("Target-versus-control specificity", figure)
            self.assertIn("Between minus within distance", figure)
            self.assertIn("development-selected rank 4", figure)
            self.assertIn('font-size="20"', figure)
            comparison = [
                {"dataset": "GSM8K", "model": "test-model", "layer": 0, "reports": result["ranks"]},
                {"dataset": "CommonsenseQA", "model": "test-model", "layer": 0, "reports": result["ranks"]},
            ]
            comparison_csv = metrics / "comparison.csv"
            comparison_svg = root / "figures/comparison.svg"
            _write_cross_dataset_curve_csv(comparison_csv, comparison)
            _write_cross_dataset_curve_svg(
                comparison_svg,
                comparison,
                fixed_rank=4,
                confidence_level=0.95,
                candidate_protocol=result["candidate_protocol"],
            )
            self.assertTrue(comparison_csv.is_file())
            combined_figure = comparison_svg.read_text(encoding="utf-8")
            self.assertIn("GSM8K", combined_figure)
            self.assertIn("CommonsenseQA", combined_figure)
            self.assertIn("rank 4", combined_figure)
            ev_comparison_svg = root / "figures/ev_comparison.svg"
            _write_cross_dataset_ev_curve_svg(
                ev_comparison_svg,
                comparison,
                fixed_rank=4,
                confidence_level=0.95,
                candidate_protocol=result["candidate_protocol"],
            )
            ev_figure = ev_comparison_svg.read_text(encoding="utf-8")
            self.assertIn("Target context", ev_figure)
            self.assertIn("Matched common", ev_figure)
            self.assertIn("Wrong context", ev_figure)
            distance_comparison_svg = root / "figures/distance_comparison.svg"
            _write_cross_dataset_distance_curve_svg(
                distance_comparison_svg,
                comparison,
                fixed_rank=4,
                confidence_level=0.95,
                candidate_protocol=result["candidate_protocol"],
            )
            distance_figure = distance_comparison_svg.read_text(encoding="utf-8")
            self.assertIn("Within context", distance_figure)
            self.assertIn("Between context", distance_figure)
            self.assertIn("Between minus within", distance_figure)


if __name__ == "__main__":
    unittest.main()
