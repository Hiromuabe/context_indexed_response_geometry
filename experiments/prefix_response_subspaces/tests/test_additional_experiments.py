import unittest
import tempfile
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.prefix_response_subspaces.analyze_rank_saturation import _batched_rank_ev, _prefix_r90, _split_indices
from experiments.prefix_response_subspaces.analyze_optimal_value_control import full_span_energy_fraction, optimal_basis
from experiments.prefix_response_subspaces.analyze_paper_functional import summarize_absolute_recovery
from experiments.prefix_response_subspaces.analyze_control_rank_sensitivity import (
    _sample_eigensystem,
    _sample_projection_curve,
    main as control_rank_main,
    mean_shift_energy_fraction,
    select_ev_matching_rank,
)
from experiments.prefix_response_subspaces.analyze_ev_matched_functional import cached_local_basis, summarize_ev_matched_cells
from experiments.prefix_response_subspaces.verify_rank0_outliers import select_outlier_cells
from experiments.prefix_response_subspaces.compute_contrast_residuals import _chunked_double_center_to_npy
from experiments.prefix_response_subspaces.src.residualization import center_train_and_evaluation, double_center, inductive_center
from experiments.prefix_response_subspaces.src.storage import load_residual_entry
from experiments.prefix_response_subspaces.src.subspaces import explained_variance, top_svd
from experiments.prefix_response_subspaces.src.subspaces import mean_projection_eigensystem, randomized_mean_projection_eigensystem, remove_shared_subspace


class AdditionalExperimentTest(unittest.TestCase):
    def test_inductive_center_matches_training_mean_formula(self):
        rng = np.random.default_rng(109)
        evaluation = rng.normal(size=(3, 7, 5)).astype(np.float32)
        auxiliary = rng.normal(size=(4, 7, 5)).astype(np.float32)
        train = np.asarray([0, 2, 4, 6])
        heldout = np.asarray([1, 3, 5])
        observed = inductive_center(evaluation, auxiliary, train, heldout)
        expected = (
            evaluation[:, heldout]
            - evaluation[:, train].mean(axis=1)[:, None]
            - auxiliary[:, heldout].mean(axis=0)[None]
            + auxiliary[:, train].mean(axis=(0, 1))[None, None]
        )
        np.testing.assert_allclose(observed.residuals, expected, rtol=0, atol=2e-6)

    def test_inductive_center_does_not_couple_heldout_target_tokens(self):
        rng = np.random.default_rng(111)
        evaluation = rng.normal(size=(2, 6, 4)).astype(np.float32)
        auxiliary = rng.normal(size=(3, 6, 4)).astype(np.float32)
        train = [0, 1, 2]
        heldout = [3, 4, 5]
        before = inductive_center(evaluation, auxiliary, train, heldout).residuals
        changed = evaluation.copy()
        changed[0, 5] += 100.0
        after = inductive_center(changed, auxiliary, train, heldout).residuals
        np.testing.assert_allclose(before[0, :2], after[0, :2], rtol=0, atol=0)
        self.assertGreater(float(np.abs(after[0, 2] - before[0, 2]).max()), 10.0)

    def test_ev_matching_rank_uses_development_only_and_global_problem_means(self):
        rows = []
        for problem, target, controls in (
            ("a", 0.8, {64: 0.4, 80: 0.7}),
            ("b", 1.0, {64: 0.5, 80: 1.0}),
        ):
            for rank, control in controls.items():
                rows.append({"problem_id": problem, "split": "development", "control": "matched_common", "control_rank": rank, "ev_target_rank64": target, "ev_control": control})
        rows.append({"problem_id": "evaluation", "split": "evaluation", "control": "matched_common", "control_rank": 64, "ev_target_rank64": 0.0, "ev_control": 100.0})
        selected = select_ev_matching_rank(rows, "matched_common", [64, 80])
        self.assertEqual(selected["selected_rank"], 80)
        self.assertAlmostEqual(selected["selected"]["target_rank64_mean_ev"], 0.9)
        self.assertAlmostEqual(selected["selected"]["control_mean_ev"], 0.85)

    def test_ev_matched_summary_uses_positive_control_minus_target_advantage(self):
        rows = [
            {"problem_id": "a", "D_rank0": 1.0, "D_target": 0.2, "D_matched_common": 0.4, "D_wrong_mean": 0.5, "G_target": 0.8, "G_matched_common": 0.6, "G_wrong_mean": 0.5},
            {"problem_id": "b", "D_rank0": 2.0, "D_target": 0.5, "D_matched_common": 0.8, "D_wrong_mean": 1.0, "G_target": 1.5, "G_matched_common": 1.2, "G_wrong_mean": 1.0},
        ]
        config = {"seed": 5, "statistics": {"bootstrap_replicates": 100, "ci": 0.95}}
        summary = summarize_ev_matched_cells(rows, config)
        self.assertAlmostEqual(summary["target_advantage_vs_matched_common"]["mean"], 0.25)
        self.assertAlmostEqual(summary["target_advantage_vs_wrong_context"]["mean"], 0.4)

    def test_mean_shift_energy_fraction_is_candidate_constant_energy_share(self):
        values = np.asarray([[3.0, 1.0], [3.0, -1.0]])
        # Constant component energy is 2 * 3^2; total energy is 20.
        self.assertAlmostEqual(mean_shift_energy_fraction(values), 18.0 / 20.0)

    def test_sample_space_projection_curve_matches_explicit_basis(self):
        rng = np.random.default_rng(113)
        train = rng.normal(size=(10, 17))
        train -= train.mean(axis=0, keepdims=True)
        target = rng.normal(size=(6, 17))
        system = _sample_eigensystem(train, 8)
        curve = _sample_projection_curve(target, train, system, [2, 5, 8])
        basis = top_svd(train, 8)
        for rank in (2, 5, 8):
            self.assertAlmostEqual(curve[rank], explained_variance(target, basis[:, :rank]), places=8)

    def test_cached_sample_space_svd_reconstructs_nested_hidden_basis(self):
        rng = np.random.default_rng(117)
        train = rng.normal(size=(12, 19))
        train -= train.mean(axis=0, keepdims=True)
        left, singular = _sample_eigensystem(train, 9)
        cache = {
            "local_positions": np.asarray([3]),
            "local_left_singular_vectors": left[None].astype(np.float32),
            "local_singular_values": singular[None],
            "local_effective_ranks": np.asarray([len(singular)]),
        }
        observed = cached_local_basis(cache, np.stack([train] * 4), 3, 7)
        expected = top_svd(train, 7)
        self.assertAlmostEqual(float(np.square(observed.T @ expected).sum()), 7.0, places=5)

    def test_control_rank_stage_runs_from_saved_states_without_model_forward(self):
        rng = np.random.default_rng(127)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("manifests", "metrics", "controls", "hidden_states", "residuals"):
                (root / name).mkdir(parents=True, exist_ok=True)
            groups = ["auxiliary", "auxiliary"] + ["analysis_train"] * 5 + ["analysis_dev", "analysis_test"]
            prefixes = [
                {"prefix_id": f"p{i}", "problem_id": f"problem-{i}", "problem_group": group, "prefix_length_bin": 0, "reasoning_progress_bin": 0}
                for i, group in enumerate(groups)
            ]
            prefix_path = root / "prefixes.jsonl"
            prefix_path.write_text("".join(json.dumps(row) + "\n" for row in prefixes), encoding="utf-8")
            z = rng.normal(size=(len(prefixes), 200, 200)).astype(np.float32)
            successor_path = root / "hidden_states/layer_0.npy"
            np.save(successor_path, z)
            hidden = {"prefix_snapshot": str(prefix_path), "layers": [{"layer": 0, "successor_path": str(successor_path)}], "model": {"checkpoint": "synthetic", "hidden_size": 200}}
            (root / "manifests/hidden_states.json").write_text(json.dumps(hidden), encoding="utf-8")
            nonaux = np.arange(2, len(prefixes))
            train_tokens = np.arange(192)
            evaluation_tokens = np.arange(192, 200)
            train, heldout = center_train_and_evaluation(z[nonaux], z[:2], train_tokens, evaluation_tokens)
            residual_path = root / "residuals/layer_0_fold_0.npz"
            np.savez(
                residual_path,
                train_residuals=train.residuals,
                evaluation_residuals=heldout.residuals,
                nonauxiliary_prefix_indices=nonaux,
                train_candidate_indices=train_tokens,
                evaluation_candidate_indices=evaluation_tokens,
            )
            residual_manifest = {"entries": [{"layer": 0, "fold": 0, "path": str(residual_path)}]}
            (root / "manifests/residuals.json").write_text(json.dumps(residual_manifest), encoding="utf-8")
            (root / "metrics/paper_geometry_summary.json").write_text(json.dumps({"selected_layer": 0, "selected_rank": 64}), encoding="utf-8")
            donors = [f"p{i}" for i in range(2, 7)]
            wrong = [
                {"prefix_id": "p7", "wrong_prefix_ids": donors, "relaxed_length_wrong_prefixes": 0},
                {"prefix_id": "p8", "wrong_prefix_ids": donors, "relaxed_length_wrong_prefixes": 0},
            ]
            (root / "controls/wrong_prefixes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in wrong), encoding="utf-8")
            config = {
                "seed": 0,
                "statistics": {"bootstrap_replicates": 20, "ci": 0.95},
                "controls": {"wrong_prefixes_per_target": 5},
                "results_root": str(root),
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with patch.object(sys, "argv", ["analyze_control_rank_sensitivity", "--config", str(config_path)]):
                control_rank_main()
            summary = json.loads((root / "metrics/control_rank_sensitivity_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["coverage"]["evaluation_problems"], 1)
            self.assertEqual(summary["coverage"]["rank_curve_rows"], 6)
            self.assertEqual(summary["coverage"]["inductive_rows"], 1)
            self.assertIn(summary["ev_matched_rank_selection"]["matched_common"]["selected_rank"], range(64, 192))
            achieved = summary["ev_matched_rank_selection"]["evaluation_achieved_match"]["matched_common"]
            self.assertIn("relative_target_minus_control_ev", achieved)
            self.assertEqual(achieved["population"], "exact-bin evaluation problems")
            self.assertIn("maximum_control_rank_reason", summary["ev_matched_rank_selection"]["definition"])
            self.assertTrue((root / "subspaces/control_rank_cache/fold_0.npz").is_file())
            inductive = json.loads((root / "metrics/inductive_centering_summary.json").read_text(encoding="utf-8"))
            self.assertIn("mean_shift_energy_fraction_rho", inductive)
            self.assertIn("rho_definition", inductive)
            self.assertIn("paired_inductive_minus_primary", inductive)
            with patch.object(sys, "argv", ["analyze_control_rank_sensitivity", "--config", str(config_path), "--summary-only"]):
                control_rank_main()

    def test_chunked_residualization_matches_in_memory_definition(self):
        rng = np.random.default_rng(107)
        z = rng.normal(size=(9, 11, 7)).astype(np.float16)
        evaluation = np.asarray([0, 2, 4, 6, 8])
        auxiliary = np.asarray([1, 3, 5, 7])
        tokens = np.asarray([0, 3, 5, 9])
        expected = double_center(z[evaluation], z[auxiliary], tokens).residuals
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "residuals.npy"
            maximum, shape = _chunked_double_center_to_npy(z, evaluation, auxiliary, tokens, path, 2)
            observed = np.load(path)
        self.assertEqual(shape, list(expected.shape))
        self.assertLess(maximum, 2e-5)
        np.testing.assert_allclose(observed, expected, rtol=0, atol=2e-6)

    def test_npy_residual_bundle_loads_as_memmaps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
            heldout = train[:, :1]
            np.save(root / "train.npy", train); np.save(root / "heldout.npy", heldout)
            bundle = load_residual_entry({
                "storage_format": "npy_bundle", "train_path": str(root / "train.npy"),
                "evaluation_path": str(root / "heldout.npy"),
                "nonauxiliary_prefix_indices": [2, 4], "train_candidate_indices": [0, 1, 2],
                "evaluation_candidate_indices": [3],
            })
            self.assertIsInstance(bundle["train_residuals"], np.memmap)
            np.testing.assert_array_equal(bundle["evaluation_residuals"], heldout)

    def test_optimal_value_basis_equals_local_when_span_is_full(self):
        rng = np.random.default_rng(101)
        train = rng.normal(size=(12, 20))
        basis, rank, identical = optimal_basis(train, None, 6)
        expected = top_svd(train, 6)
        self.assertTrue(identical)
        self.assertEqual(rank, 6)
        self.assertAlmostEqual(float(np.square(basis.T @ expected).sum()), 6.0, places=8)

    def test_optimal_value_basis_stays_inside_restricted_span(self):
        rng = np.random.default_rng(103)
        train = rng.normal(size=(20, 10))
        span = np.eye(10)[:, :4]
        basis, rank, identical = optimal_basis(train, span, 3)
        self.assertFalse(identical)
        self.assertEqual(rank, 3)
        self.assertAlmostEqual(float(np.square(basis[4:]).sum()), 0.0, places=12)

    def test_full_span_energy_fraction_quantifies_retained_interaction(self):
        target = np.asarray([[3.0, 4.0, 0.0], [0.0, 0.0, 5.0]])
        span = np.eye(3)[:, :2]
        self.assertAlmostEqual(full_span_energy_fraction(target, span), 0.5, places=12)
        self.assertAlmostEqual(full_span_energy_fraction(target, None), 1.0, places=12)

    def test_functional_scale_summary_reports_absolute_and_ratio_of_totals(self):
        rows = [
            {"problem_id":"a","D_oracle":0.0,"D_rank0":1.0,"D_local":0.25,"D_conditional_global":0.5,"D_wrong_mean":0.75},
            {"problem_id":"b","D_oracle":0.0,"D_rank0":3.0,"D_local":1.5,"D_conditional_global":2.0,"D_wrong_mean":2.5},
        ]
        config={"seed":3,"statistics":{"bootstrap_replicates":100,"ci":0.95}}
        summary=summarize_absolute_recovery(rows,config)
        self.assertAlmostEqual(summary["D_rank0"]["mean"],2.0,places=12)
        self.assertAlmostEqual(summary["recovery_fraction_local"]["mean"],2.25/4.0,places=12)

    def test_rank0_outlier_selection_uses_paired_absolute_difference(self):
        rows = []
        for prefix, difference in (("a", 0.1), ("b", 0.3), ("c", 0.2)):
            rows.extend([
                {"prefix_id": prefix, "fold": 0, "candidate_index": 1, "condition": "Rank-0-reference", "js": 1.0},
                {"prefix_id": prefix, "fold": 0, "candidate_index": 1, "condition": "Rank-0-M64", "js": 1.0 + difference},
            ])
        selected = select_outlier_cells(rows, 64, 2)
        self.assertEqual([item[1][0] for item in selected], ["b", "c"])

    def test_rank_saturation_split_is_disjoint_and_complete(self):
        indices = np.arange(256)
        train, heldout = _split_indices(indices, 128, np.random.default_rng(7))
        self.assertEqual(len(train), 128)
        self.assertEqual(len(heldout), 128)
        self.assertFalse(set(train) & set(heldout))
        self.assertEqual(set(np.concatenate((train, heldout))), set(indices))

    def test_r90_uses_rank127_reference(self):
        curve = {1: 0.1, 32: 0.7, 64: 0.91, 96: 0.98, 127: 1.0}
        self.assertEqual(_prefix_r90(curve, [1, 32, 64, 96, 127], 127, 0.9), 64)

    def test_batched_rank_ev_matches_explicit_svd_projection(self):
        rng = np.random.default_rng(17)
        train = rng.normal(size=(3, 12, 20)).astype(np.float32)
        heldout = rng.normal(size=(3, 9, 20)).astype(np.float32)
        curves = _batched_rank_ev(train, heldout, [1, 4, 8], 8)
        for prefix in range(3):
            basis = top_svd(train[prefix], 8)
            for rank in (1, 4, 8):
                self.assertAlmostEqual(curves[rank][prefix], explained_variance(heldout[prefix], basis[:, :rank]), places=5)

    def test_mean_projector_eigenvalues_encode_shared_direction(self):
        identity = np.eye(4)
        bases = [identity[:, [0, 1]], identity[:, [0, 2]], identity[:, [0, 3]]]
        eigenvalues, eigenvectors = mean_projection_eigensystem(bases)
        self.assertAlmostEqual(eigenvalues[0], 1.0, places=12)
        self.assertAlmostEqual(eigenvalues[1], 1 / 3, places=12)
        shared = eigenvectors[:, :1]
        for basis in bases:
            specific = remove_shared_subspace(basis, shared)
            self.assertEqual(specific.shape, (4, 1))
            self.assertAlmostEqual(float(np.square(shared.T @ specific).sum()), 0.0, places=12)

    def test_randomized_mean_projector_recovers_leading_shared_space(self):
        rng = np.random.default_rng(71)
        common = np.linalg.qr(rng.normal(size=(20, 3)))[0]
        bases = []
        for _ in range(12):
            extra = rng.normal(size=(20, 3))
            extra -= common @ (common.T @ extra)
            extra = np.linalg.qr(extra)[0][:, :3]
            bases.append(np.concatenate((common, extra), axis=1))
        exact_values, exact_vectors = mean_projection_eigensystem(bases)
        fast_values, fast_vectors = randomized_mean_projection_eigensystem(bases, 6, seed=3, oversampling=6, power_iterations=3)
        self.assertLess(float(np.max(np.abs(fast_values[:3] - exact_values[:3]))), 1e-5)
        self.assertAlmostEqual(float(np.square(fast_vectors[:, :3].T @ exact_vectors[:, :3]).sum()), 3.0, places=5)

    def test_remove_shared_subspace_preserves_unshared_basis(self):
        identity = np.eye(5)
        local = identity[:, :2]
        unrelated_shared = identity[:, 3:4]
        specific = remove_shared_subspace(local, unrelated_shared)
        overlap = np.square(local.T @ specific).sum()
        self.assertAlmostEqual(overlap, 2.0, places=12)


if __name__ == "__main__":
    unittest.main()
