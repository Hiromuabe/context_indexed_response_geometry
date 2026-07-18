import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from experiments.prefix_response_subspaces.analyze_shared_dictionary_controls import (
    main as shared_dictionary_main,
    parser as shared_dictionary_parser,
)
from experiments.prefix_response_subspaces.src.shared_dictionary import (
    OptimizationConfig,
    _make_data_parallel_objective,
    covariance_reconstruction_errors,
    covariance_subspace,
    estimated_covariance,
    fit_shared_covariance_model,
)
from experiments.prefix_response_subspaces.src.subspaces import explained_variance, top_svd


torch.set_num_threads(1)


def _samples_from_dictionary(rng, basis, weights, count, noise=0.0):
    rows = []
    for values in weights:
        latent = rng.normal(size=(count, basis.shape[1])) * np.sqrt(values)[None]
        rows.append(latent @ basis.T + noise * rng.normal(size=(count, basis.shape[0])))
    return np.asarray(rows, dtype=np.float64)


class SharedDictionaryControlTest(unittest.TestCase):
    def _config(self, **updates):
        values = dict(
            learning_rate=0.04, maximum_steps=500, patience=80, restarts=3,
            seed=17, epsilon=1e-12, improvement_tolerance=1e-10,
            initialization_noise=0.03, coherence_penalty=0.0,
        )
        values.update(updates)
        return OptimizationConfig(**values)

    def test_shared_cpc_recovers_basis_and_target_ev(self):
        rng = np.random.default_rng(3)
        basis = np.linalg.qr(rng.normal(size=(8, 3)))[0]
        weights = np.asarray([
            [5.0, 2.0, 0.6], [1.0, 4.0, 0.8], [3.0, 1.0, 2.5],
            [2.0, 5.0, 1.0], [4.0, 0.8, 3.0],
        ])
        train = _samples_from_dictionary(rng, basis, weights, 128, noise=0.03)
        evaluation = _samples_from_dictionary(rng, basis, weights, 96, noise=0.03)
        fit = fit_shared_covariance_model(
            torch.as_tensor(train), 3, kind="cpc", config=self._config(),
        )
        overlap = np.square(fit.basis.detach().numpy().T @ basis).sum()
        self.assertGreater(overlap, 2.9)
        cpc_ev, target_ev = [], []
        for index in range(len(train)):
            shared, _ = covariance_subspace(fit.basis, fit.weights[index], 2, kind="cpc")
            cpc_ev.append(explained_variance(evaluation[index], shared.detach().numpy()))
            target_ev.append(explained_variance(evaluation[index], top_svd(train[index], 2)))
        self.assertLess(float(np.mean(target_ev) - np.mean(cpc_ev)), 0.03)

    def test_nonorthogonal_dictionary_beats_cpc_reconstruction(self):
        rng = np.random.default_rng(7)
        basis = np.asarray([
            [1.0, 0.75, 0.10], [0.0, 0.66, 0.70], [0.0, 0.0, 0.70],
            [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
        ])
        basis /= np.linalg.norm(basis, axis=0, keepdims=True)
        weights = np.asarray([
            [6.0, 0.4, 0.2], [0.3, 5.0, 0.4], [0.2, 0.5, 6.0],
            [4.0, 3.0, 0.3], [0.4, 4.0, 3.0], [3.0, 0.3, 4.0],
        ])
        train = torch.as_tensor(_samples_from_dictionary(rng, basis, weights, 384, noise=0.005))
        cpc = fit_shared_covariance_model(train, 3, kind="cpc", config=self._config(maximum_steps=700))
        dictionary = fit_shared_covariance_model(
            train, 3, kind="dictionary", config=self._config(maximum_steps=700, coherence_penalty=0.0),
        )
        cpc_error = float(covariance_reconstruction_errors(train, cpc.basis, cpc.weights, kind="cpc").mean())
        dictionary_error = float(covariance_reconstruction_errors(
            train, dictionary.basis, dictionary.weights, kind="dictionary",
        ).mean())
        self.assertLess(dictionary_error, cpc_error - 0.01)

    def test_context_specific_rotations_favor_target_pca(self):
        rng = np.random.default_rng(11)
        train, evaluation = [], []
        for _ in range(7):
            basis = np.linalg.qr(rng.normal(size=(10, 2)))[0]
            train.append(_samples_from_dictionary(rng, basis, [[5.0, 2.0]], 128, noise=0.02)[0])
            evaluation.append(_samples_from_dictionary(rng, basis, [[5.0, 2.0]], 96, noise=0.02)[0])
        train = np.asarray(train)
        fit = fit_shared_covariance_model(
            torch.as_tensor(train), 2, kind="dictionary", config=self._config(maximum_steps=600),
        )
        target_ev, shared_ev = [], []
        for index in range(len(train)):
            target_ev.append(explained_variance(evaluation[index], top_svd(train[index], 2)))
            shared, _ = covariance_subspace(fit.basis, fit.weights[index], 2, kind="dictionary")
            shared_ev.append(explained_variance(evaluation[index], shared.detach().numpy()))
        self.assertGreater(float(np.mean(target_ev) - np.mean(shared_ev)), 0.35)

    def test_evaluation_changes_cannot_change_training_fit(self):
        rng = np.random.default_rng(13)
        train = torch.as_tensor(rng.normal(size=(4, 64, 7)))
        evaluation = rng.normal(size=(4, 20, 7))
        config = self._config(maximum_steps=250, restarts=2)
        first = fit_shared_covariance_model(train, 3, kind="cpc", config=config)
        evaluation[0] *= 1e6
        second = fit_shared_covariance_model(train, 3, kind="cpc", config=config)
        np.testing.assert_array_equal(first.basis.detach().numpy(), second.basis.detach().numpy())
        np.testing.assert_array_equal(first.weights.detach().numpy(), second.weights.detach().numpy())
        self.assertEqual(first.selected_restart, second.selected_restart)
        self.assertEqual(first.loss, second.loss)
        self.assertGreater(float(np.abs(evaluation[0]).mean()), 1e5)

    def test_cpc_and_dictionary_numerical_invariants(self):
        rng = np.random.default_rng(19)
        train = torch.as_tensor(rng.normal(size=(4, 80, 8)))
        for kind in ("cpc", "dictionary"):
            fit = fit_shared_covariance_model(
                train, 3, kind=kind, config=self._config(maximum_steps=250, restarts=2),
            )
            covariance = estimated_covariance(fit.basis, fit.weights[0])
            np.testing.assert_allclose(covariance.detach().numpy(), covariance.detach().numpy().T, atol=1e-12)
            self.assertGreaterEqual(float(torch.linalg.eigvalsh(covariance).min()), -1e-10)
            subspace, _ = covariance_subspace(fit.basis, fit.weights[0], 2, kind=kind)
            projector = subspace @ subspace.T
            np.testing.assert_allclose((projector @ projector).detach().numpy(), projector.detach().numpy(), atol=1e-10)
            if kind == "cpc":
                identity = torch.eye(3, dtype=fit.basis.dtype)
                self.assertLess(float(torch.linalg.matrix_norm(fit.basis.T @ fit.basis - identity)), 1e-10)

    def test_low_rank_loss_matches_explicit_covariance_formula(self):
        rng = np.random.default_rng(21)
        residuals = torch.as_tensor(rng.normal(size=(3, 11, 6)))
        for kind in ("cpc", "dictionary"):
            raw = torch.as_tensor(rng.normal(size=(6, 3)))
            basis = torch.linalg.qr(raw).Q if kind == "cpc" else raw / torch.linalg.vector_norm(raw, dim=0)
            weights = torch.as_tensor(rng.uniform(0.1, 2.0, size=(3, 3)))
            observed = covariance_reconstruction_errors(residuals, basis, weights, kind=kind).detach().numpy()
            expected = []
            for index in range(3):
                covariance = residuals[index].T @ residuals[index] / residuals.shape[1]
                estimate = basis @ torch.diag(weights[index]) @ basis.T
                expected.append(float(torch.square(covariance - estimate).sum() / torch.square(covariance).sum()))
            np.testing.assert_allclose(observed, expected, rtol=1e-11, atol=1e-12)

    def test_data_parallel_objective_has_shared_finite_gradients(self):
        rng = np.random.default_rng(22)
        residuals = torch.as_tensor(rng.normal(size=(4, 12, 6)))
        raw_basis = torch.as_tensor(rng.normal(size=(6, 3)))
        raw_weights = torch.as_tensor(rng.normal(size=(4, 3)))
        objective = _make_data_parallel_objective(
            raw_basis, raw_weights, "dictionary", self._config(coherence_penalty=1e-4),
        )
        indices = torch.arange(4)
        covariance_norm = torch.square(residuals @ residuals.transpose(1, 2) / residuals.shape[1]).sum((1, 2))
        losses = objective(residuals, indices, covariance_norm)
        self.assertEqual(tuple(losses.shape), (4,))
        losses.mean().backward()
        self.assertTrue(torch.isfinite(objective.raw_basis.grad).all())
        self.assertTrue(torch.isfinite(objective.raw_weights.grad).all())

    def test_cli_accepts_container_visible_data_parallel_ids(self):
        arguments = shared_dictionary_parser().parse_args([
            "--config", "config.json", "--data-parallel-device-ids", "0", "1",
        ])
        self.assertEqual(arguments.data_parallel_device_ids, [0, 1])

        arguments = shared_dictionary_parser().parse_args([
            "--config", "config.json", "--restart-parallel-device-ids", "0", "1",
        ])
        self.assertEqual(arguments.restart_parallel_device_ids, [0, 1])

    def test_smoke_rebuilds_residuals_from_hidden_states_without_forward(self):
        rng = np.random.default_rng(23)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            for name in ("manifests", "candidate_tokens", "controls", "hidden_states", "metrics"):
                (source / name).mkdir(parents=True, exist_ok=True)
            prefixes = [
                {"prefix_id": "aux0", "problem_id": "aux0", "problem_group": "auxiliary", "prefix_length_bin": 0, "reasoning_progress_bin": 0},
                {"prefix_id": "aux1", "problem_id": "aux1", "problem_group": "auxiliary", "prefix_length_bin": 0, "reasoning_progress_bin": 0},
                {"prefix_id": "train0", "problem_id": "train0", "problem_group": "analysis_train", "prefix_length_bin": 0, "reasoning_progress_bin": 0},
                {"prefix_id": "train1", "problem_id": "train1", "problem_group": "analysis_train", "prefix_length_bin": 0, "reasoning_progress_bin": 0},
                {"prefix_id": "train2", "problem_id": "train2", "problem_group": "analysis_train", "prefix_length_bin": 0, "reasoning_progress_bin": 0},
                {"prefix_id": "target0", "problem_id": "target0", "problem_group": "analysis_test", "prefix_length_bin": 0, "reasoning_progress_bin": 0},
                {"prefix_id": "target1", "problem_id": "target1", "problem_group": "analysis_test", "prefix_length_bin": 0, "reasoning_progress_bin": 0},
            ]
            prefix_path = source / "hidden_states/prefixes.jsonl"
            prefix_path.write_text("".join(json.dumps(row) + "\n" for row in prefixes), encoding="utf-8")
            state_path = source / "hidden_states/layer_0.npy"
            np.save(state_path, rng.normal(size=(len(prefixes), 256, 6)).astype(np.float32))
            self._write_json(source / "manifests/hidden_states.json", {
                "prefix_snapshot": str(prefix_path), "layers": [{"layer": 0, "successor_path": str(state_path)}],
                "model": {"checkpoint": "synthetic", "revision": "synthetic", "hidden_size": 6},
            })
            candidates = {
                "candidate_token_ids": list(range(256)),
                "folds": [{"fold_id": 0, "train_indices": list(range(192)), "evaluation_indices": list(range(192, 256))}],
            }
            self._write_json(source / "candidate_tokens/candidate_tokens.json", candidates)
            wrong = [
                {"prefix_id": "target0", "wrong_prefix_ids": ["train0"]},
                {"prefix_id": "target1", "wrong_prefix_ids": ["train1"]},
            ]
            (source / "controls/wrong_prefixes.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in wrong), encoding="utf-8",
            )
            config = {
                "profile": "synthetic_smoke", "seed": 0, "results_root": str(source),
                "model": {"checkpoint": "Qwen/Qwen2.5-Math-1.5B", "revision": "main"},
                "candidates": {"folds": 1}, "analysis": {"ranks": [1]},
                "statistics": {"bootstrap_replicates": 20, "ci": 0.95},
            }
            config_path = root / "config.json"
            self._write_json(config_path, config)
            arguments = [
                "analyze_shared_dictionary_controls", "--config", str(config_path),
                "--source-results-root", str(source), "--output-root", str(output),
                "--layer", "0", "--dictionary-sizes", "2", "--evaluation-ranks", "1",
                "--maximum-steps", "20", "--patience", "5", "--restarts", "2",
                "--coherence-penalties", "0", "--device", "cpu", "--dtype", "float64", "--smoke",
                "--leave-one-context-out", "--loco-context-limit", "1",
            ]
            self.assertFalse((source / "manifests/residuals.json").exists())
            with patch.object(sys, "argv", arguments):
                shared_dictionary_main()
            self.assertTrue((source / "manifests/residuals.json").is_file())
            manifests = list(output.glob("run_*/manifest.json"))
            self.assertEqual(len(manifests), 1)
            result = json.loads(manifests[0].read_text(encoding="utf-8"))
            self.assertEqual(result["forward_passes_executed"], 0)
            for path in result["outputs"].values():
                self.assertTrue(Path(path).is_file())
            with Path(result["outputs"]["heldout_ev_all"]).open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(any(row["leave_one_context_out"] == "True" for row in rows))

            residual_manifest = json.loads((source / "manifests/residuals.json").read_text(encoding="utf-8"))
            residual_artifact = Path(residual_manifest["entries"][0]["path"])
            residual_artifact.unlink()
            heldout_artifact = Path(result["outputs"]["heldout_ev_all"])
            heldout_artifact.unlink()
            self.assertFalse(residual_artifact.exists())
            self.assertFalse(heldout_artifact.exists())
            with patch.object(sys, "argv", arguments):
                shared_dictionary_main()
            self.assertTrue(residual_artifact.is_file())
            self.assertTrue(heldout_artifact.is_file())

    @staticmethod
    def _write_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
