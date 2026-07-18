import unittest
import importlib.util
import json
import os
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.prefix_response_subspaces.analyze_jacobian_alignment import scaled_linearization_error
from experiments.prefix_response_subspaces.analyze_candidate_distribution_transfer import main as candidate_transfer_main
from experiments.prefix_response_subspaces.analyze_context_controls import main as context_controls_main
from experiments.prefix_response_subspaces.analyze_jacobian_alignment import main as jacobian_alignment_main
from experiments.prefix_response_subspaces.analyze_subspace_stability import main as stability_main
from experiments.prefix_response_subspaces.analyze_subspace_stability import reliability_corrected_distance
from experiments.prefix_response_subspaces.analyze_response_law_state import (
    candidate_gram,
    double_center_hidden_profile,
    fixed_candidate_panel,
    gram_alignment,
    common_forced_tokens,
    partial_standardized_beta,
    response_matrix_cosine_distance,
    row_space_projection_distance,
)
from experiments.prefix_response_subspaces.extract_jacobian_responses import embedding_eigensystem
from experiments.prefix_response_subspaces.extract_context_control_states import balanced_fold_candidate_indices
from experiments.prefix_response_subspaces.run_review_experiments import formal_check_config, quick_check_config, temporary_model_download
from experiments.prefix_response_subspaces.show_review_results import collect_review_results, review_result_root
from experiments.prefix_response_subspaces.src.review_experiments import (
    auxiliary_token_statistics,
    build_context_control_records,
    center_context_block,
    load_review_tokenizer,
    review_token_category,
    select_candidates_from_logits,
    shuffled_context,
)
from experiments.prefix_response_subspaces.src.residualization import double_center
from experiments.prefix_response_subspaces.src.statistics import two_way_ratio_bootstrap
from experiments.prefix_response_subspaces.src.subspaces import normalized_projection_distance, principal_angle_cosines_squared
from experiments.prefix_response_subspaces.src.review_extraction import copy_precomputed_cells, precomputed_cell_mask


class _Tokenizer:
    all_special_ids = []

    def __len__(self):
        return 8

    def decode(self, token_ids, **_kwargs):
        return ["a", " 1", "+", "word", "!", " b", "7", "="][int(token_ids[0])]

    def batch_decode(self, rows, **kwargs):
        return [self.decode(row, **kwargs) for row in rows]


class ReviewExperimentTest(unittest.TestCase):
    @staticmethod
    def _write_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_review_categories_include_leading_whitespace(self):
        self.assertEqual(review_token_category(" 42"), "whitespace")
        self.assertEqual(review_token_category("42"), "number")
        self.assertEqual(review_token_category("+"), "operator")
        self.assertEqual(review_token_category("hello"), "word")

    def test_review_tokenizer_is_offline_first_and_fails_fast(self):
        calls = []

        class AutoTokenizer:
            @staticmethod
            def from_pretrained(source, **kwargs):
                calls.append((source, kwargs))
                raise OSError("not cached")

        config = {
            "model": {"checkpoint": "example/model", "revision": "main", "local_files_only": False},
            "review_experiments": {},
        }
        fake_transformers = types.SimpleNamespace(AutoTokenizer=AutoTokenizer)
        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            with self.assertRaisesRegex(RuntimeError, "--model-path"):
                load_review_tokenizer(config)
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0][1]["local_files_only"])

    def test_review_tokenizer_rejects_placeholder_model_path(self):
        config = {"model": {"checkpoint": "example/model"}, "review_experiments": {}}
        fake_transformers = types.SimpleNamespace(AutoTokenizer=object())
        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            with self.assertRaisesRegex(FileNotFoundError, "temporary-model-download"):
                load_review_tokenizer(config, "/absolute/path/to/Qwen2.5-Math-1.5B")

    def test_review_tokenizer_reuses_parent_manifest_local_model(self):
        calls = []
        tokenizer = object()

        class AutoTokenizer:
            @staticmethod
            def from_pretrained(source, **kwargs):
                calls.append((source, kwargs))
                return tokenizer

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            self._write_json(root / "manifests/candidate_tokens.json", {
                "model": {"model_source": str(model), "resolved_revision": "unused"},
            })
            config = {
                "model": {"checkpoint": "example/model", "revision": "main", "local_files_only": False},
                "review_experiments": {},
            }
            fake_transformers = types.SimpleNamespace(AutoTokenizer=AutoTokenizer)
            with patch.dict(sys.modules, {"transformers": fake_transformers}):
                observed, info = load_review_tokenizer(config, source_root=root)
        self.assertIs(observed, tokenizer)
        self.assertEqual(calls[0][0], str(model))
        self.assertTrue(calls[0][1]["local_files_only"])
        self.assertEqual(info["mode"], "parent_manifest:manifests/candidate_tokens.json")

    def test_temporary_model_download_uses_and_removes_isolated_cache(self):
        calls = []

        def snapshot_download(**kwargs):
            calls.append(kwargs)
            model_dir = Path(kwargs["local_dir"])
            model_dir.mkdir(parents=True)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            return str(model_dir)

        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            self._write_json(config_path, {
                "model": {"checkpoint": "example/model", "revision": "revision"},
                "review_experiments": {
                    "temporary_download_max_workers": 3,
                    "temporary_download_timeout_seconds": 45,
                },
            })
            fake_hub = types.SimpleNamespace(snapshot_download=snapshot_download)
            with patch.dict(sys.modules, {"huggingface_hub": fake_hub}), patch.dict(os.environ, {}, clear=False):
                with temporary_model_download(str(config_path)) as model_path:
                    model = Path(model_path)
                    temporary_root = model.parent
                    self.assertTrue((model / "config.json").is_file())
                    self.assertEqual(Path(calls[0]["cache_dir"]).parent, temporary_root)
                self.assertFalse(temporary_root.exists())
        self.assertEqual(calls[0]["repo_id"], "example/model")
        self.assertEqual(calls[0]["revision"], "revision")
        self.assertEqual(calls[0]["max_workers"], 3)

    def test_quick_check_config_is_small_and_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            self._write_json(config_path, {
                "profile": "full", "results_root": "results/full",
                "candidates": {"proposal_top_k": 1024},
                "statistics": {"bootstrap_replicates": 2000},
                "review_experiments": {},
            })
            with quick_check_config(str(config_path)) as quick_path:
                quick = json.loads(Path(quick_path).read_text(encoding="utf-8"))
                self.assertTrue(quick["quick_check"])
                self.assertEqual(quick["results_root"], "results/full_quick_check")
                self.assertEqual(quick["review_experiments"]["independent_candidate_set_size"], 64)
                self.assertEqual(quick["review_experiments"]["context_control_candidate_limit"], 64)
                self.assertEqual(quick["statistics"]["bootstrap_replicates"], 50)
            self.assertFalse(Path(quick_path).exists())

    def test_formal_check_config_equalizes_fits_and_bounds_jacobian(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            self._write_json(config_path, {
                "profile": "full", "results_root": "results/full",
                "candidates": {"proposal_top_k": 1024},
                "statistics": {"bootstrap_replicates": 2000},
                "review_experiments": {},
            })
            with formal_check_config(str(config_path)) as formal_path:
                formal = json.loads(Path(formal_path).read_text(encoding="utf-8"))
                review = formal["review_experiments"]
                self.assertTrue(formal["formal_check"])
                self.assertEqual(formal["results_root"], "results/full_formal_check")
                self.assertEqual(review["independent_candidate_set_size"], 160)
                self.assertTrue(review["candidate_transfer_equalize_fit_counts"])
                self.assertEqual(review["jacobian_target_contexts"], 4)
                self.assertEqual(formal["statistics"]["bootstrap_replicates"], 500)
            self.assertFalse(Path(formal_path).exists())

    def test_review_result_reader_resolves_quick_root_and_missing_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "review"
            quick = review_result_root({"results_root": str(root)}, True)
            self.assertEqual(quick, Path(f"{root}_quick_check"))
            formal = review_result_root({"results_root": str(root)}, False, True)
            self.assertEqual(formal, Path(f"{root}_formal_check"))
            self._write_json(quick / "metrics/context_control_summary.json", {"controls": {}})
            result = collect_review_results(quick)
            self.assertTrue(result["pilot_only"])
            self.assertIn("context_controls", result["available"])
            self.assertEqual(len(result["missing"]), 4)

    def test_response_law_distance_and_common_forced_tokens(self):
        matrix = np.asarray([[1.0, -1.0], [-2.0, 2.0]])
        self.assertAlmostEqual(response_matrix_cosine_distance(matrix, matrix), 0.0)
        self.assertAlmostEqual(response_matrix_cosine_distance(matrix, -matrix), 2.0)
        left = np.asarray([9.0, 8.0, 7.0, 1.0, 0.0])
        right = np.asarray([8.0, 9.0, 6.0, 2.0, 0.0])
        selected = common_forced_tokens(
            left, right, count=2, vocabulary_size=5, special_ids={0}, intersection_top_k=3,
        )
        self.assertEqual(selected, [1, 2])

    def test_response_geometry_partial_beta_controls_current_js(self):
        rng = np.random.default_rng(11)
        current = rng.normal(size=64)
        response = 0.7 * current + rng.normal(size=64)
        future = 0.5 * current + 1.2 * response + rng.normal(scale=0.1, size=64)
        rows = [
            {
                "current_normalized_js": float(current[index]),
                "response_subspace_distance": float(response[index]),
                "future_normalized_js": float(future[index]),
            }
            for index in range(len(current))
        ]
        self.assertGreater(partial_standardized_beta(rows, "response_subspace_distance"), 0.5)

    def test_contrast_geometry_centers_main_effects_and_preserves_alignment(self):
        context = np.asarray([[4.0, 1.0], [7.0, 3.0], [10.0, 8.0]])
        auxiliary = np.asarray([[1.0, 2.0], [2.0, 4.0], [4.0, 7.0]])
        residual = double_center_hidden_profile(context, auxiliary)
        np.testing.assert_allclose(residual.mean(axis=0), 0.0, atol=1e-12)
        gram = candidate_gram(residual)
        self.assertAlmostEqual(gram_alignment(gram, gram), 1.0)

    def test_row_space_distance_matches_explicit_svd_basis(self):
        rng = np.random.default_rng(31)
        left = rng.normal(size=(6, 12))
        right = rng.normal(size=(6, 12))
        observed = row_space_projection_distance(left, right, 3)
        from experiments.prefix_response_subspaces.src.subspaces import top_svd
        left_basis = top_svd(left - left.mean(axis=0), 3)
        right_basis = top_svd(right - right.mean(axis=0), 3)
        expected = normalized_projection_distance(left_basis, right_basis)
        self.assertAlmostEqual(observed, expected, places=10)

    def test_fixed_candidate_panel_uses_only_studied_candidates(self):
        logits = np.asarray([
            [0.0, 8.0, 3.0, 9.0, 2.0, 7.0],
            [0.0, 7.0, 4.0, 9.0, 1.0, 8.0],
        ])
        panel = fixed_candidate_panel(logits, [0, 1], [1, 2, 5], count=2, special_ids=set())
        self.assertEqual(panel, [1, 5])

    def test_independent_candidate_selector_uses_only_given_logits(self):
        logits = np.asarray([[5, 4, 3, 0, 0, 0, 0, 0], [4, 5, 3, 0, 0, 0, 0, 0]], dtype=np.float32)
        rows = select_candidates_from_logits(logits, tokenizer=_Tokenizer(), total=3, proposal_top_k=3)
        self.assertEqual({row["token_id"] for row in rows}, {0, 1, 2})
        by_id = {row["token_id"]: row for row in rows}
        self.assertEqual(by_id[0]["coverage"], 1.0)
        self.assertEqual(by_id[1]["coverage"], 1.0)
        self.assertEqual(by_id[2]["coverage"], 1.0)

    def test_candidate_selector_stops_decoding_after_required_categories(self):
        class CountingTokenizer:
            all_special_ids = []

            def __init__(self):
                self.decoded = 0

            def __len__(self):
                return 4096

            def batch_decode(self, rows, **_kwargs):
                self.decoded += len(rows)
                values = ["  x", "7", "+", "word", "!"]
                return [values[int(row[0]) % len(values)] for row in rows]

        tokenizer = CountingTokenizer()
        rng = np.random.default_rng(101)
        logits = rng.normal(size=(32, 4096)).astype(np.float32)
        rows = select_candidates_from_logits(
            logits, tokenizer=tokenizer, total=128, proposal_top_k=1024, decode_batch_size=256,
        )
        self.assertEqual(len(rows), 128)
        self.assertLessEqual(tokenizer.decoded, 256)

    def test_shuffled_context_preserves_endpoints_and_multiset(self):
        tokens = [1, 2, 3, 4, 5, 6]
        observed = shuffled_context(tokens, 7)
        self.assertEqual(observed[0], tokens[0])
        self.assertEqual(observed[-1], tokens[-1])
        self.assertEqual(sorted(observed), sorted(tokens))
        self.assertNotEqual(observed, tokens)

    def test_context_controls_include_all_requested_families(self):
        prefixes = [
            {"prefix_id": "aux", "problem_id": "pa", "problem_group": "auxiliary", "prefix_token_ids": [0, 1, 2, 3], "prefix_length": 4, "evaluation_suffix_token_ids": [4, 5]},
            {"prefix_id": "a", "problem_id": "p1", "problem_group": "analysis_test", "prefix_token_ids": [0, 1, 2, 3], "prefix_length": 4, "evaluation_suffix_token_ids": [4, 5, 6]},
            {"prefix_id": "b", "problem_id": "p2", "problem_group": "analysis_dev", "prefix_token_ids": [0, 5, 2, 3], "prefix_length": 4, "evaluation_suffix_token_ids": [4, 5, 6]},
        ]
        records, diagnostics = build_context_control_records(prefixes, lambda ids: "+" if 2 in ids else "none", seed=3, minimum_timepoint_gap=2)
        kinds = {row["control_type"] for row in records}
        self.assertIn("exact_length_random", kinds)
        self.assertIn("token_order_shuffled", kinds)
        self.assertIn("same_problem_timepoint", kinds)
        self.assertIn("operation_matched", kinds)
        self.assertEqual(diagnostics["target_count"], 2)

    def test_quick_context_candidates_are_balanced_across_folds(self):
        candidates = {
            "candidate_token_ids": list(range(20)),
            "folds": [
                {"evaluation_indices": list(range(0, 5))},
                {"evaluation_indices": list(range(5, 10))},
                {"evaluation_indices": list(range(10, 15))},
                {"evaluation_indices": list(range(15, 20))},
            ],
        }
        observed = balanced_fold_candidate_indices(candidates, 8)
        self.assertEqual(observed, [0, 1, 5, 6, 10, 11, 15, 16])

    def test_two_way_bootstrap_preserves_constant_ratio(self):
        denominator = np.arange(1, 13, dtype=np.float64).reshape(3, 4)
        result = two_way_ratio_bootstrap(
            denominator * 0.25, denominator, np.asarray(["a", "b", "c"]), np.asarray([1, 2, 3, 4]),
            replicates=100, seed=9,
        )
        self.assertAlmostEqual(result["mean"], 0.25)
        self.assertAlmostEqual(result["ci_low"], 0.25)
        self.assertAlmostEqual(result["ci_high"], 0.25)

    def test_chunked_single_context_centering_matches_double_center(self):
        rng = np.random.default_rng(19)
        states = rng.normal(size=(7, 11, 5)).astype(np.float32)
        auxiliary = np.asarray([0, 2, 4])
        tokens = [1, 3, 7, 9]
        statistics = auxiliary_token_statistics(states, auxiliary, tokens, chunk_size=2)
        observed = np.stack([center_context_block(states, index, tokens, *statistics) for index in (1, 5, 6)])
        expected = double_center(states[[1, 5, 6]], states[auxiliary], tokens).residuals
        np.testing.assert_allclose(observed, expected, rtol=0, atol=2e-6)

    def test_precomputed_grid_copy_reuses_only_axis_intersection(self):
        with tempfile.TemporaryDirectory() as directory:
            source = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
            source_path = Path(directory) / "source.npy"
            np.save(source_path, source)
            output = np.full((3, 3, 4), -1.0, dtype=np.float32)
            spec = {
                "context_indices": [1, -1, 0], "token_indices": [2, -1, 1],
                "layers": {0: str(source_path)}, "fingerprint": "test",
            }
            mask = precomputed_cell_mask(spec, 3, 3)
            self.assertEqual(int(mask.sum()), 4)
            copied = copy_precomputed_cells({0: output}, spec, context_chunk_size=1)
            self.assertEqual(copied, 4)
            np.testing.assert_array_equal(output[0, [0, 2]], source[1, [2, 1]])
            np.testing.assert_array_equal(output[2, [0, 2]], source[0, [2, 1]])
            self.assertTrue(np.all(output[1] == -1.0))

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch unavailable")
    def test_early_exit_endpoint_skips_deeper_decoder_blocks(self):
        import types
        import torch

        from experiments.prefix_response_subspaces.src.review_model import EarlyExitEndpointForward

        calls = [0, 0, 0]

        class Block(torch.nn.Module):
            def __init__(self, index):
                super().__init__()
                self.index = index

            def forward(self, hidden_states, **_kwargs):
                calls[self.index] += 1
                return hidden_states + 1.0

        class Decoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([Block(0), Block(1), Block(2)])

            def forward(self, input_ids, **_kwargs):
                hidden = input_ids.float().unsqueeze(-1)
                for layer in self.layers:
                    hidden = layer(hidden)
                return types.SimpleNamespace(last_hidden_state=hidden)

        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = Decoder()

        wrapped = EarlyExitEndpointForward.build(Backbone(), [1])
        endpoints, order = wrapped(
            torch.tensor([[1, 2, 3]]), torch.ones((1, 3), dtype=torch.long),
            torch.tensor([2]), torch.tensor([7]),
        )
        self.assertEqual(calls, [1, 1, 0])
        self.assertEqual(float(endpoints[0, 0, 0]), 5.0)
        self.assertEqual(int(order[0]), 7)

    def test_principal_angle_spectrum_and_distance(self):
        identity = np.eye(4)
        left = identity[:, :2]
        right = identity[:, [0, 2]]
        np.testing.assert_allclose(principal_angle_cosines_squared(left, right), [1.0, 0.0], atol=1e-12)
        self.assertAlmostEqual(normalized_projection_distance(left, right), 0.5)

    def test_embedding_eigensystem_respects_component_limit(self):
        rng = np.random.default_rng(3)
        left, singular, right = embedding_eigensystem(rng.normal(size=(8, 5)), 3)
        self.assertEqual(left.shape, (8, 3))
        self.assertEqual(singular.shape, (3,))
        self.assertEqual(right.shape, (3, 5))

    def test_scaled_linearization_removes_only_global_scale(self):
        predicted = np.asarray([[1.0, 2.0], [3.0, 4.0]])
        result = scaled_linearization_error(predicted * 2.5, predicted)
        self.assertAlmostEqual(result["optimal_scalar"], 2.5)
        self.assertAlmostEqual(result["relative_squared_error"], 0.0)
        self.assertAlmostEqual(result["matrix_cosine"], 1.0)

    def test_reliability_correction(self):
        self.assertAlmostEqual(reliability_corrected_distance(0.4, 0.1), 0.75)

    def test_candidate_transfer_stage_runs_from_saved_grid(self):
        rng = np.random.default_rng(29)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source, output = base / "source", base / "review"
            self._write_json(source / "metrics/paper_geometry_summary.json", {"selected_layer": 0, "selected_rank": 2})
            contexts = [
                {"context_id": f"p{i}", "problem_id": f"problem-{i}", "problem_group": "auxiliary" if i < 2 else "analysis_test"}
                for i in range(5)
            ]
            self._write_json(output / "hidden_states/contexts.json", contexts)
            states_path = output / "hidden_states/layer_0.npy"
            states_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(states_path, rng.normal(size=(5, 16, 8)).astype(np.float32))
            self._write_json(output / "manifests/candidate_transfer_states.json", {
                "contexts": str(output / "hidden_states/contexts.json"), "candidate_token_ids": list(range(16)),
                "layers": [{"layer": 0, "path": str(states_path)}],
            })
            groups = {
                "high_probability": list(range(8)), "low_probability": list(range(8, 16)),
                "independent_A": list(range(8)), "independent_B": list(range(8, 16)),
                "independent_A_exclusive": list(range(8)), "independent_B_exclusive": list(range(8, 16)),
            }
            self._write_json(output / "candidate_tokens/review_candidate_sets.json", {"groups": groups, "diagnostics": {"overlap": 0}})
            config = {
                "seed": 0, "source_results_root": str(source), "results_root": str(output),
                "statistics": {"bootstrap_replicates": 20, "ci": 0.95},
                "review_experiments": {"candidate_transfer_rank": 2, "candidate_transfer_minimum_rank": 2, "two_way_bootstrap_replicates": 20},
            }
            config_path = base / "config.json"
            self._write_json(config_path, config)
            with patch.object(sys, "argv", ["analyze_candidate_distribution_transfer", "--config", str(config_path)]):
                candidate_transfer_main()
            summary = json.loads((output / "metrics/candidate_distribution_transfer_summary.json").read_text(encoding="utf-8"))
            self.assertIn("high_to_low", summary["pairs"])
            self.assertEqual(summary["pairs"]["high_to_low"]["rank"], 2)
            self.assertEqual(
                summary["pairs"]["high_to_low"]["source_fit_token_count"],
                summary["pairs"]["high_to_low"]["target_fit_token_count"],
            )

    def test_context_control_stage_runs_from_saved_grid(self):
        rng = np.random.default_rng(31)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source, output = base / "source", base / "review"
            self._write_json(source / "metrics/paper_geometry_summary.json", {"selected_layer": 0, "selected_rank": 2})
            candidate_rows = {
                "candidate_token_ids": list(range(12)),
                "folds": [
                    {"fold_id": 0, "train_indices": list(range(6)), "evaluation_indices": list(range(6, 12))},
                    {"fold_id": 1, "train_indices": list(range(6, 12)), "evaluation_indices": list(range(6))},
                ],
            }
            self._write_json(source / "candidate_tokens/candidate_tokens.json", candidate_rows)
            contexts = [
                {"context_id": "aux0", "role": "auxiliary", "control_type": "none", "target_prefix_id": None, "target_problem_id": "a0", "source_prefix_id": "aux0", "exact_target_length": True, "same_last_token": True},
                {"context_id": "aux1", "role": "auxiliary", "control_type": "none", "target_prefix_id": None, "target_problem_id": "a1", "source_prefix_id": "aux1", "exact_target_length": True, "same_last_token": True},
                {"context_id": "target", "role": "target", "control_type": "self", "target_prefix_id": "target", "target_problem_id": "problem", "source_prefix_id": "target", "exact_target_length": True, "same_last_token": True},
                {"context_id": "control", "role": "control", "control_type": "exact_length_random", "target_prefix_id": "target", "target_problem_id": "problem", "source_prefix_id": "donor", "exact_target_length": True, "same_last_token": False},
            ]
            controls_path = output / "controls/review_context_controls.jsonl"
            controls_path.parent.mkdir(parents=True, exist_ok=True)
            controls_path.write_text("".join(json.dumps(row) + "\n" for row in contexts), encoding="utf-8")
            states_path = output / "hidden_states/context.npy"
            states_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(states_path, rng.normal(size=(4, 12, 8)).astype(np.float32))
            self._write_json(output / "manifests/context_control_states.json", {
                "contexts": str(controls_path), "layers": [{"layer": 0, "path": str(states_path)}],
            })
            config = {
                "seed": 0, "source_results_root": str(source), "results_root": str(output),
                "statistics": {"bootstrap_replicates": 20, "ci": 0.95},
                "review_experiments": {"context_control_rank": 2, "two_way_bootstrap_replicates": 20},
            }
            config_path = base / "config.json"
            self._write_json(config_path, config)
            with patch.object(sys, "argv", ["analyze_context_controls", "--config", str(config_path)]):
                context_controls_main()
            summary = json.loads((output / "metrics/context_control_summary.json").read_text(encoding="utf-8"))
            self.assertIn("exact_length_random", summary["controls"])
            self.assertEqual(summary["controls"]["exact_length_random"]["unique_target_problems"], 1)

    def test_jacobian_alignment_stage_runs_from_saved_responses(self):
        rng = np.random.default_rng(37)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source, output = base / "source", base / "review"
            prefixes = [
                {"prefix_id": "aux0", "problem_id": "a0", "problem_group": "auxiliary"},
                {"prefix_id": "aux1", "problem_id": "a1", "problem_group": "auxiliary"},
                {"prefix_id": "t0", "problem_id": "p0", "problem_group": "analysis_test"},
                {"prefix_id": "t1", "problem_id": "p1", "problem_group": "analysis_test"},
            ]
            prefix_path = source / "hidden_states/prefixes.jsonl"
            prefix_path.parent.mkdir(parents=True, exist_ok=True)
            prefix_path.write_text("".join(json.dumps(row) + "\n" for row in prefixes), encoding="utf-8")
            states_path = source / "hidden_states/layer_0.npy"
            np.save(states_path, rng.normal(size=(4, 8, 6)).astype(np.float32))
            self._write_json(source / "manifests/hidden_states.json", {
                "prefix_snapshot": str(prefix_path), "layers": [{"layer": 0, "successor_path": str(states_path)}],
                "model": {"hidden_size": 6},
            })
            self._write_json(source / "candidate_tokens/candidate_tokens.json", {"analysis_indices": list(range(8)), "candidate_token_ids": list(range(8))})
            self._write_json(source / "metrics/paper_geometry_summary.json", {"selected_layer": 0, "selected_rank": 2})
            jacobian_root = output / "hidden_states/jacobian_responses"
            jacobian_root.mkdir(parents=True, exist_ok=True)
            left = np.linalg.qr(rng.normal(size=(8, 3)))[0]
            embedding_path = jacobian_root / "embedding.npz"
            np.savez(embedding_path, left_vectors=left.astype(np.float32))
            entries = []
            for index, row in enumerate(prefixes):
                path = jacobian_root / f"row_{index}.npz"
                np.savez(path, weighted_responses=rng.normal(size=(3, 6)).astype(np.float32))
                entries.append({"prefix_id": row["prefix_id"], "problem_id": row["problem_id"], "role": "auxiliary" if index < 2 else "target", "path": str(path)})
            self._write_json(output / "manifests/jacobian_responses.json", {
                "layer": 0, "candidate_indices": list(range(8)), "candidate_embedding_svd": str(embedding_path),
                "embedding_components": 3, "embedding_energy_retained_fraction": 0.8, "contexts": entries,
            })
            config = {
                "seed": 0, "source_results_root": str(source), "results_root": str(output),
                "statistics": {"bootstrap_replicates": 20, "ci": 0.95},
                "review_experiments": {"jacobian_alignment_rank": 2},
            }
            config_path = base / "config.json"
            self._write_json(config_path, config)
            with patch.object(sys, "argv", ["analyze_jacobian_alignment", "--config", str(config_path)]):
                jacobian_alignment_main()
            summary = json.loads((output / "metrics/jacobian_alignment_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["target_contexts"], 2)
            self.assertEqual(summary["requested_rank"], 2)

    def test_stability_stage_runs_from_saved_residuals(self):
        rng = np.random.default_rng(41)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source, output = base / "source", base / "review"
            prefixes = [
                {"prefix_id": "target", "problem_id": "p0", "problem_group": "analysis_test"},
                {"prefix_id": "wrong", "problem_id": "p1", "problem_group": "matching_pool"},
            ]
            prefix_path = source / "hidden_states/prefixes.jsonl"
            prefix_path.parent.mkdir(parents=True, exist_ok=True)
            prefix_path.write_text("".join(json.dumps(row) + "\n" for row in prefixes), encoding="utf-8")
            self._write_json(source / "manifests/hidden_states.json", {"prefix_snapshot": str(prefix_path), "model": {"hidden_size": 6}})
            residual_path = source / "residuals/layer_0_fold_0.npz"
            residual_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                residual_path, train_residuals=rng.normal(size=(2, 10, 6)).astype(np.float32),
                evaluation_residuals=rng.normal(size=(2, 4, 6)).astype(np.float32),
                nonauxiliary_prefix_indices=np.asarray([0, 1]), train_candidate_indices=np.arange(10),
                evaluation_candidate_indices=np.arange(10, 14),
            )
            self._write_json(source / "manifests/residuals.json", {"entries": [{"layer": 0, "fold": 0, "path": str(residual_path)}]})
            wrong_path = source / "controls/wrong_prefixes.jsonl"
            wrong_path.parent.mkdir(parents=True, exist_ok=True)
            wrong_path.write_text(json.dumps({"prefix_id": "target", "wrong_prefix_ids": ["wrong"]}) + "\n", encoding="utf-8")
            self._write_json(source / "metrics/paper_geometry_summary.json", {"selected_layer": 0, "selected_rank": 2})
            config = {
                "seed": 0, "source_results_root": str(source), "results_root": str(output),
                "statistics": {"bootstrap_replicates": 20, "ci": 0.95},
                "review_experiments": {"stability_rank": 2, "subspace_candidate_bootstrap_replicates": 3},
            }
            config_path = base / "config.json"
            self._write_json(config_path, config)
            with patch.object(sys, "argv", ["analyze_subspace_stability", "--config", str(config_path)]):
                stability_main()
            summary = json.loads((output / "metrics/subspace_stability_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["rows"], 1)
            self.assertAlmostEqual(summary["random_subspace_baseline"]["expected_mean_cosine_squared"], 2 / 6)


if __name__ == "__main__":
    unittest.main()
