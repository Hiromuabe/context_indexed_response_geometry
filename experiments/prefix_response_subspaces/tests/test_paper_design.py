import unittest
import tempfile
from pathlib import Path

import numpy as np

from experiments.prefix_response_subspaces.analyze_paper_geometry import (
    _basis_label_permutation_analysis,
    _add_pooled_top_k_summary,
    _equal_energy_global_basis,
    _fit_controls,
    _interaction_energy,
    _projection_rotation_distance,
    _pooled_top_k_rows,
    _resolve_conditional_basis,
    _rotation_rows,
    _select_primary,
)
from experiments.prefix_response_subspaces.src.utils import load_config
from experiments.prefix_response_subspaces.run_paper_pipeline import _require_recomputable_analysis
from experiments.prefix_response_subspaces.run_paper_replication import _replication_config, _specs
from experiments.prefix_response_subspaces.src.subspaces import top_svd


class PaperDesignTest(unittest.TestCase):
    def test_wide_gram_solver_matches_direct_svd_subspace(self):
        rng=np.random.default_rng(51); matrix=rng.normal(size=(20,60)); direct=np.linalg.svd(matrix,full_matrices=False)[2][:7].T; optimized=top_svd(matrix,7)
        self.assertAlmostEqual(_projection_rotation_distance(direct,optimized),0.0,places=10)

    def test_fast_basis_label_permutation_returns_requested_null_replicates(self):
        rng=np.random.default_rng(81); prefixes=[
            {"prefix_id":"train","problem_id":"q0","problem_group":"analysis_train","prefix_length_bin":0,"reasoning_progress_bin":0},
            {"prefix_id":"wrong","problem_id":"q1","problem_group":"analysis_train","prefix_length_bin":0,"reasoning_progress_bin":0},
            {"prefix_id":"test-a","problem_id":"q2","problem_group":"analysis_test","prefix_length_bin":0,"reasoning_progress_bin":0},
            {"prefix_id":"test-b","problem_id":"q3","problem_group":"analysis_test","prefix_length_bin":0,"reasoning_progress_bin":0},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path=f"{directory}/residuals.npz"; np.savez(path,train_residuals=rng.normal(size=(4,8,5)),evaluation_residuals=rng.normal(size=(4,4,5)),nonauxiliary_prefix_indices=np.arange(4))
            manifest={"entries":[{"layer":2,"fold":0,"path":path}]}; config={"seed":0,"permutation":{"replicates":7,"minimum_stratum_size":2,"minimum_exchangeable_prefix_fraction":1.0}}
            null,diagnostics=_basis_label_permutation_analysis(manifest,2,1,prefixes,{"test-a":["wrong"],"test-b":["wrong"]},config,{"test-a","test-b"})
            self.assertEqual(null["delta_wrong"].shape,(7,)); self.assertTrue(diagnostics["permutation_inference_valid"]); self.assertIn("fitted-local-basis",diagnostics["method"])

    def test_recompute_analysis_fails_early_when_branch_artifacts_are_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            config={"results_root":directory,"seed":0}
            with self.assertRaisesRegex(RuntimeError,"remove --recompute-analysis"):
                _require_recomputable_analysis(config)

    def test_full_config_has_final_claim_aligned_counts_and_ranks(self):
        config=load_config("experiments/prefix_response_subspaces/configs/paper_full.yaml")
        self.assertEqual(config["data"]["evaluation_prefixes"],256)
        self.assertEqual(config["candidates"]["calibration"],64)
        self.assertEqual(config["candidates"]["analysis"],256)
        self.assertEqual(config["analysis"]["ranks"],[1,2,4,8,16,32,64])
        self.assertEqual(config["controls"]["wrong_prefixes_per_target"],5)

    def test_fast_full_preserves_primary_design_with_smaller_donor_pool(self):
        regular=load_config("experiments/prefix_response_subspaces/configs/paper_full.yaml"); fast=load_config("experiments/prefix_response_subspaces/configs/paper_full_fast.yaml")
        for path in (("data","evaluation_prefixes"),("data","analysis_dev_prefixes"),("data","analysis_train_prefixes"),("candidates","total"),("candidates","calibration"),("candidates","analysis"),("permutation","replicates")):
            self.assertEqual(regular[path[0]][path[1]],fast[path[0]][path[1]])
        self.assertEqual(fast["data"]["prefix_pool_size"],1536)

    def test_qwen3_replication_is_base_and_keeps_confirmatory_layer_rank_fixed(self):
        main=load_config("experiments/prefix_response_subspaces/configs/paper_full_fast.yaml")
        specs=_specs(main,["qwen3_17b_base","qwen3_4b_base","qwen3_8b_base"])
        expected={"qwen3_17b_base":"Qwen/Qwen3-1.7B-Base","qwen3_4b_base":"Qwen/Qwen3-4B-Base","qwen3_8b_base":"Qwen/Qwen3-8B-Base"}
        for name,checkpoint in expected.items():
            spec=next(item for item in specs if item["name"]==name)
            self.assertEqual(spec["model"]["checkpoint"],checkpoint)
            self.assertEqual(spec["model"]["model_type"],"base")
            rep=_replication_config(main,Path(main["results_root"]),spec)
            self.assertEqual(rep["model"]["target_layers"],[0])
            self.assertEqual(rep["analysis"]["ranks"],[64])
            self.assertTrue(rep["replication_independent_tokenizer"])
            if name=="qwen3_8b_base":
                self.assertEqual(rep["extraction"]["per_device_batch_size"],8)

    def test_conditional_global_uses_only_training_prefixes_in_same_stratum(self):
        prefixes=[
            {"prefix_id":"train-a","problem_group":"analysis_train","prefix_length_bin":0,"reasoning_progress_bin":0},
            {"prefix_id":"train-b","problem_group":"analysis_train","prefix_length_bin":1,"reasoning_progress_bin":0},
            {"prefix_id":"test","problem_group":"analysis_test","prefix_length_bin":0,"reasoning_progress_bin":0},
        ]
        rng=np.random.default_rng(4); train_r=rng.normal(size=(3,8,6)); local,conditional=_fit_controls(train_r,prefixes,np.arange(3),2)
        self.assertEqual(set(conditional),{(0,0),(1,0)})
        self.assertEqual(set(local),{"train-a","train-b","test"})

    def test_conditional_global_fallback_keeps_progress_and_uses_nearest_length(self):
        bases={(0,1):np.eye(3)[:,:1],(4,1):np.eye(3)[:,1:2],(2,0):np.eye(3)[:,2:3]}
        prefix={"prefix_length_bin":3,"reasoning_progress_bin":1}
        basis,exact,distance,resolved=_resolve_conditional_basis(bases,prefix)
        self.assertFalse(exact); self.assertEqual(distance,1); self.assertEqual(resolved,(4,1)); np.testing.assert_array_equal(basis,bases[(4,1)])

    def test_conditional_global_fallback_prefers_exact_bin(self):
        bases={(3,1):np.eye(3)[:,:1],(4,1):np.eye(3)[:,1:2]}
        basis,exact,distance,resolved=_resolve_conditional_basis(bases,{"prefix_length_bin":3,"reasoning_progress_bin":1})
        self.assertTrue(exact); self.assertEqual(distance,0); self.assertEqual(resolved,(3,1)); np.testing.assert_array_equal(basis,bases[(3,1)])

    def test_top_k_projection_energy_is_pooled_across_heldout_folds(self):
        prefixes=[
            {"prefix_id":"global","problem_id":"q0","problem_group":"analysis_train","prefix_length_bin":0,"reasoning_progress_bin":0},
            {"prefix_id":"wrong","problem_id":"q1","problem_group":"matching_pool","prefix_length_bin":0,"reasoning_progress_bin":0},
            {"prefix_id":"target","problem_id":"q2","problem_group":"analysis_test","prefix_length_bin":0,"reasoning_progress_bin":0},
        ]
        train=np.asarray([[[0.,1.],[0.,2.]],[[0.,1.],[0.,2.]],[[1.,0.],[2.,0.]]])
        evaluation=np.asarray([[[0.,0.]],[[0.,0.]],[[1.,0.]]])
        with tempfile.TemporaryDirectory() as directory:
            entries=[]
            for fold,candidate_index in enumerate((0,1)):
                path=f"{directory}/fold_{fold}.npz"; np.savez(path,train_residuals=train,evaluation_residuals=evaluation,nonauxiliary_prefix_indices=np.arange(3),evaluation_candidate_indices=np.asarray([candidate_index])); entries.append({"layer":4,"fold":fold,"path":path})
            rows=_pooled_top_k_rows({"entries":entries},4,1,prefixes,{"target":["wrong"]},set(),{"target":np.asarray([1,1])},[256])
        self.assertEqual(len(rows),1); self.assertEqual(rows[0]["top256_token_count"],2)
        self.assertAlmostEqual(rows[0]["delta_conditional_global_top256"],1.0,places=12)
        self.assertAlmostEqual(rows[0]["delta_wrong_top256"],1.0,places=12)

    def test_top_k_summary_keeps_primary_threshold_and_reports_sensitivity(self):
        rows=[
            {"problem_id":"q0","top256_token_count":16,"conditional_global_exact_length_bin":True,"wrong_control_exact_length_bin":True,"delta_conditional_global_top256":.4,"delta_wrong_top256":.5},
            {"problem_id":"q1","top256_token_count":8,"conditional_global_exact_length_bin":True,"wrong_control_exact_length_bin":True,"delta_conditional_global_top256":.3,"delta_wrong_top256":.4},
        ]
        summary={}; config={"seed":0,"data":{"evaluation_prefixes":2},"statistics":{"bootstrap_replicates":20,"ci":.95}}
        _add_pooled_top_k_summary(summary,rows,[256],16,config)
        coverage=summary["top_k_coverage"]["256"]
        self.assertEqual(coverage["eligible_exact_both_prefixes"],1)
        self.assertEqual(coverage["minimum_token_sensitivity"]["8"]["eligible_exact_prefixes"],2)
        self.assertEqual(summary["delta_conditional_global_top256"]["n_problems"],1)

    def test_equal_energy_global_is_invariant_to_per_prefix_rescaling(self):
        rng=np.random.default_rng(9); samples=rng.normal(size=(3,12,6)); scaled=samples.copy(); scaled[0]*=1000; scaled[1]*=.001
        left=_equal_energy_global_basis(samples,3); right=_equal_energy_global_basis(scaled,3)
        self.assertAlmostEqual(_projection_rotation_distance(left,right),0.0,places=10)

    def test_projection_rotation_distance_has_expected_endpoints(self):
        identity=np.eye(4)
        self.assertAlmostEqual(_projection_rotation_distance(identity[:,:2],identity[:,:2]),0.0)
        self.assertAlmostEqual(_projection_rotation_distance(identity[:,:2],identity[:,2:]),1.0)

    def test_split_half_between_rotation_exceeds_within_noise(self):
        rng=np.random.default_rng(12); coefficients=rng.normal(size=(3,40)); train=np.zeros((3,40,4)); train[0,:,0]=coefficients[0]; train[1,:,1]=coefficients[1]; train[2,:,2]=coefficients[2]
        prefixes=[
            {"prefix_id":"target","problem_id":"q0","problem_group":"analysis_test","prefix_length_bin":0,"reasoning_progress_bin":0},
            {"prefix_id":"wrong","problem_id":"q1","problem_group":"matching_pool","prefix_length_bin":0,"reasoning_progress_bin":0},
            {"prefix_id":"global","problem_id":"q2","problem_group":"analysis_train","prefix_length_bin":0,"reasoning_progress_bin":0},
        ]
        row=_rotation_rows(train,prefixes,np.arange(3),{"target":["wrong"]},1,4,0,7)[0]
        self.assertAlmostEqual(row["R_within"],0.0,places=12)
        self.assertAlmostEqual(row["R_between"],1.0,places=12)
        self.assertGreater(row["R_between_minus_within"],0)

    def test_interaction_energy_is_zero_for_additive_prefix_and_token_effects(self):
        prefix=np.asarray([[1.,0.],[2.,0.],[3.,0.]])
        token=np.asarray([[0.,1.],[0.,2.],[0.,3.],[0.,4.]])
        z=prefix[:,None,:]+token[None,:,:]
        result=_interaction_energy(z,np.asarray([0,1]),np.asarray([2]),np.arange(4))
        self.assertAlmostEqual(result["interaction_energy"],0.0,places=12)

    def test_primary_selection_uses_dev_only_and_smallest_rank_at_90_percent(self):
        rows=[]
        for layer,delta in ((1,.1),(2,.3)):
            for rank,ev in ((1,.5),(2,.91),(4,1.0),(16,1.0),(64,1.0)):
                rows.append({"layer":layer,"rank":rank,"delta_conditional_global":delta,"delta_wrong":delta-.01,"ev_local":ev})
        layer,rank,_diagnostics=_select_primary(rows,[1,2],[1,2,4,16,64],16,.9)
        self.assertEqual(layer,2); self.assertEqual(rank,2)


if __name__=="__main__": unittest.main()
