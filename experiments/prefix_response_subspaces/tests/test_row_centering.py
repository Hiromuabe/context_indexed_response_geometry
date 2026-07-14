import unittest
import numpy as np

from experiments.prefix_response_subspaces.src.residualization import center_train_and_evaluation


class RowCenteringTest(unittest.TestCase):
    def test_train_and_evaluation_are_centered_independently(self):
        rng = np.random.default_rng(3); evaluation = rng.normal(size=(4, 12, 7)); auxiliary = rng.normal(size=(5, 12, 7))
        train, heldout = center_train_and_evaluation(evaluation, auxiliary, range(8), range(8, 12))
        np.testing.assert_allclose(train.residuals.mean(axis=1), 0, atol=2e-7)
        np.testing.assert_allclose(heldout.residuals.mean(axis=1), 0, atol=2e-7)
        np.testing.assert_allclose(train.prefix_mean, evaluation[:, :8].mean(axis=1), atol=1e-7)
        np.testing.assert_allclose(heldout.prefix_mean, evaluation[:, 8:].mean(axis=1), atol=1e-7)
        self.assertFalse(np.allclose(train.prefix_mean, heldout.prefix_mean))

    def test_float32_large_effects_are_absolutely_recentered(self):
        rng=np.random.default_rng(31); prefix=rng.normal(size=(6,1,9)).astype(np.float32)*1000; token=rng.normal(size=(1,16,9)).astype(np.float32)*1000; interaction=rng.normal(size=(6,16,9)).astype(np.float32)
        evaluation=prefix+token+interaction; auxiliary=(rng.normal(size=(5,1,9)).astype(np.float32)*1000)+token+rng.normal(size=(5,16,9)).astype(np.float32)
        train,heldout=center_train_and_evaluation(evaluation,auxiliary,range(12),range(12,16))
        self.assertLess(float(np.abs(np.asarray(train.residuals,dtype=np.float64).mean(axis=1)).max()),2e-5)
        self.assertLess(float(np.abs(np.asarray(heldout.residuals,dtype=np.float64).mean(axis=1)).max()),2e-5)


if __name__ == "__main__": unittest.main()
