import unittest
import numpy as np

from experiments.prefix_response_subspaces.src.residualization import center_train_and_evaluation
from experiments.prefix_response_subspaces.src.subspaces import explained_variance, top_svd


class SyntheticSmokePipelineTest(unittest.TestCase):
    def test_local_beats_global_and_wrong_on_heldout_tokens(self):
        rng = np.random.default_rng(42); prefixes, tokens, hidden = 6, 20, 18
        auxiliary = rng.normal(scale=.01, size=(4,tokens,hidden)); raw = np.zeros((prefixes,tokens,hidden))
        coefficients = rng.normal(size=(tokens,2)); coefficients[:15] -= coefficients[:15].mean(axis=0); coefficients[15:] -= coefficients[15:].mean(axis=0)
        for i in range(prefixes): raw[i,:,2*i:2*i+2] = coefficients
        train, heldout = center_train_and_evaluation(raw, auxiliary, range(15), range(15,20))
        bases = [top_svd(train.residuals[i],2) for i in range(prefixes)]; global_basis = top_svd(train.residuals.reshape(-1,hidden), 2)
        for i in range(prefixes):
            local = explained_variance(heldout.residuals[i], bases[i]); wrong = explained_variance(heldout.residuals[i], bases[(i+1)%prefixes]); global_ev = explained_variance(heldout.residuals[i], global_basis)
            self.assertGreater(local, .99); self.assertGreater(local, wrong + .9); self.assertGreater(local, global_ev)


if __name__ == "__main__": unittest.main()
