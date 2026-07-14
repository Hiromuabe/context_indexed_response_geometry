import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch unavailable")
class OracleHookTest(unittest.TestCase):
    def test_oracle_replacement_changes_only_selected_batch_positions(self):
        from experiments.prefix_response_subspaces.src.hooks import make_position_replacement_hook
        hidden = torch.randn(3, 5, 7); original = hidden.clone(); positions = torch.tensor([0,2,4]); oracle = hidden[torch.arange(3), positions].clone()
        hook = make_position_replacement_hook(positions, oracle); output = hook(None, (), (hidden, "cache"))
        torch.testing.assert_close(output[0], original, atol=0, rtol=0); self.assertEqual(output[1], "cache"); torch.testing.assert_close(hidden, original)

    def test_batch_specific_replacement_and_tuple_tensor_contract(self):
        from experiments.prefix_response_subspaces.src.hooks import make_position_replacement_hook
        hidden = torch.zeros(2,3,2); positions = torch.tensor([2,0]); replacement = torch.tensor([[1.,2.],[3.,4.]])
        output = make_position_replacement_hook(positions, replacement)(None, (), hidden)
        expected = hidden.clone(); expected[0,2] = replacement[0]; expected[1,0] = replacement[1]; torch.testing.assert_close(output, expected)

    def test_end_to_end_online_oracle_reinjection_preserves_logits(self):
        from types import SimpleNamespace
        from experiments.prefix_response_subspaces.analyze_functional_recovery import FunctionalRecoveryForward
        class Block(torch.nn.Module):
            def __init__(self, width): super().__init__(); self.linear = torch.nn.Linear(width, width, bias=False); torch.nn.init.eye_(self.linear.weight)
            def forward(self, hidden, **_kwargs): return (torch.tanh(self.linear(hidden)),)
        class Decoder(torch.nn.Module):
            def __init__(self): super().__init__(); self.embed_tokens = torch.nn.Embedding(20, 4); self.layers = torch.nn.ModuleList([Block(4), Block(4)])
            def forward(self, input_ids, attention_mask, **_kwargs):
                hidden = self.embed_tokens(input_ids)
                for layer in self.layers: hidden = layer(hidden)[0]
                return SimpleNamespace(last_hidden_state=hidden)
        class LM(torch.nn.Module):
            def __init__(self): super().__init__(); self.model = Decoder(); self.lm_head = torch.nn.Linear(4,20,bias=False)
            def get_decoder(self): return self.model
            def get_output_embeddings(self): return self.lm_head
        model = FunctionalRecoveryForward.build(LM(), 0)
        ids = torch.tensor([[1,2,3],[4,5,6]]); mask = torch.ones_like(ids); positions = torch.tensor([2,1]); replacement = torch.randn(2,4); oracle = torch.ones(2,dtype=torch.bool); sample = torch.arange(2)
        js, kl, top1, overlap, logit_difference, observed = model(ids, mask, positions, replacement, oracle, sample)
        torch.testing.assert_close(js, torch.zeros_like(js), atol=1e-7, rtol=0); torch.testing.assert_close(kl, torch.zeros_like(kl), atol=1e-7, rtol=0); torch.testing.assert_close(logit_difference, torch.zeros_like(logit_difference), atol=0, rtol=0); torch.testing.assert_close(observed, sample)

    def test_clean_forward_is_shared_by_repeated_cell_ids(self):
        from types import SimpleNamespace
        from experiments.prefix_response_subspaces.analyze_functional_recovery import FunctionalRecoveryForward
        class Block(torch.nn.Module):
            def forward(self, hidden, **_kwargs): return (torch.tanh(hidden),)
        class Decoder(torch.nn.Module):
            def __init__(self): super().__init__(); self.embed_tokens=torch.nn.Embedding(20,4); self.layers=torch.nn.ModuleList([Block(),Block()]); self.batch_sizes=[]
            def forward(self,input_ids,attention_mask,**_kwargs):
                self.batch_sizes.append(len(input_ids)); hidden=self.embed_tokens(input_ids)
                for layer in self.layers: hidden=layer(hidden)[0]
                return SimpleNamespace(last_hidden_state=hidden)
        class LM(torch.nn.Module):
            def __init__(self): super().__init__(); self.model=Decoder(); self.lm_head=torch.nn.Linear(4,20,bias=False)
            def get_decoder(self): return self.model
            def get_output_embeddings(self): return self.lm_head
        lm=LM(); model=FunctionalRecoveryForward.build(lm,0); ids=torch.tensor([[1,2,3],[1,2,3],[4,5,6],[4,5,6]]); mask=torch.ones_like(ids); positions=torch.tensor([2,2,2,2]); replacement=torch.randn(4,4); oracle=torch.ones(4,dtype=torch.bool); sample=torch.arange(4); cells=torch.tensor([10,10,11,11])
        js,*_rest=model(ids,mask,positions,replacement,oracle,sample,cells)
        torch.testing.assert_close(js,torch.zeros_like(js),atol=1e-7,rtol=0); self.assertEqual(lm.model.batch_sizes,[2,2,2])

    def test_repeated_cell_float_replacements_cast_to_bfloat_controller_dtype(self):
        from types import SimpleNamespace
        from experiments.prefix_response_subspaces.analyze_functional_recovery import FunctionalRecoveryForward
        class Block(torch.nn.Module):
            def forward(self,hidden,**_kwargs): return (hidden,)
        class Decoder(torch.nn.Module):
            def __init__(self): super().__init__(); self.embed_tokens=torch.nn.Embedding(20,4,dtype=torch.bfloat16); self.layers=torch.nn.ModuleList([Block(),Block()])
            def forward(self,input_ids,attention_mask,**_kwargs):
                hidden=self.embed_tokens(input_ids)
                for layer in self.layers: hidden=layer(hidden)[0]
                return SimpleNamespace(last_hidden_state=hidden)
        class LM(torch.nn.Module):
            def __init__(self): super().__init__(); self.model=Decoder(); self.lm_head=torch.nn.Linear(4,20,bias=False,dtype=torch.bfloat16)
            def get_decoder(self): return self.model
            def get_output_embeddings(self): return self.lm_head
        model=FunctionalRecoveryForward.build(LM(),0,js_only=True); ids=torch.tensor([[1,2],[1,2]]); mask=torch.ones_like(ids); positions=torch.tensor([1,1]); replacement=torch.randn(2,4,dtype=torch.float32); oracle=torch.tensor([True,False]); sample=torch.arange(2); cells=torch.tensor([7,7])
        js,*_rest=model(ids,mask,positions,replacement,oracle,sample,cells)
        self.assertTrue(torch.isfinite(js).all())


if __name__ == "__main__": unittest.main()
