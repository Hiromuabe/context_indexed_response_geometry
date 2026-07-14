import unittest

try:
    import torch
except ImportError:
    torch = None

from experiments.prefix_response_subspaces.src.mechanism import _FirstLayerEarlyExit, make_first_layer_site_controller


@unittest.skipIf(torch is None, "PyTorch unavailable")
class MechanismHookTest(unittest.TestCase):
    def test_three_first_layer_sites_are_captured_at_requested_positions(self):
        class FakeLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.post_attention_layernorm = torch.nn.Identity()

            def forward(self, hidden_states):
                post_attention_residual = hidden_states + 1.0
                normalized = self.post_attention_layernorm(post_attention_residual)
                return normalized * 2.0

        controller = make_first_layer_site_controller(FakeLayer())
        hidden = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
        positions = torch.tensor([0, 2])
        controller.capture_at(positions)
        with self.assertRaises(_FirstLayerEarlyExit):
            controller(hidden)
        pre, post_attention, post_mlp = controller.take_captured()
        expected_pre = torch.stack((hidden[0, 0], hidden[1, 2]))
        torch.testing.assert_close(pre, expected_pre)
        torch.testing.assert_close(post_attention, expected_pre + 1.0)
        torch.testing.assert_close(post_mlp, (expected_pre + 1.0) * 2.0)

    def test_sequence_input_capture_does_not_configure_endpoint_hooks(self):
        class FakeLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.post_attention_layernorm = torch.nn.Identity()

            def forward(self, hidden_states):
                return self.post_attention_layernorm(hidden_states + 1.0)

        controller = make_first_layer_site_controller(FakeLayer())
        hidden = torch.randn(1, 5, 3)
        controller.capture_sequence_input()
        with self.assertRaises(_FirstLayerEarlyExit):
            controller(hidden)
        torch.testing.assert_close(controller.take_sequence_input(), hidden)


if __name__ == "__main__":
    unittest.main()
