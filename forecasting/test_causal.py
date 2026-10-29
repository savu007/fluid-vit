"""Run: python -m unittest forecasting.test_causal -v (no DNS files/GPU needed)."""
import copy
import unittest
import torch
from torch import nn
from .causal import install_block_test, temporal_mask
from ornl_vit.matey.models.attention_modules import AttentionBlock_all2all
from ornl_vit.matey.compat import attention_backend
from ornl_vit.matey.data_utils.shared_utils import normalize_spatiotemporal_persample


class CausalityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        # Unit residual scales make dependencies observable, unlike tiny initial scales.
        self.block = AttentionBlock_all2all(24, 3, layer_scale_init_value=1.).eval()
        self.x = torch.randn(2, 24, 32)

    def test_mask_allows_whole_current_frame(self):
        mask = temporal_mask(32, 4)
        self.assertTrue(mask[:8, :8].all())
        self.assertFalse(mask[:8, 8:].any())
        self.assertTrue(mask[24:, :].all())
        with self.assertRaises(ValueError): temporal_mask(31, 4)

    def test_unmasked_adapter_matches_upstream_outputs_and_gradients(self):
        adapter = install_block_test(copy.deepcopy(self.block), causal=False)
        a = self.x.clone().requires_grad_(); b = a.detach().clone().requires_grad_()
        with attention_backend('sdpa'):
            expected = self.block(a)
        actual = adapter(b)
        torch.testing.assert_close(actual, expected)
        expected.square().mean().backward(); actual.square().mean().backward()
        torch.testing.assert_close(a.grad, b.grad)
        for (_, p), (_, q) in zip(self.block.named_parameters(), adapter.named_parameters()):
            if p.grad is not None: torch.testing.assert_close(p.grad, q.grad)

    def test_multiple_blocks_preserve_prefix_only_with_local_normalization(self):
        for causal, local in ((False, False), (True, False), (True, True)):
            model = nn.Sequential(*[install_block_test(copy.deepcopy(self.block),
                causal=causal, frame_local_norm=local) for _ in range(3)])
            before = model(self.x)
            for prefix in (8, 16, 24):
                changed = self.x.clone()
                changed[:, :, prefix:] = torch.randn_like(changed[:, :, prefix:]) * 3 + 5
                error = (model(changed)[:, :, :prefix] - before[:, :, :prefix]).abs().max().item()
                print(f'causal={causal}, frame_local_norm={local}, prefix={prefix}: max change={error:.8g}')
                if causal and local: self.assertEqual(error, 0.)
                else: self.assertGreater(error, 1e-5)
            x = self.x.clone().requires_grad_()
            model(x)[:, :, :16].square().mean().backward()
            future_grad = x.grad[:, :, 16:].abs().max().item()
            if causal and local: self.assertEqual(future_grad, 0.)
            else: self.assertGreater(future_grad, 0.)

    def test_input_normalization_also_uses_later_frames(self):
        x = torch.randn(4, 1, 3, 4, 4, 4)
        changed = x.clone(); changed[-1] = changed[-1] * 3 + 5
        a = normalize_spatiotemporal_persample(x)[0]
        b = normalize_spatiotemporal_persample(changed)[0]
        self.assertGreater((a[0] - b[0]).abs().max().item(), .01)
        # no_grad statistics make a gradient-only test insufficient for this path.

if __name__ == '__main__':
    unittest.main()
