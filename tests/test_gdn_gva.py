"""Device checks for grouped Q/K versus materialized Q/K in FLA GDN."""
import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec('torch') is not None


@unittest.skipUnless(HAS_TORCH, 'PyTorch and a compatible accelerator runtime are required')
class GDNGroupedValueTests(unittest.TestCase):
    def test_grouped_heads_match_expanded_heads(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('A compatible CUDA-interface accelerator is required')
        from fla.ops.gated_delta_rule import (
            chunk_gated_delta_rule,
            fused_recurrent_gated_delta_rule,
        )

        torch.manual_seed(42)
        for batch, tokens, heads, value_heads, key_dim, value_dim in [
            (1, 1, 2, 4, 128, 128),
            (2, 1, 2, 8, 128, 128),
            (1, 65, 2, 4, 128, 128),
            (1, 65, 4, 4, 128, 128),
        ]:
            for has_initial_state in (False, True):
                with self.subTest(shape=(batch, tokens, heads, value_heads), initial=has_initial_state):
                    dtype = torch.bfloat16
                    q = torch.randn(batch, tokens, heads, key_dim, device='cuda', dtype=dtype)
                    k = torch.randn_like(q)
                    v = torch.randn(batch, tokens, value_heads, value_dim, device='cuda', dtype=dtype)
                    g = torch.randn(batch, tokens, value_heads, device='cuda', dtype=dtype)
                    beta = torch.randn_like(g)
                    initial = (
                        torch.randn(batch, value_heads, key_dim, value_dim, device='cuda', dtype=torch.float32)
                        if has_initial_state else None
                    )
                    initial_copy = initial.clone() if initial is not None else None
                    common = dict(
                        v=v, g=g, beta=beta, output_final_state=True,
                        use_qk_l2norm_in_kernel=True,
                        use_gate_in_kernel=True,
                        A_log=torch.zeros(value_heads, device='cuda', dtype=torch.float32),
                        dt_bias=torch.zeros(value_heads, device='cuda', dtype=dtype),
                        use_beta_sigmoid_in_kernel=True,
                    )
                    operation = fused_recurrent_gated_delta_rule if tokens == 1 and has_initial_state else chunk_gated_delta_rule
                    with torch.inference_mode():
                        expected, expected_state = operation(
                            q=q.repeat_interleave(value_heads // heads, dim=2),
                            k=k.repeat_interleave(value_heads // heads, dim=2),
                            initial_state=initial.clone() if initial is not None else None,
                            **common,
                        )
                        actual, actual_state = operation(q=q, k=k, initial_state=initial, **common)
                    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
                    torch.testing.assert_close(actual_state, expected_state, rtol=1e-3, atol=1e-3)
                    if initial is not None:
                        torch.testing.assert_close(initial, initial_copy, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
