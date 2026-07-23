import types
import unittest
from unittest import mock

import torch

from vllm_qdq_plugin.svg2_attn import impl as svg2_impl_mod
from vllm_qdq_plugin.svg2_attn.impl import SVG2Impl


class SVG2ImplTests(unittest.TestCase):
    def test_unknown_backend_kwarg_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown backend kwargs"):
            SVG2Impl(
                num_heads=2,
                num_kv_heads=2,
                head_size=64,
                softmax_scale=0.125,
                backend_kwargs={"bad_key": 1},
            )

    def test_forward_cuda_uses_sdpa_for_dense_warmup(self) -> None:
        impl = SVG2Impl(
            num_heads=2,
            num_kv_heads=2,
            head_size=64,
            softmax_scale=0.125,
            backend_kwargs={"num_q_centroids": 2, "num_k_centroids": 2},
            attention_role="self",
        )
        q = torch.randn(1, 8, 2, 64)
        sentinel = torch.randn_like(q)

        with (
            mock.patch.object(impl, "_fallback_reason", return_value=None),
            mock.patch.object(impl, "_should_use_dense_warmup", return_value=True),
            mock.patch.object(impl, "_forward_sdpa", return_value=sentinel) as sdpa_mock,
        ):
            out = impl.forward_cuda(q, q, q)

        self.assertIs(out, sentinel)
        sdpa_mock.assert_called_once()

    def test_forward_cuda_uses_svg2_path_when_eligible(self) -> None:
        impl = SVG2Impl(
            num_heads=2,
            num_kv_heads=2,
            head_size=64,
            softmax_scale=0.125,
            backend_kwargs={"num_q_centroids": 2, "num_k_centroids": 2},
            attention_role="self",
        )
        q = torch.randn(1, 8, 2, 64)
        sentinel = torch.randn_like(q)

        with (
            mock.patch.object(impl, "_fallback_reason", return_value=None),
            mock.patch.object(impl, "_should_use_dense_warmup", return_value=False),
            mock.patch.object(impl, "_forward_svg2", return_value=sentinel) as svg2_mock,
        ):
            out = impl.forward_cuda(q, q, q)

        self.assertIs(out, sentinel)
        svg2_mock.assert_called_once()

    def test_forward_svg2_preserves_nhd_shape(self) -> None:
        impl = SVG2Impl(
            num_heads=2,
            num_kv_heads=2,
            head_size=64,
            softmax_scale=0.125,
            backend_kwargs={
                "num_q_centroids": 2,
                "num_k_centroids": 4,
                "kmeans_iter_init": 1,
                "kmeans_iter_step": 1,
            },
            attention_role="self",
            attention_layer_idx=3,
            attention_num_layers=40,
        )
        query = torch.randn(1, 8, 2, 64)

        def fake_kmeans(x, n_clusters, max_iters=0, init_centroids=None):
            batch_heads, seq_len, dim = x.shape
            labels = torch.arange(seq_len, device=x.device).repeat(batch_heads, 1) % n_clusters
            centroids = x[:, :n_clusters, :].contiguous()
            cluster_sizes = torch.ones(batch_heads, n_clusters, device=x.device, dtype=torch.int32)
            return labels, centroids, cluster_sizes, max_iters

        fake_ops = {
            "apply_inverse_permutation_triton": lambda x, sorted_indices, dim=2: x,
            "batch_kmeans_Euclid": fake_kmeans,
            "dynamic_block_sparse_fwd_flashinfer": (
                lambda q, k, v, dynamic_map, qc_sz, kc_sz, is_cpu=False: q
            ),
            "identify_dynamic_map": (
                lambda qcentroids, kcentroids, q_cluster_sizes, k_cluster_sizes, top_p, min_ratio: torch.ones(
                    qcentroids.shape[0],
                    qcentroids.shape[1],
                    qcentroids.shape[2],
                    kcentroids.shape[2],
                    device=qcentroids.device,
                )
            ),
            "permute_tensor_by_labels_triton": (
                lambda x, labels, dim=2, sorted_indices=None: (
                    x,
                    sorted_indices if sorted_indices is not None else torch.arange(x.shape[dim], device=x.device),
                )
            ),
        }

        with mock.patch.object(svg2_impl_mod, "_resolve_svg2_ops", return_value=fake_ops):
            out = impl._forward_svg2(query, query, query)

        self.assertEqual(tuple(out.shape), tuple(query.shape))
        self.assertTrue(impl._centroids_initialized)

    def test_warmup_ratio_uses_forward_context_counts(self) -> None:
        impl = SVG2Impl(
            num_heads=2,
            num_kv_heads=2,
            head_size=64,
            softmax_scale=0.125,
            backend_kwargs={"first_times_fp": 0.5, "num_q_centroids": 2, "num_k_centroids": 2},
            attention_role="self",
        )

        with (
            mock.patch.object(svg2_impl_mod, "is_forward_context_available", return_value=True),
            mock.patch.object(
                svg2_impl_mod,
                "get_forward_context",
                return_value=types.SimpleNamespace(denoise_step_idx=1, num_denoise_steps=4),
            ),
        ):
            self.assertTrue(impl._should_use_dense_warmup())

        with (
            mock.patch.object(svg2_impl_mod, "is_forward_context_available", return_value=True),
            mock.patch.object(
                svg2_impl_mod,
                "get_forward_context",
                return_value=types.SimpleNamespace(denoise_step_idx=2, num_denoise_steps=4),
            ),
        ):
            self.assertFalse(impl._should_use_dense_warmup())


if __name__ == "__main__":
    unittest.main()
