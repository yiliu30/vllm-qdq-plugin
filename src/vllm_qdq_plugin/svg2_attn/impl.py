# SPDX-License-Identifier: Apache-2.0
"""SVG2/SAP attention implementation for vllm-omni diffusion models."""

from __future__ import annotations

import math
import os
import sys
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from functools import lru_cache
from typing import Any

import torch
import torch.nn.functional as F
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionImpl,
    AttentionMetadata,
)
from vllm_omni.diffusion.attention.backends.sdpa import _maybe_reshape_attn_mask
from vllm_omni.diffusion.forward_context import (
    get_forward_context,
    is_forward_context_available,
)

from .. import envs

logger = init_logger(__name__)

_SUPPORTED_HEAD_DIMS = (64, 128)
_ALLOWED_BACKEND_KWARGS = {
    "num_q_centroids",
    "num_k_centroids",
    "top_p_kmeans",
    "min_kc_ratio",
    "kmeans_iter_init",
    "kmeans_iter_step",
    "first_layers_fp",
    "first_times_fp",
    "enabled_roles",
    "fallback_backend",
}


def _debug_enabled() -> bool:
    return bool(envs.VLLM_SVG2_DEBUG)


def _debug_log(msg: str) -> None:
    if _debug_enabled():
        print(f"[VLLM_SVG2_DEBUG] {msg}", file=sys.stderr, flush=True)


def _nvtx_enabled() -> bool:
    return os.getenv("VLLM_SVG2_NVTX", "0").lower() in {"1", "true", "yes"}


def _nvtx_range(name: str):
    if _nvtx_enabled() and torch.cuda.is_available():
        return torch.cuda.nvtx.range(name)
    return nullcontext()


@lru_cache(maxsize=1)
def _resolve_svg2_ops() -> dict[str, Any]:
    try:
        from svg.kernels.triton.permute import (
            apply_inverse_permutation_triton,
            permute_tensor_by_labels_triton,
        )
        from svg.kmeans_utils import (
            batch_kmeans_Euclid,
            dynamic_block_sparse_fwd_flashinfer,
            identify_dynamic_map,
        )
    except ImportError as exc:
        raise RuntimeError(
            "SVG2_ATTN requires the local 'svg' package and its dependencies. "
            "Use the SVG2 node environment or set SVG2_REPO to an importable checkout."
        ) from exc

    return {
        "apply_inverse_permutation_triton": apply_inverse_permutation_triton,
        "batch_kmeans_Euclid": batch_kmeans_Euclid,
        "dynamic_block_sparse_fwd_flashinfer": dynamic_block_sparse_fwd_flashinfer,
        "identify_dynamic_map": identify_dynamic_map,
        "permute_tensor_by_labels_triton": permute_tensor_by_labels_triton,
    }


def _coerce_int(value: Any, key: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"SVG2_ATTN backend kwarg {key!r} must be an integer, got {value!r}.") from exc


def _coerce_float(value: Any, key: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"SVG2_ATTN backend kwarg {key!r} must be a float, got {value!r}.") from exc


def _normalize_enabled_roles(value: Any) -> tuple[str, ...]:
    if value is None:
        return ("self",)
    if isinstance(value, str):
        roles = [value]
    elif isinstance(value, Sequence):
        roles = list(value)
    else:
        raise ValueError("SVG2_ATTN backend kwarg 'enabled_roles' must be a string or sequence of strings.")

    normalized = tuple(str(role).strip().lower() for role in roles if str(role).strip())
    if not normalized:
        raise ValueError("SVG2_ATTN backend kwarg 'enabled_roles' must contain at least one role.")
    return normalized


def _normalize_backend_kwargs(backend_kwargs: Mapping[str, Any] | None) -> dict[str, Any]:
    kwargs = dict(backend_kwargs or {})
    unknown = sorted(set(kwargs) - _ALLOWED_BACKEND_KWARGS)
    if unknown:
        raise ValueError(f"SVG2_ATTN received unknown backend kwargs: {unknown}")

    normalized = {
        "num_q_centroids": _coerce_int(kwargs.pop("num_q_centroids", 300), "num_q_centroids"),
        "num_k_centroids": _coerce_int(kwargs.pop("num_k_centroids", 1000), "num_k_centroids"),
        "top_p_kmeans": _coerce_float(kwargs.pop("top_p_kmeans", 0.9), "top_p_kmeans"),
        "min_kc_ratio": _coerce_float(kwargs.pop("min_kc_ratio", 0.10), "min_kc_ratio"),
        "kmeans_iter_init": _coerce_int(kwargs.pop("kmeans_iter_init", 50), "kmeans_iter_init"),
        "kmeans_iter_step": _coerce_int(kwargs.pop("kmeans_iter_step", 2), "kmeans_iter_step"),
        "first_layers_fp": kwargs.pop("first_layers_fp", 0.03),
        "first_times_fp": kwargs.pop("first_times_fp", 0.2),
        "enabled_roles": _normalize_enabled_roles(kwargs.pop("enabled_roles", None)),
        "fallback_backend": str(kwargs.pop("fallback_backend", "TORCH_SDPA")).upper(),
    }

    if normalized["num_q_centroids"] <= 0 or normalized["num_k_centroids"] <= 0:
        raise ValueError("SVG2_ATTN requires positive num_q_centroids and num_k_centroids.")
    if not 0.0 < normalized["top_p_kmeans"] <= 1.0:
        raise ValueError("SVG2_ATTN requires top_p_kmeans in the interval (0, 1].")
    if normalized["min_kc_ratio"] < 0.0:
        raise ValueError("SVG2_ATTN requires min_kc_ratio >= 0.")
    if normalized["kmeans_iter_init"] < 0 or normalized["kmeans_iter_step"] < 0:
        raise ValueError("SVG2_ATTN requires non-negative kmeans iteration counts.")
    if normalized["fallback_backend"] != "TORCH_SDPA":
        raise ValueError("SVG2_ATTN currently supports only fallback_backend='TORCH_SDPA'.")

    return normalized


class SVG2Impl(AttentionImpl):
    """Attention implementation using SVG2's SAP kernels for Wan self-attention."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        qkv_layout: str | None = None,
        backend_kwargs: dict | None = None,
        **extra_impl_args,
    ) -> None:
        config = _normalize_backend_kwargs(backend_kwargs)

        self.causal = causal
        self.softmax_scale = softmax_scale
        self.prefix = prefix or str(extra_impl_args.get("attention_prefix") or "<unknown>")
        self.role = str(extra_impl_args.get("attention_role") or "self").lower()
        raw_layer_idx = extra_impl_args.get("attention_layer_idx")
        self.layer_idx = None if raw_layer_idx is None else int(raw_layer_idx)
        raw_num_layers = extra_impl_args.get("attention_num_layers")
        self.num_layers = None if raw_num_layers is None else int(raw_num_layers)

        self.num_q_centroids = config["num_q_centroids"]
        self.num_k_centroids = config["num_k_centroids"]
        self.top_p_kmeans = config["top_p_kmeans"]
        self.min_kc_ratio = config["min_kc_ratio"]
        self.kmeans_iter_init = config["kmeans_iter_init"]
        self.kmeans_iter_step = config["kmeans_iter_step"]
        self.first_layers_fp = config["first_layers_fp"]
        self.first_times_fp = config["first_times_fp"]
        self.enabled_roles = config["enabled_roles"]

        self._q_centroids: torch.Tensor | None = None
        self._k_centroids: torch.Tensor | None = None
        self._centroids_initialized = False

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        reason = self._fallback_reason(query, key, value, attn_metadata)
        if reason is not None:
            logger.warning_once(
                "SVG2Impl: falling back to SDPA (%s); prefix=%s q shape=%s k shape=%s",
                reason,
                self.prefix,
                tuple(query.shape),
                tuple(key.shape),
            )
            return self._forward_sdpa(query, key, value, attn_metadata)

        if self._should_use_dense_warmup():
            _debug_log(f"dense warmup prefix={self.prefix} layer={self.layer_idx}")
            return self._forward_sdpa(query, key, value, attn_metadata)

        return self._forward_svg2(query, key, value)

    def _fallback_reason(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
    ) -> str | None:
        if query.device.type != "cuda":
            return "non-CUDA tensors"
        if self.causal:
            return "causal attention unsupported"
        if self.role not in self.enabled_roles:
            return f"role={self.role!r} not enabled for SVG2"
        if self.role != "self":
            return f"role={self.role!r} unsupported"
        if attn_metadata is not None and attn_metadata.attn_mask is not None:
            return "attention masks unsupported"
        if query.shape[1] != key.shape[1] or key.shape[1] != value.shape[1]:
            return "self-attention requires matching Q/K/V sequence lengths"
        if query.shape[2] != key.shape[2] or key.shape[2] != value.shape[2]:
            return "GQA / KV-head mismatch unsupported"
        if query.shape[-1] not in _SUPPORTED_HEAD_DIMS:
            return f"unsupported head_dim={query.shape[-1]}"
        if self.num_q_centroids > query.shape[1] or self.num_k_centroids > key.shape[1]:
            return "configured centroids exceed sequence length"
        return None

    def _should_use_dense_warmup(self) -> bool:
        if self._is_within_warmup(self.first_layers_fp, self.layer_idx, self.num_layers):
            return True

        if not is_forward_context_available():
            return False

        ctx = get_forward_context()
        return self._is_within_warmup(self.first_times_fp, ctx.denoise_step_idx, ctx.num_denoise_steps)

    def _is_within_warmup(
        self,
        configured_value: Any,
        current_index: int | None,
        total_count: int | None,
    ) -> bool:
        if current_index is None:
            return False

        if isinstance(configured_value, bool):
            warmup_count = int(configured_value)
        else:
            numeric = _coerce_float(configured_value, "warmup")
            if 0.0 <= numeric <= 1.0 and total_count is not None:
                warmup_count = math.floor(numeric * total_count)
            else:
                warmup_count = int(numeric)

        return warmup_count > 0 and current_index < warmup_count

    def _forward_sdpa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        attention_mask = None
        if attn_metadata:
            attention_mask = _maybe_reshape_attn_mask(
                query, key, attn_metadata.attn_mask, mask_mode="broadcast_k"
            )
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            scale=self.softmax_scale,
            is_causal=self.causal,
        )
        return out.transpose(1, 2)

    @torch.compiler.disable()
    def _forward_svg2(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        with _nvtx_range("svg2.total"):
            logger.warning_once(
                "SVG2Impl: SVG2/SAP kernel active (prefix=%s, qc=%d, kc=%d, top_p=%.3f)",
                self.prefix,
                self.num_q_centroids,
                self.num_k_centroids,
                self.top_p_kmeans,
            )
            ops = _resolve_svg2_ops()

            with _nvtx_range("svg2.transpose"):
                q = query.transpose(1, 2).contiguous()
                k = key.transpose(1, 2).contiguous()
                v = value.transpose(1, 2).contiguous()
            _debug_log(
                f"svg2 prefix={self.prefix} q={tuple(q.shape)} qc={self.num_q_centroids} kc={self.num_k_centroids}"
            )

            cfg, num_heads, seq_len, dim = q.shape
            if not self._centroids_initialized:
                max_iters = self.kmeans_iter_init
                q_init = None
                k_init = None
            else:
                max_iters = self.kmeans_iter_step
                q_init = self._q_centroids
                k_init = self._k_centroids

            with _nvtx_range("svg2.kmeans_q"):
                qlabels, qcentroids, qcluster_sizes, _ = ops["batch_kmeans_Euclid"](
                    q.reshape(cfg * num_heads, seq_len, dim),
                    n_clusters=self.num_q_centroids,
                    max_iters=max_iters,
                    init_centroids=q_init,
                )
            with _nvtx_range("svg2.kmeans_k"):
                klabels, kcentroids, kcluster_sizes, _ = ops["batch_kmeans_Euclid"](
                    k.reshape(cfg * num_heads, seq_len, dim),
                    n_clusters=self.num_k_centroids,
                    max_iters=max_iters,
                    init_centroids=k_init,
                )
            self._q_centroids = qcentroids
            self._k_centroids = kcentroids
            self._centroids_initialized = True

            with _nvtx_range("svg2.dynamic_map"):
                q_cluster_sizes = qcluster_sizes.view(cfg, num_heads, self.num_q_centroids)
                k_cluster_sizes = kcluster_sizes.view(cfg, num_heads, self.num_k_centroids)
                dynamic_map = ops["identify_dynamic_map"](
                    qcentroids.view(cfg, num_heads, self.num_q_centroids, dim),
                    kcentroids.view(cfg, num_heads, self.num_k_centroids, dim),
                    q_cluster_sizes,
                    k_cluster_sizes,
                    self.top_p_kmeans,
                    self.min_kc_ratio,
                )

            with _nvtx_range("svg2.permute_q"):
                q_perm, q_sorted_indices = ops["permute_tensor_by_labels_triton"](q, qlabels, dim=2)
            with _nvtx_range("svg2.permute_kv"):
                k_perm, k_sorted_indices = ops["permute_tensor_by_labels_triton"](k, klabels, dim=2)
                v_perm, _ = ops["permute_tensor_by_labels_triton"](
                    v, klabels, dim=2, sorted_indices=k_sorted_indices
                )

            with _nvtx_range("svg2.sparse_flashinfer"):
                output_permuted = ops["dynamic_block_sparse_fwd_flashinfer"](
                    q_perm,
                    k_perm,
                    v_perm,
                    dynamic_map,
                    q_cluster_sizes,
                    k_cluster_sizes,
                    is_cpu=False,
                )
            with _nvtx_range("svg2.inverse_permute"):
                out = ops["apply_inverse_permutation_triton"](output_permuted, q_sorted_indices, dim=2)
            with _nvtx_range("svg2.output_transpose"):
                return out.transpose(1, 2).contiguous()
