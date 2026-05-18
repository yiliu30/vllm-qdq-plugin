# SPDX-License-Identifier: Apache-2.0
"""Sage3 Triton attention implementation for vllm-omni diffusion models.

Imports the sage3 standalone kernel and wraps it with:
- NHD↔HND layout transpose (vllm-omni uses NHD, sage3 uses HND)
- Cross-attention fallback to torch SDPA (different Q/K seq lengths)
"""

import sys

import torch
import torch.nn.functional as F

from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionImpl,
    AttentionMetadata,
)

from .. import envs

logger = init_logger(__name__)

# ── sage3 kernel import ──

_sage3_fn = None


def _load_sage3():
    """Lazy-load sage3 standalone from configured path."""
    global _sage3_fn
    if _sage3_fn is not None:
        return _sage3_fn

    sage3_path = envs.SAGE3_STANDALONE_PATH
    if sage3_path not in sys.path:
        sys.path.insert(0, sage3_path)

    from sage3 import sageattn3_standalone

    _sage3_fn = sageattn3_standalone
    logger.info("[sage3_attn plugin]: loaded sage3 kernel from %s", sage3_path)
    return _sage3_fn


class Sage3TritonImpl(AttentionImpl):
    """Attention implementation using sage3 Triton kernel."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        backend_kwargs: dict | None = None,
        **extra_impl_args,
    ) -> None:
        self.causal = causal
        self.softmax_scale = softmax_scale

        # Configuration from env vars
        self._config = envs.SAGE3_QUANT_FORMAT
        self._acc_dtype = envs.SAGE3_ACC_DTYPE

        # Override from backend_kwargs if provided
        if backend_kwargs:
            self._config = backend_kwargs.pop("sage3_config", self._config)
            self._acc_dtype = backend_kwargs.pop("sage3_acc_dtype", self._acc_dtype)
            if backend_kwargs:
                logger.warning(
                    "Sage3TritonImpl ignoring backend_kwargs: %s",
                    list(backend_kwargs.keys()),
                )

        # Eagerly load sage3
        _load_sage3()

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
    ) -> torch.Tensor:
        # Input layout: NHD = [B, N, H, D]
        return self._forward_sage3(query, key, value)

    @torch.compiler.disable()
    def _forward_sage3(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Forward using sage3 Triton kernel."""
        # sage3 expects HND = [B, H, N, D], input is NHD = [B, N, H, D]
        q = query.transpose(1, 2).contiguous()
        k = key.transpose(1, 2).contiguous()
        v = value.transpose(1, 2).contiguous()

        out = _sage3_fn(
            q, k, v,
            config=self._config,
            is_causal=self.causal,
            sm_scale=self.softmax_scale,
            acc_dtype=self._acc_dtype,
        )
        return out.transpose(1, 2)  # back to NHD
