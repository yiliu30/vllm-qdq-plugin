"""vLLM-Omni AttentionImpl wrapper for MXAttention."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import AttentionImpl, AttentionMetadata
from vllm_omni.diffusion.attention.backends.sdpa import _maybe_reshape_attn_mask

from .. import envs
from .api import mxattention_forward

logger = init_logger(__name__)


class MXAttentionImpl(AttentionImpl):
    """MXAttention with explicit SDPA fallback for unsupported requests."""

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
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.head_size = head_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.prefix = prefix
        self.mode = envs.MXATTENTION_MODE
        self.qmax = float(envs.MXATTENTION_QMAX)
        self.use_hadamard = envs.MXATTENTION_USE_HADAMARD
        if backend_kwargs:
            logger.warning("MXAttentionImpl ignoring backend_kwargs: %s", list(backend_kwargs))

    def _sdpa(self, query, key, value, attn_metadata):
        mask = None
        if attn_metadata is not None and attn_metadata.attn_mask is not None:
            mask = _maybe_reshape_attn_mask(query, key, attn_metadata.attn_mask, mask_mode="broadcast_k")
        return F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=mask,
            dropout_p=0.0,
            scale=self.softmax_scale,
            is_causal=self.causal,
            enable_gqa=query.shape[2] != key.shape[2],
        ).transpose(1, 2)

    def forward_cuda(self, query, key, value, attn_metadata: AttentionMetadata | None = None):
        unsupported = (
            query.dtype not in (torch.float16, torch.bfloat16)
            or query.shape[-1] != 128
            or query.shape[1] != key.shape[1]
            or query.shape[2] != key.shape[2]
            or key.shape[1] != value.shape[1]
            or key.shape[2] != value.shape[2]
            or (attn_metadata is not None and attn_metadata.attn_mask is not None)
        )
        if unsupported or self.mode == "fp16_reference":
            return self._sdpa(query, key, value, attn_metadata)
        return self._forward_mxattention(query, key, value)

    @torch.compiler.disable()
    def _forward_mxattention(self, query, key, value):
        """Run the optional Triton path outside torch.compile.

        The MXFP4 kernel is JIT-compiled by Triton.  Keeping this call in an
        eager island lets the runtime catch a kernel/compiler incompatibility
        and use SDPA, instead of turning it into a graph-compilation failure.
        """
        logger.warning_once(
            "MXAttention active for %s: mode=%s qmax=%s hadamard=%s shape=%s",
            self.prefix or "<unknown>",
            self.mode,
            self.qmax,
            self.use_hadamard,
            tuple(query.shape),
        )
        return mxattention_forward(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            sm_scale=self.softmax_scale,
            causal=self.causal,
            qmax=self.qmax,
            use_hadamard=self.use_hadamard,
            output_dtype=query.dtype,
            mode=self.mode,
            allow_fallback=True,
        ).transpose(1, 2)
