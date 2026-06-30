# SPDX-License-Identifier: Apache-2.0
"""SpargeAttn attention implementation for vllm-omni diffusion models.

Wraps the prebuilt ``spas_sage_attn`` block-sparse CUDA kernels and the
``auto_round_kernel`` XPU sparse kernels with:
- NHD↔HND layout transpose (vllm-omni uses NHD, SpargeAttn uses HND)
- A guard chain that falls back to torch SDPA whenever SpargeAttn's hard
  requirements are not met (fp32, cross-attention, seq_len < 128, head_dim
  not in {64,128}, or an attention mask — which SpargeAttn silently ignores).
"""

import inspect

import torch
import torch.nn.functional as F

from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionImpl,
    AttentionMetadata,
)
from vllm_omni.diffusion.forward_context import (
    get_forward_context,
    is_forward_context_available,
)

from .. import envs

logger = init_logger(__name__)

# SpargeAttn hard constraints (see SpargeAttn/spas_sage_attn/core.py asserts and
# inference_examples/modify_model/modify_wan.py guard).
_MIN_SEQ_LEN = 128
_SUPPORTED_HEAD_DIMS = (64, 128)

# Lazy-loaded XPU kernel reference.
_xpu_sparse_fn = None
_xpu_sparse_has_tensor_layout = False


def _get_xpu_sparse_fn():
    global _xpu_sparse_fn, _xpu_sparse_has_tensor_layout
    if _xpu_sparse_fn is not None:
        return _xpu_sparse_fn

    try:
        from auto_round_kernel import ARK
        ark = ARK()
    except ImportError:
        import auto_round_kernel as ark

    fn = getattr(ark, "sparge_sage2_attn_meansim_topk_xpu", None)
    if fn is None:
        raise ImportError(
            "auto_round_kernel does not expose sparge_sage2_attn_meansim_topk_xpu. "
            "Update auto-round-lib or set SPARGE_ARK_PATH to the correct ARK build."
        )
    _xpu_sparse_fn = fn
    _xpu_sparse_has_tensor_layout = "tensor_layout" in inspect.signature(fn).parameters
    return fn


class SpargeAttnImpl(AttentionImpl):
    """Attention implementation using the SpargeAttn block-sparse kernel."""

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

        # Configuration from env vars (unified across CUDA and XPU).
        self._mode = envs.SPARGE_MODE
        self._topk = float(envs.SPARGE_TOPK)
        self._cdfthreshd = float(envs.SPARGE_CDFTHRESHD)
        self._smooth_k = envs.SPARGE_SMOOTH_K
        self._simthreshd1 = float(envs.SPARGE_SIMTHRESHD1)
        self._attention_sink = envs.SPARGE_ATTENTION_SINK
        self._k_quant_granularity = int(envs.SPARGE_K_QUANT_GRANULARITY)
        self._dense_steps = int(envs.SPARGE_DENSE_STEPS)

        # Override from backend_kwargs if provided.
        if backend_kwargs:
            self._mode = backend_kwargs.pop("sparge_mode", self._mode)
            if "sparge_topk" in backend_kwargs:
                self._topk = float(backend_kwargs.pop("sparge_topk"))
            if "sparge_cdfthreshd" in backend_kwargs:
                self._cdfthreshd = float(backend_kwargs.pop("sparge_cdfthreshd"))
            if "sparge_smooth_k" in backend_kwargs:
                self._smooth_k = backend_kwargs.pop("sparge_smooth_k")
            if "sparge_simthreshd1" in backend_kwargs:
                self._simthreshd1 = float(backend_kwargs.pop("sparge_simthreshd1"))
            if "sparge_attention_sink" in backend_kwargs:
                self._attention_sink = backend_kwargs.pop("sparge_attention_sink")
            if "sparge_k_quant_granularity" in backend_kwargs:
                self._k_quant_granularity = int(
                    backend_kwargs.pop("sparge_k_quant_granularity")
                )
            if backend_kwargs:
                logger.warning(
                    "SpargeAttnImpl ignoring backend_kwargs: %s",
                    list(backend_kwargs.keys()),
                )

    # ------------------------------------------------------------------
    # CUDA forward
    # ------------------------------------------------------------------

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
        *,
        config_override: str | None = None,
    ) -> torch.Tensor:
        has_mask = attn_metadata is not None and attn_metadata.attn_mask is not None
        if (
            query.dtype == torch.float32
            or query.shape[1] != key.shape[1]
            or query.shape[1] < _MIN_SEQ_LEN
            or query.shape[-1] not in _SUPPORTED_HEAD_DIMS
            or has_mask
        ):
            return self._forward_sdpa(query, key, value)
        return self._forward_sparge_cuda(query, key, value)

    # ------------------------------------------------------------------
    # XPU forward
    # ------------------------------------------------------------------

    def forward_xpu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
    ) -> torch.Tensor:
        has_mask = attn_metadata is not None and attn_metadata.attn_mask is not None
        if (
            query.dtype == torch.float32
            or query.shape[1] != key.shape[1]
            or query.shape[1] < _MIN_SEQ_LEN
            or query.shape[-1] not in _SUPPORTED_HEAD_DIMS
            or has_mask
        ):
            return self._forward_sdpa(query, key, value)
        return self._forward_sparge_xpu(query, key, value)

    # ------------------------------------------------------------------
    # Shared SDPA fallback
    # ------------------------------------------------------------------

    def _forward_sdpa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """SDPA fallback when SpargeAttn's constraints are not met."""
        # Input is NHD [B, N, H, D]; SDPA expects [B, H, N, D].
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v, scale=self.softmax_scale, is_causal=False
        )
        return out.transpose(1, 2)  # back to NHD

    # ------------------------------------------------------------------
    # CUDA sparse kernel
    # ------------------------------------------------------------------

    @torch.compiler.disable()
    def _forward_sparge_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        from spas_sage_attn import (
            spas_sage2_attn_meansim_cuda,
            spas_sage2_attn_meansim_topk_cuda,
        )

        logger.warning_once(
            "SpargeAttnImpl: CUDA SpargeAttn kernel active (mode=%s, topk=%s, "
            "cdfthreshd=%s) — q shape %s",
            self._mode,
            self._topk,
            self._cdfthreshd,
            tuple(query.shape),
        )
        # SpargeAttn expects HND = [B, H, N, D], input is NHD = [B, N, H, D].
        q = query.transpose(1, 2).contiguous()
        k = key.transpose(1, 2).contiguous()
        v = value.transpose(1, 2).contiguous()

        if self._mode == "cdfthreshd":
            out = spas_sage2_attn_meansim_cuda(
                q,
                k,
                v,
                is_causal=self.causal,
                scale=self.softmax_scale,
                cdfthreshd=self._cdfthreshd,
                tensor_layout="HND",
            )
        else:
            out = spas_sage2_attn_meansim_topk_cuda(
                q,
                k,
                v,
                is_causal=self.causal,
                scale=self.softmax_scale,
                topk=self._topk,
                tensor_layout="HND",
            )
        return out.transpose(1, 2)  # back to NHD

    # ------------------------------------------------------------------
    # XPU sparse kernel
    # ------------------------------------------------------------------

    @torch.compiler.disable()
    def _forward_sparge_xpu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        fn = _get_xpu_sparse_fn()

        # Use dense (topk=1.0) for early denoising steps if configured.
        topk = self._topk
        if self._dense_steps > 0 and is_forward_context_available():
            step_idx = get_forward_context().denoise_step_idx
            if step_idx is not None and step_idx < self._dense_steps:
                topk = 1.0

        logger.warning_once(
            "SpargeAttnImpl: XPU SpargeAttn kernel active (topk=%s, smooth_k=%s, "
            "simthreshd1=%s, attention_sink=%s, k_quant_granularity=%s, "
            "dense_steps=%s) — q shape %s",
            self._topk,
            self._smooth_k,
            self._simthreshd1,
            self._attention_sink,
            self._k_quant_granularity,
            self._dense_steps,
            tuple(query.shape),
        )

        orig_dtype = query.dtype
        q = query.to(torch.float16).contiguous() if orig_dtype != torch.float16 else query.contiguous()
        k = key.to(torch.float16).contiguous() if orig_dtype != torch.float16 else key.contiguous()
        v = value.to(torch.float16).contiguous() if orig_dtype != torch.float16 else value.contiguous()

        sparse_kwargs = {
            "is_causal": self.causal,
            "scale": self.softmax_scale,
            "topk": topk,
            "smooth_k": self._smooth_k,
            "simthreshd1": self._simthreshd1,
            "attention_sink": self._attention_sink,
            "k_quant_granularity": self._k_quant_granularity,
        }

        if _xpu_sparse_has_tensor_layout:
            out = fn(q, k, v, tensor_layout="NHD", **sparse_kwargs)
        else:
            out = fn(q, k, v, **sparse_kwargs)

        if orig_dtype != torch.float16:
            out = out.to(orig_dtype)
        return out
