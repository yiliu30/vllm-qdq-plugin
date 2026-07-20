# SPDX-License-Identifier: Apache-2.0
"""SpargeAttn attention implementation for vllm-omni diffusion models.

Wraps the prebuilt ``spas_sage_attn`` block-sparse CUDA kernels with:
- NHD↔HND layout transpose (vllm-omni uses NHD, SpargeAttn uses HND)
- A guard chain that falls back to torch SDPA whenever SpargeAttn's hard
  requirements are not met (fp32, seq_len < 128, head_dim not in {64,128}, or
  an attention mask — which SpargeAttn silently ignores).
"""

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

# Imported lazily (this module is only imported when SpargeAttn is selected), so a
# missing/unbuilt spas_sage_attn surfaces a clear ImportError only at that point.
from spas_sage_attn import (
    spas_sage2_attn_meansim_cuda,
    spas_sage2_attn_meansim_topk_cuda,
)

from .. import envs

logger = init_logger(__name__)

# SpargeAttn hard constraints (see SpargeAttn/spas_sage_attn/core.py asserts and
# inference_examples/modify_model/modify_wan.py guard).
_MIN_SEQ_LEN = 128
_SUPPORTED_HEAD_DIMS = (64, 128)


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
        self.requires_gqa = num_heads != num_kv_heads
        self.prefix = prefix or "<unknown>"

        # Configuration from env vars.
        self._mode = envs.SPARGE_MODE
        self._topk = float(envs.SPARGE_TOPK)
        self._cdfthreshd = float(envs.SPARGE_CDFTHRESHD)
        self._debug_context = bool(envs.SPARGE_DEBUG_CONTEXT)

        # Override from backend_kwargs if provided.
        if backend_kwargs:
            backend_kwargs = backend_kwargs.copy()
            self._mode = backend_kwargs.pop("sparge_mode", self._mode)
            if "sparge_topk" in backend_kwargs:
                self._topk = float(backend_kwargs.pop("sparge_topk"))
            if "sparge_cdfthreshd" in backend_kwargs:
                self._cdfthreshd = float(backend_kwargs.pop("sparge_cdfthreshd"))
            if backend_kwargs:
                logger.warning(
                    "SpargeAttnImpl ignoring backend_kwargs: %s",
                    list(backend_kwargs.keys()),
                )

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
        *,
        config_override: str | None = None,
    ) -> torch.Tensor:
        # config_override is accepted for signature-compat with the sage3 routing
        # patch (which guards on isinstance Sage3TritonImpl and never targets us);
        # SpargeAttn has no per-(layer,step) config, so it is ignored.
        #
        # Input layout: NHD = [B, N, H, D]. Fall back to SDPA whenever SpargeAttn's
        # requirements are not met.
        has_mask = attn_metadata is not None and attn_metadata.attn_mask is not None
        if (
            query.dtype == torch.float32
            or query.shape[1] < _MIN_SEQ_LEN  # seq_len too short
            or query.shape[-1] not in _SUPPORTED_HEAD_DIMS  # head_dim unsupported
            or has_mask  # SpargeAttn silently ignores masks
        ):
            return self._forward_sdpa(query, key, value, attn_metadata)
        return self._forward_sparge(query, key, value)

    def _forward_sdpa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        """SDPA fallback when SpargeAttn's constraints are not met."""
        attention_mask = None
        if attn_metadata:
            attention_mask = _maybe_reshape_attn_mask(
                query, key, attn_metadata.attn_mask, mask_mode="broadcast_k"
            )
        # Input is NHD [B, N, H, D]; SDPA expects [B, H, N, D].
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        enable_gqa = q.shape[1] != k.shape[1]
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            scale=self.softmax_scale,
            is_causal=self.causal,
            enable_gqa=enable_gqa,
        )
        return out.transpose(1, 2)  # back to NHD

    @torch.compiler.disable()
    def _forward_sparge(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Forward using the SpargeAttn kernel."""
        step_idx = None
        if is_forward_context_available():
            step_idx = get_forward_context().denoise_step_idx
        logger.warning_once(
            "SpargeAttnImpl: SpargeAttn kernel active (mode=%s, topk=%s, "
            "cdfthreshd=%s, prefix=%s) — q shape %s",
            self._mode,
            self._topk,
            self._cdfthreshd,
            self.prefix,
            tuple(query.shape),
        )
        if self._debug_context:
            print(
                "SPARGE_CONTEXT "
                f"prefix={self.prefix} "
                f"step={step_idx} "
                f"mode={self._mode} "
                f"q_tokens={query.shape[1]} "
                f"k_tokens={key.shape[1]} "
                f"q_heads={query.shape[2]} "
                f"kv_heads={key.shape[2]}"
            )
        # SpargeAttn expects HND = [B, H, N, D], input is NHD = [B, N, H, D].
        q = query.transpose(1, 2).contiguous()
        k = key.transpose(1, 2).contiguous()
        v = value.transpose(1, 2).contiguous()
        if q.shape[1] != k.shape[1]:
            if q.shape[1] % k.shape[1] != 0:
                logger.warning_once(
                    "SpargeAttnImpl: q heads (%d) not divisible by kv heads (%d); "
                    "falling back to SDPA for this shape",
                    q.shape[1],
                    k.shape[1],
                )
                return self._forward_sdpa(query, key, value)
            repeat = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(repeat, dim=1).contiguous()
            v = v.repeat_interleave(repeat, dim=1).contiguous()

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
