# SPDX-License-Identifier: Apache-2.0
"""SageSLA attention implementation for vllm-omni diffusion models."""

from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionImpl,
    AttentionMetadata,
)
from vllm_omni.diffusion.attention.backends.sdpa import _maybe_reshape_attn_mask

from .. import envs

logger = init_logger(__name__)

_MIN_SEQ_LEN = 128
_SUPPORTED_HEAD_DIMS = (64, 128)
_COMPARE_SDPA_CALLS = 0


def _debug_enabled() -> bool:
    return os.environ.get("VLLM_SLA_DEBUG", "0") == "1"


def _debug_log(msg: str) -> None:
    if _debug_enabled():
        print(f"[VLLM_SLA_DEBUG] {msg}", file=sys.stderr, flush=True)


class SLAImpl(AttentionImpl):
    """Attention implementation using SageSLA."""

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
        self._prefix = prefix
        self._topk = float(envs.SLA_TOPK)
        self._feature_map = envs.SLA_FEATURE_MAP
        if backend_kwargs:
            if "sla_topk" in backend_kwargs:
                self._topk = float(backend_kwargs.pop("sla_topk"))
            self._feature_map = backend_kwargs.pop(
                "sla_feature_map", self._feature_map
            )
            if backend_kwargs:
                logger.warning(
                    "SLAImpl ignoring backend_kwargs: %s",
                    list(backend_kwargs.keys()),
                )

        from SageSLA import SageSparseLinearAttention

        _debug_log(
            f"init prefix={prefix!r} num_heads={num_heads} num_kv_heads={num_kv_heads} head_size={head_size} causal={causal}"
        )
        self._module = SageSparseLinearAttention(
            head_dim=head_size,
            topk=self._topk,
            feature_map=self._feature_map,
        ).eval()

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        reason = self._fallback_reason(query, key, value, attn_metadata)
        protect_prefix_tokens = self._get_protect_prefix_tokens(attn_metadata)
        _debug_log(
            "forward "
            f"q={tuple(query.shape)} k={tuple(key.shape)} v={tuple(value.shape)} "
            f"protect_prefix_tokens={protect_prefix_tokens} reason={reason or 'sla'}"
        )
        if reason is not None:
            logger.warning_once(
                "SLAImpl: falling back to SDPA (%s); q shape %s",
                reason,
                tuple(query.shape),
            )
            return self._forward_sdpa(query, key, value, attn_metadata)
        out = self._forward_sla(
            query,
            key,
            value,
            protect_prefix_tokens=protect_prefix_tokens,
        )
        self._maybe_compare_with_sdpa(
            out,
            query,
            key,
            value,
            attn_metadata,
            protect_prefix_tokens,
        )
        return out

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
        if attn_metadata is not None and attn_metadata.attn_mask is not None:
            return "attention masks unsupported"
        if key.shape[1] != value.shape[1]:
            return "key/value sequence mismatch"
        if key.shape[2] != value.shape[2]:
            return "key/value head mismatch"
        if query.shape[-1] not in _SUPPORTED_HEAD_DIMS:
            return f"unsupported head_dim={query.shape[-1]}"
        if query.shape[1] < _MIN_SEQ_LEN:
            return f"seq_len<{_MIN_SEQ_LEN}"

        blk_k = 128 if self._get_cuda_arch(query.device.index) == "sm90" else 64
        kv_blocks = math.ceil(key.shape[1] / blk_k)
        if int(self._topk * kv_blocks) < 1:
            return "configured topk resolves to zero KV blocks"
        return None

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
            enable_gqa=q.shape[1] != k.shape[1],
        )
        return out.transpose(1, 2)

    @torch.compiler.disable()
    def _forward_sla(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        protect_prefix_tokens: int = 0,
    ) -> torch.Tensor:
        logger.warning_once(
            "SLAImpl: SageSLA kernel active (topk=%s, feature_map=%s) — q shape %s",
            self._topk,
            self._feature_map,
            tuple(query.shape),
        )
        _debug_log(
            f"forward_sla q={tuple(query.shape)} protect_prefix_tokens={protect_prefix_tokens}"
        )
        if next(self._module.parameters()).device != query.device:
            self._module = self._module.to(device=query.device)
        # SageSLA expects HND = [B, H, N, D], input is NHD = [B, N, H, D].
        q = query.transpose(1, 2).contiguous()
        k = key.transpose(1, 2).contiguous()
        v = value.transpose(1, 2).contiguous()
        if q.shape[1] != k.shape[1]:
            if q.shape[1] % k.shape[1] != 0:
                raise ValueError(
                    "SLAImpl requires q heads to be divisible by kv heads; "
                    f"got q_heads={q.shape[1]} kv_heads={k.shape[1]}"
                )
            repeat = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(repeat, dim=1).contiguous()
            v = v.repeat_interleave(repeat, dim=1).contiguous()
            _debug_log(
                f"expanded kv heads repeat={repeat} -> k={tuple(k.shape)} v={tuple(v.shape)}"
            )
        with torch.no_grad():
            out = self._module(
                q,
                k,
                v,
                protect_prefix_tokens=protect_prefix_tokens,
            )
        return out.transpose(1, 2).contiguous()

    def _maybe_compare_with_sdpa(
        self,
        sla_out: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
        protect_prefix_tokens: int,
    ) -> None:
        if not envs.SLA_COMPARE_SDPA:
            return

        global _COMPARE_SDPA_CALLS
        max_calls = int(envs.SLA_COMPARE_MAX_CALLS or "0")
        if max_calls > 0 and _COMPARE_SDPA_CALLS >= max_calls:
            return
        _COMPARE_SDPA_CALLS += 1

        sdpa_out = self._forward_sdpa(query, key, value, attn_metadata)
        sla_flat = sla_out.reshape(-1).float()
        sdpa_flat = sdpa_out.reshape(-1).float()
        cosine = float(F.cosine_similarity(sla_flat, sdpa_flat, dim=0).item())
        diff = (sla_out.float() - sdpa_out.float()).abs()
        mean_abs = float(diff.mean().item())
        max_abs = float(diff.max().item())

        blk_k = 128 if self._get_cuda_arch(query.device.index) == "sm90" else 64
        kv_blocks = math.ceil(key.shape[1] / blk_k)
        requested_blocks = min(kv_blocks, int(self._topk * kv_blocks))
        logger.warning(
            "SLAImpl compare[%d]: prefix=%s topk=%.4f kv_blocks=%d requested_blocks=%d protect_prefix_tokens=%d "
            "q=%s k=%s cosine=%.8f mean_abs=%.8e max_abs=%.8e",
            _COMPARE_SDPA_CALLS,
            self._prefix or "<none>",
            self._topk,
            kv_blocks,
            requested_blocks,
            protect_prefix_tokens,
            tuple(query.shape),
            tuple(key.shape),
            cosine,
            mean_abs,
            max_abs,
        )

    @staticmethod
    def _get_protect_prefix_tokens(attn_metadata: AttentionMetadata | None) -> int:
        if attn_metadata is None:
            return 0
        protect_prefix_tokens = attn_metadata.extra.get("sla_protect_prefix_tokens", 0)
        if protect_prefix_tokens is None:
            return 0
        return max(0, int(protect_prefix_tokens))

    @staticmethod
    def _get_cuda_arch(device_index: int | None) -> str:
        major, minor = torch.cuda.get_device_capability(device_index)
        return f"sm{major}{minor}"
