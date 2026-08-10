"""Public MXAttention forward API and hardware dispatch."""

from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F

from .reference import DEFAULT_QMAX, normalized_fwht, sdpa_reference

SUPPORTED_MODES = {
    "fp16_reference",
    "ocp_mxfp4_direct",
    "uos_only",
    "pnq_only",
    "uos_pnq",
    "mxattention_full",
}

logger = logging.getLogger(__name__)
_HARDWARE_DISABLED = False


def _pad_seq(x: torch.Tensor, length: int) -> torch.Tensor:
    if x.shape[-2] == length:
        return x.contiguous()
    return F.pad(x, (0, 0, 0, length - x.shape[-2])).contiguous()


def _hardware_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    sm_scale: float,
    causal: bool,
    qmax: float,
    pnq: bool,
) -> torch.Tensor:
    from vllm_qdq_plugin.sage3_attn.sage3.mxfp4_hw_kernel import (
        mxfp4_flash_attention,
        mxfp4_pnq_flash_attention,
    )
    from .reference import quantize_mxfp4_uos

    b, h, m, d = q.shape
    n = k.shape[-2]
    if d != 128:
        raise ValueError(f"MXAttention initially requires head_dim=128, got {d}")
    if h != k.shape[1] or h != v.shape[1] or n != v.shape[-2]:
        raise ValueError("MXAttention initial path requires equal Q/K/V head and sequence counts")
    mp = math.ceil(m / 128) * 128
    np = math.ceil(n / 64) * 64
    q = _pad_seq(q, mp)
    k = _pad_seq(k, np)
    v = _pad_seq(v, np)

    q_p, q_s = quantize_mxfp4_uos(q, qmax=qmax)
    k_p, k_s = quantize_mxfp4_uos(k, qmax=qmax)
    v_t = v.permute(0, 1, 3, 2).contiguous()
    v_p, v_s = quantize_mxfp4_uos(v_t, qmax=qmax)

    if pnq:
        out = mxfp4_pnq_flash_attention(
            q_p,
            k_p,
            v_p,
            q_s,
            k_s,
            v_s,
            valid_q_len=m,
            valid_k_len=n,
            causal=causal,
            sm_scale=sm_scale,
            qmax=qmax,
        )
    else:
        # The existing Sage3 kernel remains the direct MXFP4 baseline.  It
        # has no PNQ update, so this branch is intentionally separate.
        out = mxfp4_flash_attention(
            q_p,
            k_p,
            v_p,
            q_s,
            k_s,
            v_s,
            causal=causal,
            sm_scale=sm_scale,
        )[:, :, :m, :]
    return out.to(q.dtype)


def mxattention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    sm_scale: float | None = None,
    causal: bool = False,
    qmax: float = DEFAULT_QMAX,
    use_hadamard: bool = True,
    output_dtype: torch.dtype | None = None,
    mode: str = "mxattention_full",
    allow_fallback: bool = True,
) -> torch.Tensor:
    """Run MXAttention on tensors shaped ``[B, H, N, D]``."""
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"unknown MXAttention mode {mode!r}; choose from {sorted(SUPPORTED_MODES)}")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("MXAttention expects [B, H, N, D] tensors")
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])

    if mode == "fp16_reference":
        out = sdpa_reference(q, k, v, sm_scale=sm_scale, causal=causal)
    else:
        if mode == "ocp_mxfp4_direct":
            effective_qmax, pnq, hadamard = 6.0, False, False
        elif mode == "uos_only":
            effective_qmax, pnq, hadamard = qmax, False, False
        elif mode == "pnq_only":
            effective_qmax, pnq, hadamard = 6.0, True, False
        elif mode == "uos_pnq":
            effective_qmax, pnq, hadamard = qmax, True, False
        else:
            effective_qmax, pnq, hadamard = qmax, True, use_hadamard
        q_fallback, k_fallback = q, k
        if hadamard:
            q = normalized_fwht(q)
            k = normalized_fwht(k)
        global _HARDWARE_DISABLED
        if _HARDWARE_DISABLED:
            out = sdpa_reference(q_fallback, k_fallback, v, sm_scale=sm_scale, causal=causal)
        else:
            try:
                out = _hardware_attention(
                    q,
                    k,
                    v,
                    sm_scale=sm_scale,
                    causal=causal,
                    qmax=effective_qmax,
                    pnq=pnq,
                )
            except (ImportError, RuntimeError, ValueError) as exc:
                if not allow_fallback:
                    raise
                if isinstance(exc, RuntimeError):
                    # A Triton compiler/runtime failure is process-wide for a
                    # given environment.  Avoid retrying the same failed JIT
                    # for every Wan block; SDPA remains the safe fallback.
                    _HARDWARE_DISABLED = True
                logger.warning(
                    "MXAttention hardware path unavailable; using SDPA fallback: %s",
                    exc,
                )
                out = sdpa_reference(q_fallback, k_fallback, v, sm_scale=sm_scale, causal=causal)
    return out.to(output_dtype or q.dtype)
