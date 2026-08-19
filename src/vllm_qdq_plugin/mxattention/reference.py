"""CPU/PyTorch reference pieces for MXAttention.

This module is intentionally independent from Triton so it can serve as the
golden reference for packed E2M1/E8M0 codes, UOS, and PNQ.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
DEFAULT_QMAX = 7.25
GROUP_SIZE = 32


def _e2m1_code(abs_x: torch.Tensor) -> torch.Tensor:
    """Nearest-even positive E2M1 code, including exact midpoint behavior."""
    code = torch.zeros_like(abs_x, dtype=torch.uint8)
    code = torch.where(abs_x > 0.25, torch.ones_like(code), code)
    code = torch.where(abs_x >= 0.75, torch.full_like(code, 2), code)
    code = torch.where(abs_x > 1.25, torch.full_like(code, 3), code)
    code = torch.where(abs_x >= 1.75, torch.full_like(code, 4), code)
    code = torch.where(abs_x > 2.5, torch.full_like(code, 5), code)
    code = torch.where(abs_x >= 3.5, torch.full_like(code, 6), code)
    code = torch.where(abs_x > 5.0, torch.full_like(code, 7), code)
    return code


def _e2m1_decode(code: torch.Tensor) -> torch.Tensor:
    values = torch.tensor(E2M1_VALUES, device=code.device, dtype=torch.float32)
    return values[code.to(torch.long)]


def _e8m0_to_float(code: torch.Tensor) -> torch.Tensor:
    exponent = code.to(torch.int32) - 127
    return torch.exp2(exponent.to(torch.float32))


def quantize_mxfp4_uos(
    x: torch.Tensor,
    *,
    group_size: int = GROUP_SIZE,
    qmax: float = DEFAULT_QMAX,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize the last dimension to packed OCP MXFP4 using UOS.

    Returns packed bytes and uint8 E8M0 scale codes.  The low nibble stores
    the first logical value in each pair.
    """
    if x.shape[-1] % group_size or x.shape[-1] % 2:
        raise ValueError("MXFP4 quantization requires an even dimension divisible by group_size")
    if not torch.is_floating_point(x):
        raise TypeError(f"expected floating input, got {x.dtype}")
    xf = x.float()
    groups = xf.reshape(*xf.shape[:-1], -1, group_size)
    amax = groups.abs().amax(dim=-1)
    nonzero = amax > 0
    exponent = torch.zeros_like(amax, dtype=torch.int32)
    safe = torch.where(nonzero, amax, torch.ones_like(amax))
    exponent = torch.ceil(torch.log2(safe / qmax)).to(torch.int32)
    exponent = torch.where(nonzero, exponent, torch.zeros_like(exponent))
    exponent = exponent.clamp(-127, 127)
    scale_codes = torch.where(nonzero, exponent + 127, torch.full_like(exponent, 127)).to(torch.uint8)
    scale = _e8m0_to_float(scale_codes)
    normalized = groups / scale.unsqueeze(-1)
    normalized = normalized.reshape_as(xf)
    magnitude = _e2m1_code(normalized.abs())
    sign = (normalized < 0).to(torch.uint8) << 3
    nibbles = sign | magnitude
    packed = nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)
    return packed, scale_codes


def dequantize_mxfp4_uos(
    packed: torch.Tensor,
    scale_codes: torch.Tensor,
    *,
    group_size: int = GROUP_SIZE,
) -> torch.Tensor:
    """Decode packed MXFP4/E8M0 values to FP32."""
    even = packed & 0x0F
    odd = (packed >> 4) & 0x0F
    codes = torch.empty((*packed.shape[:-1], packed.shape[-1] * 2), device=packed.device, dtype=torch.uint8)
    codes[..., 0::2] = even
    codes[..., 1::2] = odd
    values = _e2m1_decode(codes & 0x7)
    signs = torch.where((codes & 0x8) != 0, -1.0, 1.0)
    scale = _e8m0_to_float(scale_codes).repeat_interleave(group_size, dim=-1)
    return values * signs * scale[..., : values.shape[-1]]


def normalized_fwht(x: torch.Tensor) -> torch.Tensor:
    """Apply a normalized Walsh-Hadamard transform on the last dimension."""
    d = x.shape[-1]
    if d & (d - 1):
        raise ValueError("Hadamard rotation requires a power-of-two head dimension")
    y = x.float()
    prefix = x.shape[:-1]
    h = 1
    while h < d:
        y = y.reshape(*prefix, -1, 2 * h)
        left, right = y[..., :h], y[..., h:]
        y = torch.cat((left + right, left - right), dim=-1)
        h *= 2
    return (y.reshape_as(x).float() / math.sqrt(d)).to(x.dtype)


def materialized_pnq_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    sm_scale: float | None = None,
    causal: bool = False,
    qmax: float = DEFAULT_QMAX,
    use_hadamard: bool = False,
) -> torch.Tensor:
    """Materialized reference for UOS Q/K/V plus PNQ probability updates."""
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
    if use_hadamard:
        q = normalized_fwht(q)
        k = normalized_fwht(k)
    q_p, q_s = quantize_mxfp4_uos(q, qmax=qmax)
    k_p, k_s = quantize_mxfp4_uos(k, qmax=qmax)
    v_t = v.transpose(-1, -2).contiguous()
    v_p, v_s = quantize_mxfp4_uos(v_t, qmax=qmax)
    qd = dequantize_mxfp4_uos(q_p, q_s)
    kd = dequantize_mxfp4_uos(k_p, k_s)
    vd = dequantize_mxfp4_uos(v_p, v_s).transpose(-1, -2).contiguous()
    scores = torch.matmul(qd, kd.transpose(-1, -2)) * sm_scale
    if causal:
        mask = torch.triu(torch.ones(scores.shape[-2:], device=scores.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, -torch.inf)
    row_max = scores.max(dim=-1, keepdim=True).values
    p_tilde = torch.exp(scores - row_max)
    padded_n = math.ceil(p_tilde.shape[-1] / GROUP_SIZE) * GROUP_SIZE
    if padded_n != p_tilde.shape[-1]:
        p_tilde = F.pad(p_tilde, (0, padded_n - p_tilde.shape[-1]))
    p_p, p_s = quantize_mxfp4_uos(p_tilde, qmax=qmax)
    p_hat = dequantize_mxfp4_uos(p_p, p_s)[..., :scores.shape[-1]]
    return torch.matmul(p_hat, vd) / p_hat.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)


def sdpa_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, sm_scale: float, causal: bool) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=0.0,
        scale=sm_scale,
        is_causal=causal,
    )
