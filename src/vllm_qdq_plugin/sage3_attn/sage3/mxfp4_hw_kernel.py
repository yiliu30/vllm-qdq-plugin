"""
MXFP4 (E2M1) Flash Attention on NVIDIA B200 (sm_100)
=====================================================

Educational standalone implementation of Flash Attention v2 using:
- Data format: E2M1 (4-bit FP, packed 2 per byte) for Q, K, V, and P
- Scale format: E8M0 (uint8, pure power-of-2 scales)
- Scale grouping: group_size=32 along reduction dimension
- Hardware: NVIDIA Blackwell (sm_100); uses native tl.dot_scaled tensor-core
  MMA for the MXAttention UOS/PNQ path
- In-kernel P quantization: uses PTX cvt.rn.satfinite.e2m1x2.f32

The kernel computes:
    O = softmax(Q @ K^T * sm_scale) @ V

where Q, K, V are in MXFP4 format (E2M1 packed data + E8M0 block scales).
"""

import torch
import triton
import triton.language as tl

from .triton_utils import compute_p_scale_inv, compute_uos_p_scale_inv



# ============================================================================
# Helper Functions
# ============================================================================


def fp8e8m0_to_float32(scale_uint8: torch.Tensor) -> torch.Tensor:
    """Convert E8M0 scale (uint8) to float32: value = 2^(x - 127)."""
    x = scale_uint8.to(torch.int32)
    x = x << 23
    return x.view(torch.float32)


# E2M1 representable positive values (magnitude only):
# 0b000=0, 0b001=0.5, 0b010=1.0, 0b011=1.5, 0b100=2.0, 0b101=3.0, 0b110=4.0, 0b111=6.0
_E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


def _float_to_e2m1_nibble(x: torch.Tensor) -> torch.Tensor:
    """Convert float tensor to E2M1 4-bit encoding (uint8, values 0-15).
    sign(1)|exp(2)|man(1). Max representable: ±6.0."""
    sign = (x < 0).to(torch.uint8) << 3
    ax = x.abs().clamp(max=6.0)

    # Thresholds for rounding to nearest E2M1 value
    # 0→[0,0.25), 0.5→[0.25,0.75), 1.0→[0.75,1.25), 1.5→[1.25,1.75),
    # 2.0→[1.75,2.5), 3.0→[2.5,3.5), 4.0→[3.5,5.0), 6.0→[5.0,∞)
    code = torch.zeros_like(ax, dtype=torch.uint8)
    code = torch.where(ax >= 0.25, torch.ones_like(code), code)
    code = torch.where(ax >= 0.75, torch.full_like(code, 2), code)
    code = torch.where(ax >= 1.25, torch.full_like(code, 3), code)
    code = torch.where(ax >= 1.75, torch.full_like(code, 4), code)
    code = torch.where(ax >= 2.5, torch.full_like(code, 5), code)
    code = torch.where(ax >= 3.5, torch.full_like(code, 6), code)
    code = torch.where(ax >= 5.0, torch.full_like(code, 7), code)

    return sign | code


def _e2m1_nibble_to_float(nibble: torch.Tensor) -> torch.Tensor:
    """Convert E2M1 4-bit encoding back to float32."""
    sign = ((nibble >> 3) & 1).float() * -2.0 + 1.0
    mag_code = (nibble & 0x7).long()
    values = _E2M1_VALUES.to(nibble.device)
    mag = values[mag_code]
    return sign * mag


def quantize_to_mxfp4(x: torch.Tensor, group_size: int = 32) -> tuple:
    """
    Quantize float tensor to MXFP4 (E2M1) with E8M0 block scales.
    Groups along the last dimension with group_size=32.

    Returns:
        x_packed: [..., K//2] uint8 (two e2m1 values per byte, low nibble=even, high nibble=odd)
        scales: [..., K // group_size] uint8 E8M0 scales
    """
    assert x.shape[-1] % group_size == 0
    assert x.shape[-1] % 2 == 0

    original_shape = x.shape
    x_f32 = x.float()

    *batch_dims, K = x_f32.shape
    num_groups = K // group_size
    x_grouped = x_f32.reshape(*batch_dims, num_groups, group_size)

    # Compute per-group amax
    amax = x_grouped.abs().amax(dim=-1)  # [..., num_groups]
    e2m1_max = 6.0

    # E8M0 scale: 2^ceil(log2(amax / 6.0)), clamped
    amax_safe = amax.clamp(min=e2m1_max * (2**-126))
    log2_ratio = torch.ceil(torch.log2(amax_safe / e2m1_max))
    log2_ratio = log2_ratio.clamp(-127, 127)
    e8m0_encoding = (log2_ratio + 127).to(torch.uint8)

    # Compute float scale and divide
    scale_f32 = fp8e8m0_to_float32(e8m0_encoding)
    scale_expanded = scale_f32.unsqueeze(-1).expand_as(x_grouped)
    x_scaled = x_grouped / scale_expanded  # values in [-6, 6]
    x_scaled = x_scaled.reshape(original_shape)

    # Encode to E2M1 nibbles
    nibbles = _float_to_e2m1_nibble(x_scaled)  # [..., K] uint8 (4-bit values)

    # Pack pairs: low nibble = even index, high nibble = odd index
    even = nibbles[..., 0::2]  # [..., K//2]
    odd = nibbles[..., 1::2]   # [..., K//2]
    packed = (odd << 4) | even  # [..., K//2] uint8

    return packed, e8m0_encoding


def dequantize_mxfp4(x_packed: torch.Tensor, scales: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    """Dequantize MXFP4 packed tensor back to float32."""
    # Unpack
    even = x_packed & 0x0F
    odd = (x_packed >> 4) & 0x0F

    # Interleave back to full size
    *batch_dims, K_half = even.shape
    K = K_half * 2
    full = torch.zeros(*batch_dims, K, dtype=torch.uint8, device=x_packed.device)
    full[..., 0::2] = even
    full[..., 1::2] = odd

    # Convert nibbles to float
    x_f32 = _e2m1_nibble_to_float(full)

    # Multiply by scale
    scale_f32 = fp8e8m0_to_float32(scales)
    scale_expanded = scale_f32.repeat_interleave(group_size, dim=-1)
    scale_expanded = scale_expanded[..., :K]
    return x_f32 * scale_expanded


# ============================================================================
# Triton Kernel
# ============================================================================


@triton.jit
def _fp32x2_to_fp4x2(x_lo, x_hi):
    """Convert two f32 values to packed E2M1x2 using Blackwell PTX."""
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b8 tmp;
            cvt.rn.satfinite.e2m1x2.f32 tmp, $1, $2;
            cvt.u32.u8 $0, tmp;
        }
        """,
        constraints="=r,f,f",
        args=[x_hi, x_lo],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    ).to(tl.uint8)


@triton.jit
def _decode_positive_e2m1(code):
    """Decode a positive E2M1 code in Triton."""
    return tl.where(
        code == 0,
        0.0,
        tl.where(
            code == 1,
            0.5,
            tl.where(
                code == 2,
                1.0,
                tl.where(
                    code == 3,
                    1.5,
                    tl.where(code == 4, 2.0, tl.where(code == 5, 3.0, tl.where(code == 6, 4.0, 6.0))),
                ),
            ),
        ),
    )


@triton.jit
def _decode_e2m1(code):
    """Decode signed E2M1 nibbles to FP32 values."""
    magnitude = _decode_positive_e2m1(code & 0x7)
    sign = tl.where((code & 0x8) != 0, -1.0, 1.0)
    return sign * magnitude


@triton.jit
def _encode_e2m1(x):
    """Encode FP32 values to signed E2M1 nibbles without inline PTX."""
    ax = tl.abs(x)
    code = tl.where(ax > 0.25, 1, 0)
    code = tl.where(ax >= 0.75, 2, code)
    code = tl.where(ax > 1.25, 3, code)
    code = tl.where(ax >= 1.75, 4, code)
    code = tl.where(ax > 2.5, 5, code)
    code = tl.where(ax >= 3.5, 6, code)
    code = tl.where(ax > 5.0, 7, code)
    sign = tl.where(x < 0.0, 8, 0)
    return (sign | code).to(tl.uint32)


@triton.jit
def _decode_mxfp4_matrix(
    data_ptr,
    scale_ptr,
    rows,
    cols,
    data_stride_row,
    data_stride_col,
    scale_stride_row,
    scale_stride_col,
):
    """Decode a row/column-major MXFP4 matrix without materializing it in HBM."""
    packed_ptrs = (
        data_ptr
        + rows[:, None] * data_stride_row
        + (cols[None, :] // 2) * data_stride_col
    )
    packed = tl.load(packed_ptrs).to(tl.uint32)
    odd = (cols[None, :] & 1) != 0
    codes = tl.where(odd, packed >> 4, packed & 0xF)

    scale_ptrs = (
        scale_ptr
        + rows[:, None] * scale_stride_row
        + (cols[None, :] // 32) * scale_stride_col
    )
    scale = tl.load(scale_ptrs).to(tl.float32)
    return _decode_e2m1(codes) * tl.math.exp2(scale - 127.0)


@triton.jit
def _decode_mxfp4_registers(packed, scale, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """Decode packed MXFP4 registers, preserving the pair interleave order."""
    packed_pairs = tl.broadcast_to(
        packed[:, :, None], [BLOCK_M, BLOCK_N // 2, 2]
    ).to(tl.uint32)
    pair_lane = tl.arange(0, 2)[None, None, :]
    codes = tl.where(pair_lane == 0, packed_pairs & 0xF, packed_pairs >> 4)
    codes = tl.reshape(codes, [BLOCK_M, BLOCK_N])
    scale = tl.math.exp2(scale.to(tl.float32) - 127.0)
    scale = tl.reshape(
        tl.broadcast_to(scale[:, :, None], [BLOCK_M, BLOCK_N // 32, 32]),
        [BLOCK_M, BLOCK_N],
    )
    return _decode_e2m1(codes) * scale


@triton.jit
def _mxfp4_attn_fwd_inner_scaled(
    acc0, acc1, acc2, acc3, l_i, m_i, q, q_scale,  #
    K_ptr, K_scale_ptr, V_ptr, V_scale_ptr,  #
    Delta_s_ptr,  #
    stride_kn, stride_kk,  #
    stride_ks_n, stride_ks_k,  #
    stride_vd, stride_vn,  #
    stride_vs_d, stride_vs_n,  #
    stride_delta_g, stride_delta_n,  #
    start_m, qk_scale,  #
    offs_m, offs_n,  #
    N_CTX: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    STAGE: tl.constexpr,  #
    HAS_DELTA_S: tl.constexpr,  #
    UOS_QMAX: tl.constexpr,  #
):
    """MXAttention inner loop using native MXFP4 scaled MMA with UOS/PNQ."""
    if STAGE == 1:
        lo, hi = 0, tl.minimum(start_m * BLOCK_M, N_CTX)
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, tl.minimum((start_m + 1) * BLOCK_M, N_CTX)
        lo = tl.multiple_of(lo, BLOCK_M)
    else:
        lo, hi = 0, N_CTX

    BLOCK_N_PACKED: tl.constexpr = BLOCK_N // 2
    HEAD_DIM_PACKED: tl.constexpr = HEAD_DIM // 2
    HEAD_DIM_SCALES: tl.constexpr = HEAD_DIM // 32
    BLOCK_N_SCALES: tl.constexpr = BLOCK_N // 32

    offs_k_packed = tl.arange(0, HEAD_DIM_PACKED)
    offs_k_scale = tl.arange(0, HEAD_DIM_SCALES)
    offs_n_packed = tl.arange(0, BLOCK_N_PACKED)
    offs_n_scale = tl.arange(0, BLOCK_N_SCALES)

    for start_n in tl.range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # Q @ K^T: native MXFP4 tensor-core MMA.  Q/K are packed E2M1 and
        # their E8M0 scales are supplied directly to dot_scaled.
        k = tl.load(
            K_ptr
            + (start_n + offs_n)[:, None] * stride_kn
            + offs_k_packed[None, :] * stride_kk
        )
        k_scale = tl.load(
            K_scale_ptr
            + (start_n + offs_n)[:, None] * stride_ks_n
            + offs_k_scale[None, :] * stride_ks_k
        )
        qk = tl.dot_scaled(
            q, q_scale, "e2m1", tl.trans(k), k_scale, "e2m1"
        )

        if HAS_DELTA_S:
            group_id = start_m
            ds_ptrs = (
                Delta_s_ptr
                + group_id * stride_delta_g
                + (start_n + offs_n) * stride_delta_n
            )
            qk = qk + tl.load(ds_ptrs)[None, :]

        valid_n = (start_n + offs_n[None, :]) < N_CTX
        if STAGE == 2:
            mask = (offs_m[:, None] >= (start_n + offs_n[None, :])) & valid_n
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
        else:
            qk = qk * qk_scale
            qk = tl.where(valid_n, qk, -1.0e6)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        p = tl.where(valid_n, p, 0.0)
        alpha = tl.math.exp2(m_i - m_ij)

        # UOS/PNQ probability quantization.
        p_reshaped = tl.reshape(p, [BLOCK_M, BLOCK_N // 32, 32])
        p_amax = tl.max(p_reshaped, 2)
        FP4_E2M1_MAX: tl.constexpr = UOS_QMAX
        p_e8m0, inv_scale_expanded = compute_uos_p_scale_inv(
            p_amax, FP4_E2M1_MAX, BLOCK_M, BLOCK_N
        )
        p_scaled = p * inv_scale_expanded
        p_pairs = tl.reshape(p_scaled, [BLOCK_M, BLOCK_N_PACKED, 2])
        p_even, p_odd = tl.split(p_pairs)
        p_packed = (
            _encode_e2m1(p_even) | (_encode_e2m1(p_odd) << 4)
        ).to(tl.uint8)

        # PNQ normalization uses the represented, quantized probability.
        packed_u32 = p_packed.to(tl.uint32)
        p_code_even = packed_u32 & 0x7
        p_code_odd = (packed_u32 >> 4) & 0x7
        p_scale = tl.math.exp2(p_e8m0.to(tl.float32) - 127.0)
        p_scale_full = tl.reshape(
            tl.broadcast_to(p_scale[:, :, None], [BLOCK_M, BLOCK_N // 32, 32]),
            [BLOCK_M, BLOCK_N],
        )
        p_scale_pairs = tl.reshape(
            p_scale_full, [BLOCK_M, BLOCK_N_PACKED, 2]
        )
        p_scale_even, p_scale_odd = tl.split(p_scale_pairs)
        l_ij = tl.sum(
            _decode_positive_e2m1(p_code_even) * p_scale_even
            + _decode_positive_e2m1(p_code_odd) * p_scale_odd,
            1,
        )
        l_i = l_i * alpha + l_ij

        acc0 = acc0 * alpha[:, None]
        acc1 = acc1 * alpha[:, None]
        acc2 = acc2 * alpha[:, None]
        acc3 = acc3 * alpha[:, None]

        offs_d0 = tl.arange(0, 32)
        offs_d1 = tl.arange(32, 64)
        offs_d2 = tl.arange(64, 96)
        offs_d3 = tl.arange(96, 128)
        v0 = tl.load(
            V_ptr
            + offs_d0[:, None] * stride_vd
            + (start_n // 2 + offs_n_packed)[None, :] * stride_vn
        )
        v_scale0 = tl.load(
            V_scale_ptr
            + offs_d0[:, None] * stride_vs_d
            + (start_n // 32 + offs_n_scale)[None, :] * stride_vs_n
        )
        v1 = tl.load(
            V_ptr
            + offs_d1[:, None] * stride_vd
            + (start_n // 2 + offs_n_packed)[None, :] * stride_vn
        )
        v_scale1 = tl.load(
            V_scale_ptr
            + offs_d1[:, None] * stride_vs_d
            + (start_n // 32 + offs_n_scale)[None, :] * stride_vs_n
        )
        v2 = tl.load(
            V_ptr
            + offs_d2[:, None] * stride_vd
            + (start_n // 2 + offs_n_packed)[None, :] * stride_vn
        )
        v_scale2 = tl.load(
            V_scale_ptr
            + offs_d2[:, None] * stride_vs_d
            + (start_n // 32 + offs_n_scale)[None, :] * stride_vs_n
        )
        v3 = tl.load(
            V_ptr
            + offs_d3[:, None] * stride_vd
            + (start_n // 2 + offs_n_packed)[None, :] * stride_vn
        )
        v_scale3 = tl.load(
            V_scale_ptr
            + offs_d3[:, None] * stride_vs_d
            + (start_n // 32 + offs_n_scale)[None, :] * stride_vs_n
        )

        acc0 = tl.dot_scaled(
            p_packed, p_e8m0, "e2m1", tl.trans(v0), v_scale0, "e2m1", acc0
        )
        acc1 = tl.dot_scaled(
            p_packed, p_e8m0, "e2m1", tl.trans(v1), v_scale1, "e2m1", acc1
        )
        acc2 = tl.dot_scaled(
            p_packed, p_e8m0, "e2m1", tl.trans(v2), v_scale2, "e2m1", acc2
        )
        acc3 = tl.dot_scaled(
            p_packed, p_e8m0, "e2m1", tl.trans(v3), v_scale3, "e2m1", acc3
        )
        m_i = m_ij

    return acc0, acc1, acc2, acc3, l_i, m_i


@triton.jit
def _mxfp4_attn_fwd_inner(
    acc, l_i, m_i, q, q_scale,  #
    K_ptr, K_scale_ptr, V_ptr, V_scale_ptr,  #
    Delta_s_ptr,  #
    stride_kn, stride_kk,  #
    stride_ks_n, stride_ks_k,  #
    stride_vd, stride_vn,  #
    stride_vs_d, stride_vs_n,  #
    stride_delta_g, stride_delta_n,  #
    start_m, qk_scale,  #
    offs_m, offs_n,  #
    N_CTX: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    STAGE: tl.constexpr,  #
    HAS_DELTA_S: tl.constexpr,  #
):
    # Determine loop bounds based on causal stage
    if STAGE == 1:
        lo, hi = 0, tl.minimum(start_m * BLOCK_M, N_CTX)
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, tl.minimum((start_m + 1) * BLOCK_M, N_CTX)
        lo = tl.multiple_of(lo, BLOCK_M)
    else:  # STAGE == 3: non-causal, full range
        lo, hi = 0, N_CTX

    BLOCK_N_PACKED: tl.constexpr = BLOCK_N // 2

    HEAD_DIM_PACKED: tl.constexpr = HEAD_DIM // 2
    offs_k_n = tl.arange(0, BLOCK_N)
    offs_head_packed = tl.arange(0, HEAD_DIM_PACKED)
    offs_scale_k = tl.arange(0, HEAD_DIM // 32)
    offs_scale_n = tl.arange(0, BLOCK_N // 32)
    offs_n_packed = tl.arange(0, BLOCK_N_PACKED)
    offs_head_full = tl.arange(0, HEAD_DIM)

    for start_n in tl.range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # Load packed MXFP4 K and its E8M0 scales.
        k_ptrs = K_ptr + (start_n + offs_k_n[:, None]) * stride_kn + offs_head_packed[None, :] * stride_kk
        k = tl.load(k_ptrs)
        ks_ptrs = (
            K_scale_ptr
            + (start_n + offs_k_n[:, None]) * stride_ks_n
            + offs_scale_k[None, :] * stride_ks_k
        )
        k_scale = tl.load(ks_ptrs)

        # Q @ K^T through native Blackwell MXFP4 scaled MMA.
        qk = tl.dot_scaled(q, q_scale, "e2m1", tl.trans(k), k_scale, "e2m1")

        # -- Add delta_s correction (before scaling) --
        if HAS_DELTA_S:
            group_id = start_m  # BLOCK_M=128 = GROUP_SIZE
            ds_ptrs = Delta_s_ptr + group_id * stride_delta_g + (start_n + offs_n) * stride_delta_n
            ds_tile = tl.load(ds_ptrs)  # [BLOCK_N]
            qk = qk + ds_tile[None, :]

        # -- Online softmax --
        valid_n = (start_n + offs_n[None, :]) < N_CTX
        if STAGE == 2:
            mask = (offs_m[:, None] >= (start_n + offs_n[None, :])) & valid_n
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            qk = qk * qk_scale
            qk = tl.where(valid_n, qk, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]

        p = tl.math.exp2(qk)
        p = tl.where(valid_n, p, 0.0)
        alpha = tl.math.exp2(m_i - m_ij)

        acc = acc * alpha[:, None]

        # -- Quantize P to MXFP4 (E2M1) on the fly --
        # Compute per-group-of-32 amax along BLOCK_N (reduction dim)
        p_reshaped = tl.reshape(p, [BLOCK_M, BLOCK_N // 32, 32])
        p_amax = tl.max(p_reshaped, 2)  # [BLOCK_M, BLOCK_N // 32]

        FP4_E2M1_MAX: tl.constexpr = 6.0
        p_e8m0, inv_scale_expanded = compute_p_scale_inv(
            p_amax, FP4_E2M1_MAX, BLOCK_M, BLOCK_N,
        )

        # Scale P values
        p_scaled = p * inv_scale_expanded

        # Pack to FP4 with the native Blackwell conversion instruction.
        # Reshape [BLOCK_M, BLOCK_N] → [BLOCK_M, BLOCK_N//2, 2], then split
        p_pairs = tl.reshape(p_scaled, [BLOCK_M, BLOCK_N_PACKED, 2])
        p_even, p_odd = tl.split(p_pairs)  # each [BLOCK_M, BLOCK_N//2]

        p_packed = _fp32x2_to_fp4x2(p_even, p_odd)

        l_ij = tl.sum(p, 1)
        l_i = l_i * alpha + l_ij

        # Load packed MXFP4 V and its E8M0 scales.
        v_ptrs = V_ptr + offs_head_full[:, None] * stride_vd + (start_n // 2 + offs_n_packed[None, :]) * stride_vn
        v = tl.load(v_ptrs)
        vs_ptrs = (
            V_scale_ptr
            + offs_head_full[:, None] * stride_vs_d
            + (start_n // 32 + offs_scale_n[None, :]) * stride_vs_n
        )
        v_scale = tl.load(vs_ptrs)

        # P @ V through native Blackwell MXFP4 scaled MMA.
        acc = tl.dot_scaled(p_packed, p_e8m0, "e2m1", tl.trans(v), v_scale, "e2m1", acc)

        m_i = m_ij

    return acc, l_i, m_i


@triton.jit
def _mxfp4_attn_fwd(
    Q, K, V, Out,  #
    Q_scale, K_scale, V_scale,  #
    Delta_s,  #
    sm_scale,  #
    stride_qz, stride_qh, stride_qm, stride_qk,  #
    stride_kz, stride_kh, stride_kn, stride_kk,  #
    stride_vz, stride_vh, stride_vd, stride_vn,  #
    stride_oz, stride_oh, stride_om, stride_ok,  #
    stride_qsz, stride_qsh, stride_qsm, stride_qsk,  #
    stride_ksz, stride_ksh, stride_ksn, stride_ksk,  #
    stride_vsz, stride_vsh, stride_vsd, stride_vsn,  #
    stride_dsz, stride_dsh, stride_dsg, stride_dsn,  #
    Z, H, N_CTX,  #
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    STAGE: tl.constexpr,  #
    HAS_DELTA_S: tl.constexpr,  #
    PNQ: tl.constexpr,  #
    UOS_QMAX: tl.constexpr,  #
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    # Base pointers for this batch/head
    q_offset = off_z * stride_qz + off_h * stride_qh
    k_offset = off_z * stride_kz + off_h * stride_kh
    v_offset = off_z * stride_vz + off_h * stride_vh
    o_offset = off_z * stride_oz + off_h * stride_oh
    qs_offset = off_z * stride_qsz + off_h * stride_qsh
    ks_offset = off_z * stride_ksz + off_h * stride_ksh
    vs_offset = off_z * stride_vsz + off_h * stride_vsh

    Q_ptr = Q + q_offset
    K_ptr = K + k_offset
    V_ptr = V + v_offset
    Q_scale_ptr = Q_scale + qs_offset
    K_scale_ptr = K_scale + ks_offset
    V_scale_ptr = V_scale + vs_offset

    # Delta_s pointer for this batch/head
    Delta_s_ptr = Delta_s + off_z * stride_dsz + off_h * stride_dsh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # MXAttention's UOS/PNQ path uses native MXFP4 scaled MMA throughout.
    if PNQ:
        offs_head_packed = tl.arange(0, HEAD_DIM // 2)
        offs_head_scale = tl.arange(0, HEAD_DIM // 32)
        q = tl.load(
            Q_ptr
            + offs_m[:, None] * stride_qm
            + offs_head_packed[None, :] * stride_qk
        )
        q_scale = tl.load(
            Q_scale_ptr
            + offs_m[:, None] * stride_qsm
            + offs_head_scale[None, :] * stride_qsk
        )

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        acc0 = tl.zeros([BLOCK_M, 32], dtype=tl.float32)
        acc1 = tl.zeros([BLOCK_M, 32], dtype=tl.float32)
        acc2 = tl.zeros([BLOCK_M, 32], dtype=tl.float32)
        acc3 = tl.zeros([BLOCK_M, 32], dtype=tl.float32)
        qk_scale = sm_scale * 1.44269504

        if STAGE & 1:
            acc0, acc1, acc2, acc3, l_i, m_i = _mxfp4_attn_fwd_inner_scaled(
                acc0, acc1, acc2, acc3, l_i, m_i, q, q_scale,
                K_ptr, K_scale_ptr, V_ptr, V_scale_ptr,
                Delta_s_ptr,
                stride_kn, stride_kk,
                stride_ksn, stride_ksk,
                stride_vd, stride_vn,
                stride_vsd, stride_vsn,
                stride_dsg, stride_dsn,
                start_m, qk_scale,
                offs_m, offs_n,
                N_CTX, BLOCK_M, HEAD_DIM, BLOCK_N,
                4 - STAGE,
                HAS_DELTA_S,
                UOS_QMAX,
            )
        if STAGE & 2:
            acc0, acc1, acc2, acc3, l_i, m_i = _mxfp4_attn_fwd_inner_scaled(
                acc0, acc1, acc2, acc3, l_i, m_i, q, q_scale,
                K_ptr, K_scale_ptr, V_ptr, V_scale_ptr,
                Delta_s_ptr,
                stride_kn, stride_kk,
                stride_ksn, stride_ksk,
                stride_vd, stride_vn,
                stride_vsd, stride_vsn,
                stride_dsg, stride_dsn,
                start_m, qk_scale,
                offs_m, offs_n,
                N_CTX, BLOCK_M, HEAD_DIM, BLOCK_N,
                2,
                HAS_DELTA_S,
                UOS_QMAX,
            )

        acc0 = acc0 / l_i[:, None]
        acc1 = acc1 / l_i[:, None]
        acc2 = acc2 / l_i[:, None]
        acc3 = acc3 / l_i[:, None]
        offs_d0 = tl.arange(0, 32)
        offs_d1 = tl.arange(32, 64)
        offs_d2 = tl.arange(64, 96)
        offs_d3 = tl.arange(96, 128)
        tl.store(Out + o_offset + offs_m[:, None] * stride_om + offs_d0[None, :] * stride_ok, acc0.to(Out.dtype.element_ty))
        tl.store(Out + o_offset + offs_m[:, None] * stride_om + offs_d1[None, :] * stride_ok, acc1.to(Out.dtype.element_ty))
        tl.store(Out + o_offset + offs_m[:, None] * stride_om + offs_d2[None, :] * stride_ok, acc2.to(Out.dtype.element_ty))
        tl.store(Out + o_offset + offs_m[:, None] * stride_om + offs_d3[None, :] * stride_ok, acc3.to(Out.dtype.element_ty))
    else:
        HEAD_DIM_PACKED: tl.constexpr = HEAD_DIM // 2
        q = tl.load(
            Q_ptr + offs_m[:, None] * stride_qm + tl.arange(0, HEAD_DIM_PACKED)[None, :] * stride_qk
        )
        q_scale = tl.load(
            Q_scale_ptr
            + offs_m[:, None] * stride_qsm
            + tl.arange(0, HEAD_DIM // 32)[None, :] * stride_qsk
        )
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        qk_scale = sm_scale * 1.44269504

        if STAGE & 1:
            acc, l_i, m_i = _mxfp4_attn_fwd_inner(
                acc, l_i, m_i, q, q_scale,
                K_ptr, K_scale_ptr, V_ptr, V_scale_ptr,
                Delta_s_ptr,
                stride_kn, stride_kk,
                stride_ksn, stride_ksk,
                stride_vd, stride_vn,
                stride_vsd, stride_vsn,
                stride_dsg, stride_dsn,
                start_m, qk_scale,
                offs_m, offs_n,
                N_CTX, BLOCK_M, HEAD_DIM, BLOCK_N,
                4 - STAGE,
                HAS_DELTA_S,
            )
        if STAGE & 2:
            acc, l_i, m_i = _mxfp4_attn_fwd_inner(
                acc, l_i, m_i, q, q_scale,
                K_ptr, K_scale_ptr, V_ptr, V_scale_ptr,
                Delta_s_ptr,
                stride_kn, stride_kk,
                stride_ksn, stride_ksk,
                stride_vd, stride_vn,
                stride_vsd, stride_vsn,
                stride_dsg, stride_dsn,
                start_m, qk_scale,
                offs_m, offs_n,
                N_CTX, BLOCK_M, HEAD_DIM, BLOCK_N,
                2,
                HAS_DELTA_S,
            )

        acc = acc / l_i[:, None]
        o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + tl.arange(0, HEAD_DIM)[None, :] * stride_ok
        tl.store(o_ptrs, acc.to(Out.dtype.element_ty))


# ============================================================================
# Wrapper Function
# ============================================================================


def mxfp4_flash_attention(
    q_packed: torch.Tensor,
    k_packed: torch.Tensor,
    v_packed: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    *,
    valid_q_len: int | None = None,
    valid_k_len: int | None = None,
    causal: bool = False,
    sm_scale: float = None,
    delta_s: torch.Tensor = None,
) -> torch.Tensor:
    """
    MXFP4 (E2M1) Flash Attention forward pass with E8M0 block scales.

    Args:
        q_packed: [B, H, M, D//2] uint8 — packed E2M1 queries (along HEAD_DIM)
        k_packed: [B, H, N, D//2] uint8 — packed E2M1 keys (along HEAD_DIM)
        v_packed: [B, H, D, N//2] uint8 — packed E2M1 values (along N, col-major)
        q_scale: [B, H, M, D//32] uint8 — E8M0 scales for Q
        k_scale: [B, H, N, D//32] uint8 — E8M0 scales for K
        v_scale: [B, H, D, N//32] uint8 — E8M0 scales for V
        valid_q_len: original query length before padding
        valid_k_len: original key/value length before padding
        causal: whether to apply causal mask
        sm_scale: softmax scale (default: 1/sqrt(HEAD_DIM))
        delta_s: [B, H, num_groups, N] float32 — QK smoothing correction (optional)

    Returns:
        output: [B, H, M, D] float32
    """
    B, H, M, D_packed = q_packed.shape
    D = D_packed * 2  # HEAD_DIM
    N = k_packed.shape[2]
    if valid_q_len is None:
        valid_q_len = M
    if valid_k_len is None:
        valid_k_len = N

    if sm_scale is None:
        sm_scale = 1.0 / (D ** 0.5)

    BLOCK_M = 128
    BLOCK_N = 64

    assert D in (64, 128), f"HEAD_DIM must be 64 or 128, got {D}"
    assert M % BLOCK_M == 0, f"M={M} must be divisible by BLOCK_M={BLOCK_M}"
    assert N % BLOCK_N == 0, f"N={N} must be divisible by BLOCK_N={BLOCK_N}"
    if valid_q_len > M or valid_k_len > N:
        raise ValueError("valid sequence lengths cannot exceed padded tensor lengths")

    output = torch.empty((B, H, M, D), dtype=torch.float32, device=q_packed.device)

    STAGE = 3 if causal else 1

    has_delta_s = delta_s is not None
    if not has_delta_s:
        # Dummy tensor — pointer won't be dereferenced when HAS_DELTA_S=False
        delta_s = torch.zeros(1, 1, 1, 1, dtype=torch.float32, device=q_packed.device)

    grid = (triton.cdiv(M, BLOCK_M), B * H)

    _mxfp4_attn_fwd[grid](
        q_packed, k_packed, v_packed, output,
        q_scale, k_scale, v_scale,
        delta_s,
        sm_scale,
        # Q strides [B, H, M, D//2]
        q_packed.stride(0), q_packed.stride(1), q_packed.stride(2), q_packed.stride(3),
        # K strides [B, H, N, D//2]
        k_packed.stride(0), k_packed.stride(1), k_packed.stride(2), k_packed.stride(3),
        # V strides [B, H, D, N//2]
        v_packed.stride(0), v_packed.stride(1), v_packed.stride(2), v_packed.stride(3),
        # O strides
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        # Q_scale strides [B, H, M, D//32]
        q_scale.stride(0), q_scale.stride(1), q_scale.stride(2), q_scale.stride(3),
        # K_scale strides [B, H, N, D//32]
        k_scale.stride(0), k_scale.stride(1), k_scale.stride(2), k_scale.stride(3),
        # V_scale strides [B, H, D, N//32]
        v_scale.stride(0), v_scale.stride(1), v_scale.stride(2), v_scale.stride(3),
        # Delta_s strides [B, H, num_groups, N]
        delta_s.stride(0), delta_s.stride(1), delta_s.stride(2), delta_s.stride(3),
        # Dimensions
        B, H, valid_k_len,
        # Compile-time constants
        HEAD_DIM=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        STAGE=STAGE,
        HAS_DELTA_S=has_delta_s,
        PNQ=False,
        UOS_QMAX=6.0,
        num_warps=4,
        num_stages=4,
    )

    return output[:, :, :valid_q_len, :]


def mxfp4_pnq_flash_attention(
    q_packed: torch.Tensor,
    k_packed: torch.Tensor,
    v_packed: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    *,
    valid_q_len: int,
    valid_k_len: int,
    causal: bool = False,
    sm_scale: float | None = None,
    qmax: float = 7.25,
) -> torch.Tensor:
    """MXFP4 attention with UOS probability quantization and PNQ.

    The Q/K/V tensors are expected to be prequantized with the same ``qmax``
    convention.  K/V may be padded to a multiple of the tile size; the
    original lengths are used to mask the padded probability positions.
    """
    B, H, M, D_packed = q_packed.shape
    D = D_packed * 2
    N = k_packed.shape[2]
    if sm_scale is None:
        sm_scale = 1.0 / (D**0.5)
    if D != 128:
        raise ValueError(f"MXAttention initially requires HEAD_DIM=128, got {D}")
    if valid_q_len > M or valid_k_len > N:
        raise ValueError("valid sequence lengths cannot exceed padded tensor lengths")

    output = torch.empty((B, H, M, D), dtype=torch.float32, device=q_packed.device)
    stage = 3 if causal else 1
    dummy_delta = torch.zeros(1, 1, 1, 1, dtype=torch.float32, device=q_packed.device)
    grid = (triton.cdiv(M, 128), B * H)

    _mxfp4_attn_fwd[grid](
        q_packed,
        k_packed,
        v_packed,
        output,
        q_scale,
        k_scale,
        v_scale,
        dummy_delta,
        sm_scale,
        q_packed.stride(0),
        q_packed.stride(1),
        q_packed.stride(2),
        q_packed.stride(3),
        k_packed.stride(0),
        k_packed.stride(1),
        k_packed.stride(2),
        k_packed.stride(3),
        v_packed.stride(0),
        v_packed.stride(1),
        v_packed.stride(2),
        v_packed.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        q_scale.stride(0),
        q_scale.stride(1),
        q_scale.stride(2),
        q_scale.stride(3),
        k_scale.stride(0),
        k_scale.stride(1),
        k_scale.stride(2),
        k_scale.stride(3),
        v_scale.stride(0),
        v_scale.stride(1),
        v_scale.stride(2),
        v_scale.stride(3),
        dummy_delta.stride(0),
        dummy_delta.stride(1),
        dummy_delta.stride(2),
        dummy_delta.stride(3),
        B,
        H,
        valid_k_len,
        HEAD_DIM=D,
        BLOCK_M=128,
        BLOCK_N=64,
        STAGE=stage,
        HAS_DELTA_S=False,
        PNQ=True,
        UOS_QMAX=qmax,
        num_warps=8,
        num_stages=1,
    )
    return output[:, :, :valid_q_len, :]
