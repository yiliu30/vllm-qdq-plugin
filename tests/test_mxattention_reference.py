import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from vllm_qdq_plugin.mxattention.reference import (
    dequantize_mxfp4_uos,
    materialized_pnq_attention,
    normalized_fwht,
    quantize_mxfp4_uos,
    sdpa_reference,
)


def test_e2m1_midpoints_use_nearest_even():
    values = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32
    )
    sample = torch.cat((values, torch.zeros(32 - values.numel())))
    packed, _ = quantize_mxfp4_uos(sample)
    codes = torch.stack((packed & 0xF, packed >> 4), dim=-1).flatten()[: values.numel()]
    assert codes.tolist() == [0, 2, 2, 4, 4, 6, 6]


def test_zero_group_has_unit_scale_and_zero_codes():
    packed, scales = quantize_mxfp4_uos(torch.zeros(1, 32))
    assert packed.eq(0).all()
    assert scales.item() == 127


def test_uos_round_trip_shape_and_finite_values():
    x = torch.randn(2, 4, 64, 128, dtype=torch.bfloat16)
    packed, scales = quantize_mxfp4_uos(x)
    restored = dequantize_mxfp4_uos(packed, scales)
    assert restored.shape == x.shape
    assert torch.isfinite(restored).all()
    assert restored.abs().amax() <= 7.25 * 2.0


def test_normalized_fwht_preserves_norm_and_dot_product():
    q = torch.randn(2, 8, 128)
    k = torch.randn(2, 5, 128)
    qr = normalized_fwht(q)
    kr = normalized_fwht(k)
    assert torch.allclose(q.norm(dim=-1), qr.norm(dim=-1), rtol=1e-5, atol=1e-5)
    assert torch.allclose(q @ k.transpose(-1, -2), qr @ kr.transpose(-1, -2), rtol=1e-5, atol=1e-5)


def test_sdpa_reference_preserves_hnd_layout():
    q = torch.randn(1, 3, 5, 8)
    k = torch.randn(1, 3, 5, 8)
    v = torch.randn(1, 3, 5, 8)
    sm_scale = 1.0 / math.sqrt(q.shape[-1])

    expected = F.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=0.0,
        scale=sm_scale,
        is_causal=False,
    )

    assert torch.allclose(
        sdpa_reference(q, k, v, sm_scale=sm_scale, causal=False),
        expected,
    )


def _manual_pnq_reference(q, k, v, *, quantize_v_transposed):
    sm_scale = 1.0 / math.sqrt(q.shape[-1])
    q_p, q_s = quantize_mxfp4_uos(q)
    k_p, k_s = quantize_mxfp4_uos(k)
    if quantize_v_transposed:
        v_q = v.transpose(-1, -2).contiguous()
        v_p, v_s = quantize_mxfp4_uos(v_q)
        vd = dequantize_mxfp4_uos(v_p, v_s).transpose(-1, -2).contiguous()
    else:
        v_p, v_s = quantize_mxfp4_uos(v)
        vd = dequantize_mxfp4_uos(v_p, v_s)
    qd = dequantize_mxfp4_uos(q_p, q_s)
    kd = dequantize_mxfp4_uos(k_p, k_s)
    scores = torch.matmul(qd, kd.transpose(-1, -2)) * sm_scale
    p_tilde = torch.exp(scores - scores.max(dim=-1, keepdim=True).values)
    p_p, p_s = quantize_mxfp4_uos(p_tilde)
    p_hat = dequantize_mxfp4_uos(p_p, p_s)[..., : scores.shape[-1]]
    return torch.matmul(p_hat, vd) / p_hat.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(torch.float32).tiny
    )


def test_materialized_pnq_quantizes_v_along_pv_reduction_dimension():
    torch.manual_seed(0)
    q = torch.randn(1, 1, 32, 64)
    k = torch.randn(1, 1, 32, 64)
    row_scale = torch.linspace(0.25, 8.0, 32).reshape(1, 1, 32, 1)
    col_pattern = torch.sin(torch.linspace(0.0, 12.0, 64)).reshape(1, 1, 1, 64)
    v = row_scale * col_pattern

    actual = materialized_pnq_attention(q, k, v, use_hadamard=False)
    expected = _manual_pnq_reference(q, k, v, quantize_v_transposed=True)
    old_axis = _manual_pnq_reference(q, k, v, quantize_v_transposed=False)

    assert torch.allclose(actual, expected)
    assert not torch.allclose(actual, old_axis, atol=1e-6, rtol=1e-6)


def test_triton_e2m1_encoder_uses_nearest_even_midpoints():
    source = (
        Path(__file__).parents[1]
        / "src/vllm_qdq_plugin/sage3_attn/sage3/mxfp4_hw_kernel.py"
    ).read_text()
    assert "tl.where(ax > 0.25, 1, 0)" in source
    assert "tl.where(ax >= 0.75, 2, code)" in source
    assert "tl.where(ax > 1.25, 3, code)" in source
    assert "tl.where(ax >= 1.75, 4, code)" in source
    assert "tl.where(ax > 2.5, 5, code)" in source
    assert "tl.where(ax >= 3.5, 6, code)" in source
    assert "tl.where(ax > 5.0, 7, code)" in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_blackwell_native_scaled_mma_path():
    capability = torch.cuda.get_device_capability()
    if capability[0] < 10:
        pytest.skip("requires NVIDIA Blackwell")

    import triton

    if tuple(int(part) for part in triton.__version__.split(".")[:2]) < (3, 7):
        pytest.skip("requires Triton 3.7.1 or newer")

    from vllm_qdq_plugin.mxattention import mxattention_forward
    from vllm_qdq_plugin.mxattention.triton_workaround import install

    install()
    q = torch.randn((1, 1, 128, 128), device="cuda", dtype=torch.float16)
    k = torch.randn((1, 1, 128, 128), device="cuda", dtype=torch.float16)
    v = torch.randn((1, 1, 128, 128), device="cuda", dtype=torch.float16)
    out = mxattention_forward(
        q, k, v, mode="uos_pnq", allow_fallback=False
    )
    assert out.shape == q.shape
    assert torch.isfinite(out).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_blackwell_non_pnq_masks_padded_keys():
    capability = torch.cuda.get_device_capability()
    if capability[0] < 10:
        pytest.skip("requires NVIDIA Blackwell")

    import triton

    if tuple(int(part) for part in triton.__version__.split(".")[:2]) < (3, 7):
        pytest.skip("requires Triton 3.7.1 or newer")

    from vllm_qdq_plugin.mxattention.reference import quantize_mxfp4_uos
    from vllm_qdq_plugin.mxattention.triton_workaround import install
    from vllm_qdq_plugin.sage3_attn.sage3.mxfp4_hw_kernel import mxfp4_flash_attention

    install()
    q = torch.randn((1, 1, 128, 128), device="cuda", dtype=torch.float16)
    k = torch.randn((1, 1, 128, 128), device="cuda", dtype=torch.float16)
    v = torch.randn((1, 1, 128, 128), device="cuda", dtype=torch.float16)
    k[:, :, 65:, :] = 0
    v[:, :, 65:, :] = 0

    q_p, q_s = quantize_mxfp4_uos(q)
    k_p, k_s = quantize_mxfp4_uos(k)
    v_p, v_s = quantize_mxfp4_uos(v.transpose(-1, -2).contiguous())
    masked = mxfp4_flash_attention(
        q_p,
        k_p,
        v_p,
        q_s,
        k_s,
        v_s,
        valid_q_len=128,
        valid_k_len=65,
    )
    unmasked = mxfp4_flash_attention(
        q_p,
        k_p,
        v_p,
        q_s,
        k_s,
        v_s,
        valid_q_len=128,
        valid_k_len=128,
    )

    assert masked.shape == q.shape
    assert torch.isfinite(masked).all()
    assert not torch.allclose(masked, unmasked, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("length", [31, 32, 33, 63, 64, 65])
def test_quantizer_requires_group_aligned_last_dimension(length):
    if length % 32:
        with pytest.raises(ValueError):
            quantize_mxfp4_uos(torch.ones(length))
    else:
        packed, scales = quantize_mxfp4_uos(torch.ones(length))
        assert packed.shape[-1] == length // 2
        assert scales.shape[-1] == length // 32
