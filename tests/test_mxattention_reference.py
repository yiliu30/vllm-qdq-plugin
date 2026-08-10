import pytest
import torch

from vllm_qdq_plugin.mxattention.reference import (
    dequantize_mxfp4_uos,
    normalized_fwht,
    quantize_mxfp4_uos,
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


@pytest.mark.parametrize("length", [31, 32, 33, 63, 64, 65])
def test_quantizer_requires_group_aligned_last_dimension(length):
    if length % 32:
        with pytest.raises(ValueError):
            quantize_mxfp4_uos(torch.ones(length))
    else:
        packed, scales = quantize_mxfp4_uos(torch.ones(length))
        assert packed.shape[-1] == length // 2
        assert scales.shape[-1] == length // 32
