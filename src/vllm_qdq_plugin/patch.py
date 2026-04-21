# SPDX-License-Identifier: Apache-2.0
"""Monkey-patch vllm._custom_ops to inject QDQ on activations.

Wraps marlin_gemm and moe_wna16_marlin_gemm at the ops level so that
every call site (dense, MoE gate+up, MoE down) gets QDQ automatically.
"""

import logging
import sys

import torch

logger = logging.getLogger(__name__)


def apply_patches():
    """Patch ops.marlin_gemm and ops.moe_wna16_marlin_gemm with QDQ wrappers."""
    import vllm._custom_ops as ops
    from vllm.scalar_type import ScalarType, scalar_types

    from .qdq.mxfp4 import mxfp4_qdq
    from .trace import trace_qdq

    _orig_marlin_gemm = ops.marlin_gemm
    _orig_moe_marlin_gemm = ops.moe_wna16_marlin_gemm

    def _patched_marlin_gemm(
        a: torch.Tensor,
        c: torch.Tensor | None,
        b_q_weight: torch.Tensor,
        b_bias: torch.Tensor | None,
        b_scales: torch.Tensor,
        a_scales: torch.Tensor | None,
        global_scale: torch.Tensor | None,
        b_zeros: torch.Tensor | None,
        g_idx: torch.Tensor | None,
        perm: torch.Tensor | None,
        workspace: torch.Tensor,
        b_q_type: ScalarType,
        size_m: int,
        size_n: int,
        size_k: int,
        is_k_full: bool = True,
        use_atomic_add: bool = False,
        use_fp32_reduce: bool = False,
        is_zp_float: bool = False,
    ) -> torch.Tensor:
        # MXFP4 QDQ — extend with elif for other dtypes
        if b_q_type == scalar_types.float4_e2m1f and a.dim() == 2:
            trace_qdq("marlin_gemm", a.shape, a.dtype)
            a = mxfp4_qdq(a, group_size=32)

        return _orig_marlin_gemm(
            a, c, b_q_weight, b_bias, b_scales, a_scales, global_scale,
            b_zeros, g_idx, perm, workspace, b_q_type, size_m, size_n,
            size_k, is_k_full, use_atomic_add, use_fp32_reduce, is_zp_float,
        )

    def _patched_moe_marlin_gemm(
        input: torch.Tensor,
        output: torch.Tensor | None,
        b_qweight: torch.Tensor,
        b_bias: torch.Tensor | None,
        b_scales: torch.Tensor,
        a_scales: torch.Tensor | None,
        global_scale: torch.Tensor | None,
        b_qzeros: torch.Tensor | None,
        g_idx: torch.Tensor | None,
        perm: torch.Tensor | None,
        workspace: torch.Tensor,
        sorted_token_ids: torch.Tensor,
        expert_ids: torch.Tensor,
        num_tokens_past_padded: torch.Tensor,
        topk_weights: torch.Tensor,
        moe_block_size: int,
        top_k: int,
        mul_topk_weights: bool,
        b_q_type: ScalarType,
        size_m: int,
        size_n: int,
        size_k: int,
        is_k_full: bool,
        use_atomic_add: bool,
        use_fp32_reduce: bool,
        is_zp_float: bool,
        thread_k: int = -1,
        thread_n: int = -1,
        blocks_per_sm: int = -1,
    ) -> torch.Tensor:
        # MXFP4 QDQ — extend with elif for other dtypes
        if b_q_type == scalar_types.float4_e2m1f and input.dim() == 2:
            trace_qdq("moe_wna16_marlin_gemm", input.shape, input.dtype)
            input = mxfp4_qdq(input, group_size=32)

        return _orig_moe_marlin_gemm(
            input, output, b_qweight, b_bias, b_scales, a_scales,
            global_scale, b_qzeros, g_idx, perm, workspace,
            sorted_token_ids, expert_ids, num_tokens_past_padded,
            topk_weights, moe_block_size, top_k, mul_topk_weights,
            b_q_type, size_m, size_n, size_k, is_k_full,
            use_atomic_add, use_fp32_reduce, is_zp_float,
            thread_k, thread_n, blocks_per_sm,
        )

    # Patch the module attribute
    ops.marlin_gemm = _patched_marlin_gemm
    ops.moe_wna16_marlin_gemm = _patched_moe_marlin_gemm

    # Also patch any modules that already imported these via
    # `from vllm import _custom_ops as ops` (they hold a ref to the module,
    # so updating the module attribute is sufficient). But for any direct
    # `from vllm._custom_ops import marlin_gemm` we need to fix those too.
    for mod in list(sys.modules.values()):
        try:
            if getattr(mod, "marlin_gemm", None) is _orig_marlin_gemm:
                mod.marlin_gemm = _patched_marlin_gemm
            if getattr(mod, "moe_wna16_marlin_gemm", None) is _orig_moe_marlin_gemm:
                mod.moe_wna16_marlin_gemm = _patched_moe_marlin_gemm
        except Exception:
            pass

    logger.info(
        "QDQ patches applied: marlin_gemm, moe_wna16_marlin_gemm (MXFP4)"
    )
