# SPDX-License-Identifier: Apache-2.0
"""SVG2 attention backend registration for vllm-omni."""

from vllm_omni.diffusion.attention.backends.abstract import AttentionBackend


class SVG2AttentionBackend(AttentionBackend):
    """Out-of-tree SVG2/SAP attention backend for Wan self-attention."""

    accept_output_buffer: bool = False

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128]

    @staticmethod
    def get_name() -> str:
        return "SVG2_ATTN"

    @staticmethod
    def get_impl_cls() -> type:
        from .impl import SVG2Impl

        return SVG2Impl
