# SPDX-License-Identifier: Apache-2.0
"""SageSLA attention backend registration for vllm-omni."""

from vllm_omni.diffusion.attention.backends.abstract import AttentionBackend


class SLABackend(AttentionBackend):
    """Out-of-tree SageSLA attention backend.

    Registered via the vllm_omni plugin entrypoint when VLLM_SLA_ATTN=1.
    Overrides the in-tree SAGE_ATTN backend slot.
    """

    accept_output_buffer: bool = False

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128]

    @staticmethod
    def get_name() -> str:
        return "SAGE_ATTN"

    @staticmethod
    def get_impl_cls() -> type:
        from .impl import SLAImpl

        return SLAImpl
