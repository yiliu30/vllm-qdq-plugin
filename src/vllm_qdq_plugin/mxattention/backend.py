"""vLLM-Omni adapter for the plugin MXAttention implementation."""

from vllm_omni.diffusion.attention.backends.abstract import AttentionBackend


class MXAttentionBackend(AttentionBackend):
    accept_output_buffer = False
    supported_platforms = ("cuda",)

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [128]

    @staticmethod
    def get_name() -> str:
        # This class is registered as the plugin override for the existing
        # SAGE_ATTN slot, so the host-visible name must remain SAGE_ATTN.
        return "SAGE_ATTN"

    @staticmethod
    def get_impl_cls() -> type:
        from .impl import MXAttentionImpl

        return MXAttentionImpl
