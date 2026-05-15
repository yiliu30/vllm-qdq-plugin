# SPDX-License-Identifier: Apache-2.0
"""vLLM QDQ Plugin — out-of-tree activation quant-dequant simulation.

Registered as a vllm.general_plugins entry point. Activated by VLLM_QDQ=1.
Also provides sage3 Triton attention backend for vllm-omni (VLLM_SAGE3_TRITON=1).
"""

from vllm.logger import init_logger

from . import envs

logger = init_logger(__name__)


def register():
    """Called by vLLM plugin loader in every process (main + workers)."""

    if not envs.VLLM_QDQ:
        return

    from .patch import apply_patches

    apply_patches()
    logger.info("vllm-qdq-plugin: patches applied (VLLM_QDQ enabled)")


def register_omni():
    """Called by vllm-omni plugin loader in diffusion workers.

    Conditionally overrides SAGE_ATTN backend with sage3 Triton implementation.
    When VLLM_SAGE3_TRITON=0 (default), does nothing — original in-tree backend used.
    """
    if not envs.VLLM_SAGE3_TRITON:
        return

    try:
        from vllm_omni.diffusion.attention.backends.registry import (
            DiffusionAttentionBackendEnum,
            register_diffusion_backend,
        )

        register_diffusion_backend(
            DiffusionAttentionBackendEnum.SAGE_ATTN,
            "vllm_qdq_plugin.sage3_attn.backend.Sage3TritonBackend",
        )
        logger.warning(
            "vllm-qdq-plugin: registered sage3 Triton backend as SAGE_ATTN "
            "(VLLM_SAGE3_TRITON=1)"
        )
    except ImportError as e:
        logger.warning(
            "vllm-qdq-plugin: cannot register sage3 backend — "
            "vllm_omni not available (%s)",
            e,
        )
