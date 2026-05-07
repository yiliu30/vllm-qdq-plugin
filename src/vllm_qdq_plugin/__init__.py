# SPDX-License-Identifier: Apache-2.0
"""vLLM QDQ Plugin — out-of-tree activation quant-dequant simulation.

Registered as a vllm.general_plugins entry point. Activated by VLLM_QDQ=1.
"""

import logging

from . import envs

logger = logging.getLogger(__name__)


def register():
    """Called by vLLM plugin loader in every process (main + workers)."""

    if not envs.VLLM_QDQ:
        return

    from .patch import apply_patches

    apply_patches()
    logger.info("vllm-qdq-plugin: patches applied (VLLM_QDQ enabled)")
