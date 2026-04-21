# SPDX-License-Identifier: Apache-2.0
"""vLLM QDQ Plugin — out-of-tree activation quant-dequant simulation.

Registered as a vllm.general_plugins entry point. Activated by VLLM_QDQ=1.
"""

import logging

logger = logging.getLogger(__name__)


def register():
    """Called by vLLM plugin loader in every process (main + workers)."""
    import os

    if os.environ.get("VLLM_QDQ", "0") != "1":
        return

    from .patch import apply_patches

    apply_patches()
    logger.info("vllm-qdq-plugin: patches applied (VLLM_QDQ=1)")
