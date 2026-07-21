# SPDX-License-Identifier: Apache-2.0
"""Patch diffusion attention dispatch so SLA can handle fp32 inputs."""

from __future__ import annotations

from vllm.logger import init_logger

logger = init_logger(__name__)

_INSTALLED = False
_ORIG = None


def install() -> None:
    global _INSTALLED, _ORIG
    if _INSTALLED:
        return

    from vllm_omni.diffusion.attention.layer import Attention
    from .impl import SLAImpl

    _ORIG = Attention._run_local_attention

    def _patched(self, query, key, value, attn_metadata):
        # Let SageSLA see fp32 inputs directly; the upstream module downcasts
        # internally before running its sparse/linear kernels.
        if isinstance(self.attention, SLAImpl):
            return self.attention.forward(query, key, value, attn_metadata)
        return _ORIG(self, query, key, value, attn_metadata)

    Attention._run_local_attention = _patched
    _INSTALLED = True
    logger.warning("vllm-qdq-plugin: installed SLA fp32 attention dispatch patch")
