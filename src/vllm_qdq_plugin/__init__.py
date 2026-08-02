# SPDX-License-Identifier: Apache-2.0
"""vLLM QDQ Plugin — out-of-tree activation quant-dequant simulation.

Registered as a vllm.general_plugins entry point. Activated by VLLM_QDQ=1.
Also provides out-of-tree diffusion attention backends for vllm-omni.
"""

from vllm.logger import init_logger

from . import envs

logger = init_logger(__name__)


def _install_svg2_flashinfer_buffer_shim() -> None:
    try:
        import torch
        import flashinfer.sparse
    except Exception:
        return

    cls = flashinfer.sparse.VariableBlockSparseAttentionWrapper
    if getattr(cls, "__vllm_qdq_svg2_buffer_shim__", False):
        return

    orig_init = cls.__init__
    orig_reset = cls.reset_workspace_buffer

    def _ensure_buffers(self) -> None:
        if not hasattr(self, "_vector_sparse_indices_buffer"):
            self._vector_sparse_indices_buffer = torch.empty(
                (128 * 1024 * 1024,), dtype=torch.int32, device=self.device
            )
        if not hasattr(self, "_vector_sparse_indptr_buffer"):
            self._vector_sparse_indptr_buffer = torch.empty(
                (32768,), dtype=torch.int32, device=self.device
            )

    def patched_init(self, float_workspace_buffer, backend="auto"):
        orig_init(self, float_workspace_buffer, backend=backend)
        _ensure_buffers(self)

    def patched_reset(
        self,
        float_workspace_buffer,
        int_workspace_buffer,
        vector_sparse_indices_buffer=None,
        vector_sparse_indptr_buffer=None,
    ):
        orig_reset(self, float_workspace_buffer, int_workspace_buffer)
        _ensure_buffers(self)
        if vector_sparse_indices_buffer is not None:
            self._vector_sparse_indices_buffer = vector_sparse_indices_buffer
        if vector_sparse_indptr_buffer is not None:
            self._vector_sparse_indptr_buffer = vector_sparse_indptr_buffer

    cls.__init__ = patched_init
    cls.reset_workspace_buffer = patched_reset
    cls.__vllm_qdq_svg2_buffer_shim__ = True
    logger.warning_once(
        "vllm-qdq-plugin: installed SVG2 FlashInfer buffer shim for VariableBlockSparseAttentionWrapper"
    )


def register():
    """Called by vLLM plugin loader in every process (main + workers)."""

    if not envs.VLLM_QDQ:
        return

    from .patch import apply_patches

    apply_patches()
    logger.info("vllm-qdq-plugin: patches applied (VLLM_QDQ enabled)")


def register_omni_sage3_triton():
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


def register_omni_sage3_cute():
    try:
        from vllm_omni.diffusion.attention.backends.registry import (
            DiffusionAttentionBackendEnum,
            register_diffusion_backend,
        )

        register_diffusion_backend(
            DiffusionAttentionBackendEnum.SAGE_ATTN,
            "vllm_qdq_plugin.sage3_attn.backend.Sage3CuteBackend",
        )
        logger.warning(
            "vllm-qdq-plugin: registered sage3 cute backend as SAGE_ATTN "
            "(VLLM_SAGE3_CUTE=1)"
        )
    except ImportError as e:
        logger.warning(
            "vllm-qdq-plugin: cannot register sage3 cute backend — "
            "vllm_omni not available (%s)",
            e,
        )


def register_omni_sparge_attn():
    import importlib.util
    import sys

    # Put the SpargeAttn repo on sys.path if its package is not already importable.
    # register_omni runs at worker startup, before any Attention module is built and
    # thus before backend.get_impl_cls() triggers the spas_sage_attn import — so the
    # path is in place by the time the impl is loaded.
    if importlib.util.find_spec("spas_sage_attn") is None and envs.SPARGE_ATTN_REPO:
        sys.path.insert(0, envs.SPARGE_ATTN_REPO)
        logger.warning(
            "vllm-qdq-plugin: added SPARGE_ATTN_REPO to sys.path (%s)",
            envs.SPARGE_ATTN_REPO,
        )

    try:
        from vllm_omni.diffusion.attention.backends.registry import (
            DiffusionAttentionBackendEnum,
            register_diffusion_backend,
        )

        register_diffusion_backend(
            DiffusionAttentionBackendEnum.SAGE_ATTN,
            "vllm_qdq_plugin.sparge_attn.backend.SpargeAttnBackend",
        )
        logger.warning(
            "vllm-qdq-plugin: registered SpargeAttn backend as SAGE_ATTN "
            "(VLLM_SPARGE_ATTN=1)"
        )
    except ImportError as e:
        logger.warning(
            "vllm-qdq-plugin: cannot register SpargeAttn backend — "
            "vllm_omni not available (%s)",
            e,
        )


def register_omni_sla_attn():
    import importlib.util
    import sys

    if importlib.util.find_spec("SageSLA") is None and envs.SLA_REPO:
        sys.path.insert(0, envs.SLA_REPO)
        logger.warning(
            "vllm-qdq-plugin: added SLA_REPO to sys.path (%s)",
            envs.SLA_REPO,
        )
    if importlib.util.find_spec("spas_sage_attn") is None and envs.SPARGE_ATTN_REPO:
        sys.path.insert(0, envs.SPARGE_ATTN_REPO)
        logger.warning(
            "vllm-qdq-plugin: added SPARGE_ATTN_REPO to sys.path (%s)",
            envs.SPARGE_ATTN_REPO,
        )

    try:
        from vllm_omni.diffusion.attention.backends.registry import (
            DiffusionAttentionBackendEnum,
            register_diffusion_backend,
        )

        register_diffusion_backend(
            DiffusionAttentionBackendEnum.SAGE_ATTN,
            "vllm_qdq_plugin.sla_attn.backend.SLABackend",
        )
        from .sla_attn import fp32_patch

        fp32_patch.install()
        logger.warning(
            "vllm-qdq-plugin: registered SLA backend as SAGE_ATTN "
            "(VLLM_SLA_ATTN=1)"
        )
    except ImportError as e:
        logger.warning(
            "vllm-qdq-plugin: cannot register SLA backend — "
            "vllm_omni not available (%s)",
            e,
        )


def register_omni_svg2_attn():
    import importlib.util
    import os
    import sys

    svg2_repo = envs.SVG2_REPO
    if importlib.util.find_spec("svg") is None and svg2_repo and os.path.isdir(svg2_repo):
        sys.path.insert(0, svg2_repo)
        logger.warning(
            "vllm-qdq-plugin: added SVG2_REPO to sys.path (%s)",
            svg2_repo,
        )

    _install_svg2_flashinfer_buffer_shim()

    try:
        from vllm_omni.diffusion.attention.backends.registry import (
            DiffusionAttentionBackendEnum,
            register_diffusion_backend,
        )

        register_diffusion_backend(
            DiffusionAttentionBackendEnum.SVG2_ATTN,
            "vllm_qdq_plugin.svg2_attn.backend.SVG2AttentionBackend",
        )
        logger.warning_once(
            "vllm-qdq-plugin: registered SVG2 backend as SVG2_ATTN"
        )
    except (ImportError, AttributeError) as e:
        logger.warning(
            "vllm-qdq-plugin: cannot register SVG2 backend — "
            "vllm_omni SVG2 slot unavailable (%s)",
            e,
        )


def _maybe_install_route(route_file: str):
    """Install per-(layer, step) attention routing if a route file is set.

    A bad route file logs and falls back to un-routed behavior rather than
    killing worker startup.
    """
    if not route_file:
        return
    try:
        from .sage3_attn.routing import patch_attention

        patch_attention.install(route_file)
    except Exception as e:
        logger.warning(
            "vllm-qdq-plugin: failed to install sage3 attention routing "
            "from %s (%s) — continuing without routing",
            route_file,
            e,
        )


def register_omni():
    """Called by vllm-omni plugin loader in diffusion workers.

    Conditionally overrides SAGE_ATTN backend with sage3 Triton implementation.
    When VLLM_SAGE3_TRITON=0 (default), does nothing — original in-tree backend used.
    """
    register_omni_svg2_attn()

    # These backends all override the SAGE_ATTN slot, so only one may be active.
    sage3_requested = envs.VLLM_SAGE3_TRITON or envs.VLLM_SAGE3_CUTE
    overrides_requested = sum(
        bool(flag)
        for flag in (envs.VLLM_SLA_ATTN, envs.VLLM_SPARGE_ATTN, sage3_requested)
    )
    if overrides_requested > 1:
        raise RuntimeError(
            "vllm-qdq-plugin: VLLM_SLA_ATTN, VLLM_SPARGE_ATTN, and "
            "VLLM_SAGE3_{TRITON,CUTE} are mutually exclusive"
        )

    if envs.VLLM_SLA_ATTN:
        register_omni_sla_attn()
        logger.warning_once("vllm-qdq-plugin: registered SLA backend for vllm-omni")
    elif envs.VLLM_SPARGE_ATTN:
        register_omni_sparge_attn()
        logger.warning_once(
            "vllm-qdq-plugin: registered SpargeAttn backend for vllm-omni"
        )
    elif envs.VLLM_SAGE3_TRITON:
        register_omni_sage3_triton()
        logger.warning_once(
            "vllm-qdq-plugin: registered sage3 Triton backend for vllm-omni"
        )
        _maybe_install_route(envs.SAGE3_ROUTE_FILE)
    elif envs.VLLM_SAGE3_CUTE:
        register_omni_sage3_cute()
        logger.warning_once(
            "vllm-qdq-plugin: registered sage3 cute backend for vllm-omni"
        )
    else:
        logger.warning_once(
            "vllm-qdq-plugin: no custom attention backend registered for "
            "vllm-omni SAGE_ATTN overrides — set VLLM_SLA_ATTN=1, "
            "VLLM_SPARGE_ATTN=1, VLLM_SAGE3_TRITON=1, or "
            "VLLM_SAGE3_CUTE=1 to enable"
        )
