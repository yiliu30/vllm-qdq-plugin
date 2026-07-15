# SPDX-License-Identifier: Apache-2.0
"""SpargeAttn attention implementation for vllm-omni diffusion models.

Wraps the prebuilt ``spas_sage_attn`` block-sparse CUDA kernels and the
``auto_round_kernel`` XPU sparse kernels with:
- NHD↔HND layout transpose (vllm-omni uses NHD, SpargeAttn uses HND)
- A guard chain that falls back to torch SDPA whenever SpargeAttn's hard
  requirements are not met (fp32, seq_len < 128, head_dim
  not in {64,128}, or an attention mask — which SpargeAttn silently ignores).
  Self-attention GQA/MQA is supported by the XPU sparse path as long as
  ``num_heads_q`` is divisible by ``num_heads_kv``.
"""

import importlib.util
import inspect
import os
import sys
import sysconfig
from pathlib import Path

import torch
import torch.nn.functional as F

from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionImpl,
    AttentionMetadata,
)
from vllm_omni.diffusion.forward_context import (
    get_forward_context,
    is_forward_context_available,
)

from .. import envs

logger = init_logger(__name__)

# SpargeAttn hard constraints (see SpargeAttn/spas_sage_attn/core.py asserts and
# inference_examples/modify_model/modify_wan.py guard).
_MIN_SEQ_LEN = 128
_SUPPORTED_HEAD_DIMS = (64, 128)

# Lazy-loaded XPU kernel reference.
_xpu_sparse_fn = None
_xpu_sparse_has_tensor_layout = False
_xpu_sparse_signature_params: set[str] = set()
_xpu_sparse_call_count = 0
_xpu_sparse_dump_count = 0
_sdpa_fallback_log_count = 0


def _parse_optional_positive_int(value, field_name: str) -> int | None:
    if value is None or value == "":
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be a positive integer when set, got {value!r}")
    return parsed


def _require_xpu_sparse_kwarg_support(
    field_name: str,
    param_name: str,
    value: int | None,
) -> None:
    if value is None:
        return
    if param_name not in _xpu_sparse_signature_params:
        raise RuntimeError(
            "The loaded auto_round_kernel sparse XPU binding does not support "
            f"{field_name} (expected kwarg {param_name!r}). "
            "Update the ARK checkout/build under SPARGE_ARK_PATH or unset the override."
        )


def _ensure_ark_xpu_binding(ark_module) -> None:
    if getattr(ark_module, "xpu_lib", None) is not None and hasattr(ark_module.xpu_lib, "sage_sparse"):
        return

    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not ext_suffix:
        raise RuntimeError("Unable to determine Python extension suffix for the current interpreter")

    search_roots = [
        Path(envs.SPARGE_ARK_PATH) / "xbuild" if envs.SPARGE_ARK_PATH else None,
        Path(envs.SPARGE_ARK_PATH) / "xbuild_diffuser" if envs.SPARGE_ARK_PATH else None,
        Path(envs.SPARGE_ARK_PATH) if envs.SPARGE_ARK_PATH else None,
    ]
    candidates = []
    for root in search_roots:
        if root is not None and root.exists():
            candidates.extend(sorted(root.glob(f"auto_round_kernel_xpu*{ext_suffix}")))
    if not candidates:
        raise RuntimeError(
            "Unable to locate a built XPU extension for SpargeAttn. "
            f"Expected suffix {ext_suffix!r} under {[str(p) for p in search_roots if p is not None]}."
        )

    ext_path = candidates[-1]
    spec = importlib.util.spec_from_file_location("auto_round_kernel_xpu", ext_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load extension spec from {ext_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["auto_round_kernel_xpu"] = module
    spec.loader.exec_module(module)
    required = ("sage_sparse", "sage_dynamic_quant_layout")
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise RuntimeError(f"Loaded extension is missing required XPU bindings {missing}: {ext_path}")
    ark_module.xpu_lib = module


def _get_xpu_sparse_fn():
    global _xpu_sparse_fn, _xpu_sparse_has_tensor_layout, _xpu_sparse_signature_params
    if _xpu_sparse_fn is not None:
        return _xpu_sparse_fn

    try:
        from auto_round_kernel import ARK
        ark = ARK()
    except ImportError:
        import auto_round_kernel as ark

    _ensure_ark_xpu_binding(ark)

    fn = getattr(ark, "sparge_sage2_attn_meansim_topk_xpu", None)
    if fn is None:
        raise ImportError(
            "auto_round_kernel does not expose sparge_sage2_attn_meansim_topk_xpu. "
            "Update auto-round-lib or set SPARGE_ARK_PATH to the correct ARK build."
        )
    signature = inspect.signature(fn)
    _xpu_sparse_fn = fn
    _xpu_sparse_signature_params = set(signature.parameters)
    _xpu_sparse_has_tensor_layout = "tensor_layout" in _xpu_sparse_signature_params
    return fn


def _maybe_dump_xpu_sparse_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    tensor_layout: str,
    topk: float,
    smooth_k: bool,
    simthreshd1: float,
    attention_sink: bool,
    k_quant_granularity: int,
    query_tile_tokens: int | None,
    sparse_q_block_tokens: int | None,
    sparse_k_block_tokens: int | None,
) -> None:
    global _xpu_sparse_call_count, _xpu_sparse_dump_count

    call_index = _xpu_sparse_call_count
    _xpu_sparse_call_count += 1

    if not envs.SPARGE_DUMP_INPUTS:
        return

    start_index = int(envs.SPARGE_DUMP_START_INDEX)
    max_dumps = int(envs.SPARGE_DUMP_MAX)
    if call_index < start_index or _xpu_sparse_dump_count >= max_dumps:
        return

    step_idx = None
    if is_forward_context_available():
        step_idx = get_forward_context().denoise_step_idx

    dump_dir = Path(envs.SPARGE_DUMP_DIR)
    os.makedirs(dump_dir, exist_ok=True)
    dump_path = dump_dir / f"sparge_xpu_call_{call_index:04d}.pt"

    torch.save(
        {
            "query": query.detach().to("cpu"),
            "key": key.detach().to("cpu"),
            "value": value.detach().to("cpu"),
            "call_index": call_index,
            "denoise_step_idx": step_idx,
            "query_shape": list(query.shape),
            "key_shape": list(key.shape),
            "value_shape": list(value.shape),
            "dtype": str(query.dtype),
            "tensor_layout": tensor_layout,
            "topk": topk,
            "smooth_k": smooth_k,
            "simthreshd1": simthreshd1,
            "attention_sink": attention_sink,
            "k_quant_granularity": k_quant_granularity,
            "query_tile_tokens": query_tile_tokens,
            "sparse_q_block_tokens": sparse_q_block_tokens,
            "sparse_k_block_tokens": sparse_k_block_tokens,
        },
        dump_path,
    )
    logger.info("Dumped Sparge XPU inputs to %s", dump_path)
    _xpu_sparse_dump_count += 1


def _maybe_log_sdpa_fallback(
    backend_name: str,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    has_mask: bool,
    is_causal: bool = False,
    force_dense: bool = False,
) -> None:
    global _sdpa_fallback_log_count

    if _sdpa_fallback_log_count >= 8:
        return

    reasons = []
    if query.dtype == torch.float32:
        reasons.append("dtype=float32")
    if query.shape[1] < _MIN_SEQ_LEN:
        reasons.append(f"seq_len<{_MIN_SEQ_LEN}")
    if query.shape[-1] not in _SUPPORTED_HEAD_DIMS:
        reasons.append(f"head_dim={query.shape[-1]}")
    if has_mask:
        reasons.append("attn_mask")
    if is_causal:
        reasons.append("causal")
    if force_dense:
        reasons.append("force_dense")

    if not reasons:
        reasons.append("unknown")

    logger.warning(
        "SpargeAttnImpl: %s falling back to SDPA (%s) q=%s k=%s v=%s",
        backend_name,
        ", ".join(reasons),
        tuple(query.shape),
        tuple(key.shape),
        tuple(value.shape),
    )
    _sdpa_fallback_log_count += 1


class SpargeAttnImpl(AttentionImpl):
    """Attention implementation using the SpargeAttn block-sparse kernel."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        qkv_layout: str | None = None,
        backend_kwargs: dict | None = None,
        **extra_impl_args,
    ) -> None:
        self.causal = causal
        self.softmax_scale = softmax_scale

        # Configuration from env vars (unified across CUDA and XPU).
        self._mode = envs.SPARGE_MODE
        self._topk = float(envs.SPARGE_TOPK)
        self._cdfthreshd = float(envs.SPARGE_CDFTHRESHD)
        self._smooth_k = envs.SPARGE_SMOOTH_K
        self._simthreshd1 = float(envs.SPARGE_SIMTHRESHD1)
        self._attention_sink = envs.SPARGE_ATTENTION_SINK
        self._k_quant_granularity = int(envs.SPARGE_K_QUANT_GRANULARITY)
        self._xpu_tensor_layout = envs.SPARGE_XPU_TENSOR_LAYOUT
        self._dense_steps = int(envs.SPARGE_DENSE_STEPS)
        self._force_dense = False
        self._query_tile_tokens = _parse_optional_positive_int(
            envs.SPARGE_QUERY_TILE_TOKENS,
            "SPARGE_QUERY_TILE_TOKENS",
        )
        self._sparse_q_block_tokens = _parse_optional_positive_int(
            envs.SPARGE_SPARSE_Q_BLOCK_TOKENS,
            "SPARGE_SPARSE_Q_BLOCK_TOKENS",
        )
        self._sparse_k_block_tokens = _parse_optional_positive_int(
            envs.SPARGE_SPARSE_K_BLOCK_TOKENS,
            "SPARGE_SPARSE_K_BLOCK_TOKENS",
        )

        # Override from backend_kwargs if provided.
        if backend_kwargs:
            self._mode = backend_kwargs.pop("sparge_mode", self._mode)
            if "sparge_topk" in backend_kwargs:
                self._topk = float(backend_kwargs.pop("sparge_topk"))
            if "sparge_cdfthreshd" in backend_kwargs:
                self._cdfthreshd = float(backend_kwargs.pop("sparge_cdfthreshd"))
            if "sparge_smooth_k" in backend_kwargs:
                self._smooth_k = backend_kwargs.pop("sparge_smooth_k")
            if "sparge_simthreshd1" in backend_kwargs:
                self._simthreshd1 = float(backend_kwargs.pop("sparge_simthreshd1"))
            if "sparge_attention_sink" in backend_kwargs:
                self._attention_sink = backend_kwargs.pop("sparge_attention_sink")
            if "sparge_k_quant_granularity" in backend_kwargs:
                self._k_quant_granularity = int(
                    backend_kwargs.pop("sparge_k_quant_granularity")
                )
            if "sparge_xpu_tensor_layout" in backend_kwargs:
                self._xpu_tensor_layout = str(
                    backend_kwargs.pop("sparge_xpu_tensor_layout")
                ).upper()
                if self._xpu_tensor_layout not in ("NHD", "HND"):
                    raise ValueError(
                        "sparge_xpu_tensor_layout must be either 'NHD' or 'HND', "
                        f"got {self._xpu_tensor_layout!r}"
                    )
            if "sparge_force_dense" in backend_kwargs:
                self._force_dense = bool(backend_kwargs.pop("sparge_force_dense"))
            if "sparge_query_tile_tokens" in backend_kwargs:
                self._query_tile_tokens = _parse_optional_positive_int(
                    backend_kwargs.pop("sparge_query_tile_tokens"),
                    "sparge_query_tile_tokens",
                )
            if "sparge_sparse_q_block_tokens" in backend_kwargs:
                self._sparse_q_block_tokens = _parse_optional_positive_int(
                    backend_kwargs.pop("sparge_sparse_q_block_tokens"),
                    "sparge_sparse_q_block_tokens",
                )
            if "sparge_sparse_k_block_tokens" in backend_kwargs:
                self._sparse_k_block_tokens = _parse_optional_positive_int(
                    backend_kwargs.pop("sparge_sparse_k_block_tokens"),
                    "sparge_sparse_k_block_tokens",
                )
            if backend_kwargs:
                logger.warning(
                    "SpargeAttnImpl ignoring backend_kwargs: %s",
                    list(backend_kwargs.keys()),
                )

    # ------------------------------------------------------------------
    # CUDA forward
    # ------------------------------------------------------------------

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
        *,
        config_override: str | None = None,
    ) -> torch.Tensor:
        has_mask = attn_metadata is not None and attn_metadata.attn_mask is not None
        if (
            self._force_dense
            or query.dtype == torch.float32
            or query.shape[1] < _MIN_SEQ_LEN
            or query.shape[-1] not in _SUPPORTED_HEAD_DIMS
            or has_mask
        ):
            _maybe_log_sdpa_fallback(
                "cuda",
                query,
                key,
                value,
                has_mask=has_mask,
                force_dense=self._force_dense,
            )
            return self._forward_sdpa(query, key, value)
        return self._forward_sparge_cuda(query, key, value)

    # ------------------------------------------------------------------
    # XPU forward
    # ------------------------------------------------------------------

    def forward_xpu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
    ) -> torch.Tensor:
        has_mask = attn_metadata is not None and attn_metadata.attn_mask is not None
        if (
            self._force_dense
            or self.causal
            or query.dtype == torch.float32
            or query.shape[1] < _MIN_SEQ_LEN
            or query.shape[-1] not in _SUPPORTED_HEAD_DIMS
            or has_mask
        ):
            _maybe_log_sdpa_fallback(
                "xpu",
                query,
                key,
                value,
                has_mask=has_mask,
                is_causal=self.causal,
                force_dense=self._force_dense,
            )
            return self._forward_sdpa(query, key, value)
        return self._forward_sparge_xpu(query, key, value)

    # ------------------------------------------------------------------
    # Shared SDPA fallback
    # ------------------------------------------------------------------

    def _forward_sdpa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """SDPA fallback when SpargeAttn's constraints are not met."""
        # Input is NHD [B, N, H, D]; SDPA expects [B, H, N, D].
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        enable_gqa = q.shape[1] != k.shape[1]
        if enable_gqa:
            if k.shape[1] != v.shape[1]:
                raise RuntimeError(
                    "SpargeAttn SDPA fallback received mismatched KV head counts: "
                    f"key={k.shape[1]}, value={v.shape[1]}"
                )
            if q.shape[1] % k.shape[1] != 0:
                raise RuntimeError(
                    "SpargeAttn SDPA fallback requires query heads to be divisible by KV heads "
                    f"for GQA, got query={q.shape[1]}, kv={k.shape[1]}"
                )
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=self.softmax_scale,
            is_causal=False,
            enable_gqa=enable_gqa,
        )
        return out.transpose(1, 2)  # back to NHD

    # ------------------------------------------------------------------
    # CUDA sparse kernel
    # ------------------------------------------------------------------

    @torch.compiler.disable()
    def _forward_sparge_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        from spas_sage_attn import (
            spas_sage2_attn_meansim_cuda,
            spas_sage2_attn_meansim_topk_cuda,
        )

        logger.warning_once(
            "SpargeAttnImpl: CUDA SpargeAttn kernel active (mode=%s, topk=%s, "
            "cdfthreshd=%s) — q shape %s",
            self._mode,
            self._topk,
            self._cdfthreshd,
            tuple(query.shape),
        )
        # SpargeAttn expects HND = [B, H, N, D], input is NHD = [B, N, H, D].
        q = query.transpose(1, 2).contiguous()
        k = key.transpose(1, 2).contiguous()
        v = value.transpose(1, 2).contiguous()

        if self._mode == "cdfthreshd":
            out = spas_sage2_attn_meansim_cuda(
                q,
                k,
                v,
                is_causal=self.causal,
                scale=self.softmax_scale,
                cdfthreshd=self._cdfthreshd,
                tensor_layout="HND",
            )
        else:
            out = spas_sage2_attn_meansim_topk_cuda(
                q,
                k,
                v,
                is_causal=self.causal,
                scale=self.softmax_scale,
                topk=self._topk,
                tensor_layout="HND",
            )
        return out.transpose(1, 2)  # back to NHD

    # ------------------------------------------------------------------
    # XPU sparse kernel
    # ------------------------------------------------------------------

    @torch.compiler.disable()
    def _forward_sparge_xpu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        fn = _get_xpu_sparse_fn()
        _require_xpu_sparse_kwarg_support(
            "SPARGE_QUERY_TILE_TOKENS",
            "query_tile_tokens",
            self._query_tile_tokens,
        )
        _require_xpu_sparse_kwarg_support(
            "SPARGE_SPARSE_Q_BLOCK_TOKENS",
            "sparse_q_block_tokens",
            self._sparse_q_block_tokens,
        )
        _require_xpu_sparse_kwarg_support(
            "SPARGE_SPARSE_K_BLOCK_TOKENS",
            "sparse_k_block_tokens",
            self._sparse_k_block_tokens,
        )

        # Use dense (topk=1.0) for early denoising steps if configured.
        topk = self._topk
        if self._dense_steps > 0 and is_forward_context_available():
            step_idx = get_forward_context().denoise_step_idx
            if step_idx is not None and step_idx < self._dense_steps:
                topk = 1.0

        logger.warning_once(
            "SpargeAttnImpl: XPU SpargeAttn kernel active (topk=%s, smooth_k=%s, "
            "simthreshd1=%s, attention_sink=%s, k_quant_granularity=%s, "
            "dense_steps=%s, query_tile_tokens=%s, sparse_q_block_tokens=%s, "
            "sparse_k_block_tokens=%s, xpu_tensor_layout=%s) — q shape %s",
            self._topk,
            self._smooth_k,
            self._simthreshd1,
            self._attention_sink,
            self._k_quant_granularity,
            self._dense_steps,
            self._query_tile_tokens,
            self._sparse_q_block_tokens,
            self._sparse_k_block_tokens,
            self._xpu_tensor_layout,
            tuple(query.shape),
        )

        orig_dtype = query.dtype
        # Keep bf16/fp16 inputs in their native dtype. FLUX exercises the ARK sparse
        # path with bf16 tensors successfully; forcing Cosmos bf16 activations down to
        # fp16 is an integration difference that can perturb output quality.
        target_dtype = orig_dtype if orig_dtype in (torch.float16, torch.bfloat16) else torch.float16
        q = query.to(target_dtype).contiguous() if query.dtype != target_dtype else query.contiguous()
        k = key.to(target_dtype).contiguous() if key.dtype != target_dtype else key.contiguous()
        v = value.to(target_dtype).contiguous() if value.dtype != target_dtype else value.contiguous()
        if self._xpu_tensor_layout == "HND":
            # vllm-omni hands diffusion attention tensors over as NHD [B, N, H, D].
            # The ARK binding also supports HND [B, H, N, D], so transpose here
            # when the user wants to exercise the HND sparse path.
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()

        _maybe_dump_xpu_sparse_inputs(
            q,
            k,
            v,
            tensor_layout=self._xpu_tensor_layout,
            topk=topk,
            smooth_k=self._smooth_k,
            simthreshd1=self._simthreshd1,
            attention_sink=self._attention_sink,
            k_quant_granularity=self._k_quant_granularity,
            query_tile_tokens=self._query_tile_tokens,
            sparse_q_block_tokens=self._sparse_q_block_tokens,
            sparse_k_block_tokens=self._sparse_k_block_tokens,
        )

        sparse_kwargs = {
            "is_causal": self.causal,
            "scale": self.softmax_scale,
            "topk": topk,
            "smooth_k": self._smooth_k,
            "simthreshd1": self._simthreshd1,
            "attention_sink": self._attention_sink,
            "k_quant_granularity": self._k_quant_granularity,
        }
        if self._query_tile_tokens is not None and "query_tile_tokens" in _xpu_sparse_signature_params:
            sparse_kwargs["query_tile_tokens"] = self._query_tile_tokens
        if self._sparse_q_block_tokens is not None and "sparse_q_block_tokens" in _xpu_sparse_signature_params:
            sparse_kwargs["sparse_q_block_tokens"] = self._sparse_q_block_tokens
        if self._sparse_k_block_tokens is not None and "sparse_k_block_tokens" in _xpu_sparse_signature_params:
            sparse_kwargs["sparse_k_block_tokens"] = self._sparse_k_block_tokens

        if _xpu_sparse_has_tensor_layout:
            out = fn(q, k, v, tensor_layout=self._xpu_tensor_layout, **sparse_kwargs)
        else:
            out = fn(q, k, v, **sparse_kwargs)

        if self._xpu_tensor_layout == "HND":
            out = out.transpose(1, 2).contiguous()
        if out.dtype != orig_dtype:
            out = out.to(orig_dtype)
        return out
