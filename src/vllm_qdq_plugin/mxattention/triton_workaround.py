"""Runtime workaround for Triton's Blackwell scaled-MMA accumulator pass.

Triton 3.6/3.7 registers ``OptimizeAccumulatorInit`` for SM100.  With a
``tl.dot_scaled`` inside a loop, that pass can rewrite the accumulator
initializer into an invalid immutable TMEM allocation.  The source-level fix
is in Triton's Blackwell compiler pass; this module provides the equivalent
process-local workaround until that fix is available in the installed wheel.
"""

from __future__ import annotations

import contextvars
import functools
import threading
from typing import Any


_installed = False
_lock = threading.Lock()
_skip_for_capability: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "vllm_qdq_skip_tmem_accumulator_init", default=False
)


def install() -> bool:
    """Install the SM100-only compiler hook.

    Returns ``True`` when installed (or already installed), and ``False`` if
    the installed Triton does not expose the expected NVIDIA compiler API.
    """
    global _installed
    if _installed:
        return True

    with _lock:
        if _installed:
            return True

        try:
            from triton.backends.nvidia.compiler import CUDABackend, passes
        except (ImportError, AttributeError):
            return False

        original_pass = passes.ttgpuir.add_optimize_accumulator_init
        original_make_ttgir = CUDABackend.make_ttgir

        @functools.wraps(original_pass)
        def add_optimize_accumulator_init(pm: Any) -> None:
            if not _skip_for_capability.get():
                original_pass(pm)

        @functools.wraps(original_make_ttgir)
        def make_ttgir(mod: Any, metadata: Any, opt: Any, capability: int) -> Any:
            token = _skip_for_capability.set(capability >= 100)
            try:
                return original_make_ttgir(mod, metadata, opt, capability)
            finally:
                _skip_for_capability.reset(token)

        passes.ttgpuir.add_optimize_accumulator_init = add_optimize_accumulator_init
        CUDABackend.make_ttgir = staticmethod(make_ttgir)
        _installed = True
        return True

