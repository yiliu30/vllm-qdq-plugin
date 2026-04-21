# SPDX-License-Identifier: Apache-2.0
"""Optional trace logging for QDQ calls. Enable with VLLM_QDQ_TRACE=1."""

import os

TRACE_ENABLED = os.environ.get("VLLM_QDQ_TRACE", "0") == "1"

_call_count = 0


def trace_qdq(op_name: str, shape, dtype):
    """Print a trace line if tracing is enabled."""
    if not TRACE_ENABLED:
        return
    global _call_count
    _call_count += 1
    if _call_count <= 200:
        print(f"[QDQ] op={op_name} shape={shape} dtype={dtype}")
