# vllm-qdq-plugin

Out-of-tree [vLLM](https://github.com/vllm-project/vllm) plugin that simulates activation quant-dequant (QDQ) before quantized GEMM kernels. Useful for studying the accuracy impact of "real" quantized compute vs weight-only dequant approaches.

## How It Works

The plugin registers as a `vllm.general_plugins` entry point, which vLLM loads automatically in **all processes** (main + workers). It monkey-patches the low-level op wrappers in `vllm._custom_ops` to inject QDQ on input activations before the actual kernel call. This means:

- Zero vLLM source modifications
- Works with both `LLM()` Python API and `vllm serve`
- Covers all call sites automatically (dense linear, MoE gate+up, MoE down)

## Installation

```bash
pip install git+https://github.com/yiliu30/vllm-qdq-plugin.git

# Or for development:
git clone https://github.com/yiliu30/vllm-qdq-plugin.git
pip install -e vllm-qdq-plugin/
```

## Usage

```bash
# Enable QDQ
VLLM_QDQ=1 python my_script.py

# With vllm serve
VLLM_QDQ=1 vllm serve /path/to/model --tensor-parallel-size 2

# Enable trace logging (prints shape/dtype for each QDQ call)
VLLM_QDQ=1 VLLM_QDQ_TRACE=1 vllm serve /path/to/model

# Force MXFP4 QDQ on Marlin MoE when dtype-based detection is not enough
VLLM_QDQ=1 VLLM_MARLIN_MOE_QDQ_MODE=FORCE_MXFP4 vllm serve /path/to/model
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `VLLM_QDQ` | `0` | Set to `1` to enable QDQ |
| `VLLM_QDQ_TRACE` | `0` | Set to `1` to print trace lines (up to 200) |
| `VLLM_MARLIN_MOE_QDQ_MODE` | `0` | Set to `FORCE_MXFP4` to apply MXFP4 QDQ in `moe_wna16_marlin_gemm` when dtype-based routing is not sufficient. Matching is case-insensitive. |

## Support Status

| Dtype | Op | Status | Notes |
|---|---|---|---|
| **MXFP4** (E2M1 + E8M0 scales) | `marlin_gemm` | ✅ Supported | Dense quantized linear (MXFP4 via Marlin) |
| **MXFP4** (E2M1 + E8M0 scales) | `moe_wna16_marlin_gemm` | ✅ Supported | MoE quantized linear (MXFP4 via Marlin) |

### How QDQ Works

For MXFP4, the QDQ simulates:
1. **Quantize**: Scale activations per group of 32 using E8M0 (power-of-2) scales, then round to nearest FP4 E2M1 value `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`
2. **Dequantize**: Multiply back by the scale to restore the original dtype

This introduces the same quantization noise that a "real" MXFP4 GEMM would produce on the input side, while keeping the actual computation in bf16 via Marlin's weight-only dequant kernel.

## Adding New Dtypes

1. Create a new QDQ implementation in `src/vllm_qdq_plugin/qdq/` (e.g., `fp8.py`)
2. Add an `elif` branch in `patch.py` where the dtype check happens
3. The QDQ function signature: `(x: Tensor, **config) -> Tensor` — same shape and dtype in/out

## License

Apache-2.0
