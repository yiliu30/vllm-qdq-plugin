# MXAttention Handoff

## Scope

This branch adds an experimental MXAttention backend for NVIDIA Blackwell
GPUs. It implements UOS MXFP4 quantization and PNQ probability updates for
Wan/vLLM-Omni self-attention.

The optimized PNQ path uses native `tl.dot_scaled` for both QK and PV. PV is
split into four 32-column accumulators because Blackwell TMEM layout
legalization rejects one `[128, 128]` scaled-MMA accumulator.

## Runtime requirements

- NVIDIA Blackwell / SM100
- Triton 3.7.1 or newer
- Torch 2.11.0+cu130 was used for validation
- Head dimension 128
- Equal Q/K/V head and sequence counts for the optimized path

The plugin installs a process-local Python workaround for Triton's
`OptimizeAccumulatorInit` pass when `VLLM_MXATTENTION=1`. This bypasses the
Blackwell pass that creates an immutable uninitialized TMEM accumulator for
loop-carried `tl.dot_scaled`. A source-level Triton compiler fix remains the
preferred production solution.

## Enablement

```bash
VLLM_MXATTENTION=1 \
DIFFUSION_ATTENTION_BACKEND=SAGE_ATTN
```

Defaults:

```text
MXATTENTION_MODE=mxattention_full
MXATTENTION_QMAX=7.25
MXATTENTION_USE_HADAMARD=1
```

Unsupported masks, cross-attention, GQA/MQA, and unsupported head sizes use
the SDPA fallback.

## Wan model-card E2E validation

Validated on GPU3 with the local
`Wan2.2-TI2V-5B-Diffusers` checkpoint using:

- Resolution: 1280x704
- Frames: 121
- Inference steps: 50
- Guidance scale: 5.0
- FPS: 24
- Flow shift: 5.0
- Solver: UniPC

The run completed all 50 steps in approximately 260.5 seconds. The output
was validated as an MP4 with 121 frames at 1280x704 and 24 FPS:

```text
/software/hshen/yiliu7/docker_tmp/wan_ti2v_mxattention_modelcard_1280x704_121f_50steps.mp4
```

Example command:

```bash
CUDA_VISIBLE_DEVICES=3 \
PYTHONPATH=/workspace/vllm-qdq-plugin/src \
VLLM_MXATTENTION=1 \
DIFFUSION_ATTENTION_BACKEND=SAGE_ATTN \
/workspace/vllm-omni/.venv/bin/python \
  /workspace/vllm-omni/examples/offline_inference/image_to_video/image_to_video.py \
  --model /dev/shm/.hf.tmp/Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --image /dev/shm/.hf.tmp/Wan-AI/Wan2.2-TI2V-5B-Diffusers/examples/i2v_input.JPG \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --height 704 --width 1280 --num-frames 121 \
  --num-inference-steps 50 --guidance-scale 5.0 \
  --flow-shift 5.0 --fps 24 --seed 42 \
  --output wan_ti2v_mxattention_modelcard_1280x704_121f_50steps.mp4
```

## Wan T2V A14B model-card E2E validation

Validated on GPU3 with the local
`Wan2.2-T2V-A14B-Diffusers` checkpoint using the model-card settings:

- Resolution: 1280x720
- Frames: 81
- Inference steps: 40
- Guidance scale: 4.0
- High-noise guidance scale: 3.0
- FPS: 16
- Flow shift: 5.0
- Seed: 42

The run completed all 40 steps in 2684.1 seconds. The output was validated
as an MP4 with 81 frames at 1280x720 and 16 FPS. CPU offload was enabled to
fit the A14B checkpoint:

```text
/software/hshen/yiliu7/docker_tmp/wan_t2v_a14b_mxattention_modelcard_1280x720_81f_40steps.mp4
```

Example command:

```bash
CUDA_VISIBLE_DEVICES=3 \
PYTHONPATH=/workspace/vllm-qdq-plugin/src \
VLLM_MXATTENTION=1 \
DIFFUSION_ATTENTION_BACKEND=SAGE_ATTN \
/workspace/vllm-omni/.venv/bin/python \
  /workspace/vllm-omni/examples/offline_inference/text_to_video/text_to_video.py \
  --model /dev/shm/.hf.tmp/Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --height 720 --width 1280 --num-frames 81 \
  --num-inference-steps 40 --guidance-scale 4.0 \
  --guidance-scale-high 3.0 --flow-shift 5.0 --fps 16 --seed 42 \
  --enable-cpu-offload \
  --output wan_t2v_a14b_mxattention_modelcard_1280x720_81f_40steps.mp4
```

## Validation completed

- Native QK/PV `tl.dot_scaled` probes on GPU3
- Native MXAttention forward, causal and noncausal
- Padded and multi-block sequence cases
- Public MXAttention API with `allow_fallback=False`
- Wan E2E generation with explicit `SAGE_ATTN`
- MP4 frame-count and resolution validation
- Python compilation checks and `git diff --check`

## Follow-up

1. Replace the runtime Python workaround with the upstream Triton Blackwell
   accumulator-init fix when it is available.
2. Add a full GPU CI job for Triton 3.7.1+ and SM100.
3. Benchmark the native path against the platform attention backend at the
   model-card resolution and sequence length.
