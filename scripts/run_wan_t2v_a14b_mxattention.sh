#!/usr/bin/env bash
set -euo pipefail

MODEL=/dev/shm/.hf.tmp/Wan-AI/Wan2.2-T2V-A14B-Diffusers
OUTPUT=/workspace/vllm-omni/wan_t2v_a14b_mxattention_modelcard_1280x720_81f_40steps.mp4
RESULTS_DIR=/software/hshen/yiliu7/docker_tmp

cd /workspace/vllm-omni

CUDA_VISIBLE_DEVICES=7 \
PYTHONPATH=/workspace/vllm-qdq-plugin/src \
VLLM_MXATTENTION=1 \
DIFFUSION_ATTENTION_BACKEND=SAGE_ATTN \
/workspace/vllm-omni/.venv/bin/python \
  /workspace/vllm-omni/examples/offline_inference/text_to_video/text_to_video.py \
  --model "$MODEL" \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --negative-prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
  --height 720 \
  --width 1280 \
  --num-frames 81 \
  --num-inference-steps 40 \
  --guidance-scale 4.0 \
  --guidance-scale-high 3.0 \
  --flow-shift 5.0 \
  --fps 16 \
  --seed 42 \
  --enable-cpu-offload \
  --output "$OUTPUT"

test -s "$OUTPUT"
cp "$OUTPUT" "$RESULTS_DIR/"
echo "Saved: $OUTPUT"
echo "Copied: $RESULTS_DIR/$(basename "$OUTPUT")"
