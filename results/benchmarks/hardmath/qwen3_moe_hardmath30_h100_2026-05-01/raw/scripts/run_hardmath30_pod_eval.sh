#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
mkdir -p logs

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASET_PATH="${DATASET_PATH:-${ROOT_DIR}/data/random_hardmath_mini_30_seed11.json}"
MAX_EXAMPLES="${MAX_EXAMPLES:-30}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16384}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-20480}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-qwen3_moe_math_outputs_vllm}"

COMMON_ARGS=(
  --math-dataset-path "${DATASET_PATH}"
  --max-examples "${MAX_EXAMPLES}"
  --qwen-enable-thinking
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --max-model-len "${MAX_MODEL_LEN}"
  --no-enforce-eager
  --routing-mode hybrid
  --bank-profile heuristic_completion
  --planner-bank-gain 2.0
  --math-hygiene-bank-gain 1.5
  --checker-bank-gain 2.0
  --completion-bank-gain 2.0
  --layer-ids 12 16 20 24 28 32 36
  --auto-select-heads
  --max-heads-per-layer 4
  --max-layers 5
  --layer-selection-metric top_head_score_sum
  --batch-size "${BATCH_SIZE}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --gpu-memory-utilization 0.90
  --dtype bfloat16
  --output-dir "${OUTPUT_DIR}"
)

echo "[pod_eval] plain baseline"
"${PYTHON_BIN}" run_math_benchmark_vllm.py \
  --methods plain \
  "${COMMON_ARGS[@]}" \
  --rho 0.0 \
  --positive-gain 0.0 \
  --run-name "hardmath30_seed11_qwen3_cot_plain_t${MAX_NEW_TOKENS}_h100"

RHOS=(3.0 4.0 5.0 6.0)
POSITIVE_GAINS=(2.0 3.0 3.0 4.0)

for index in "${!RHOS[@]}"; do
  RHO="${RHOS[$index]}"
  POSITIVE_GAIN="${POSITIVE_GAINS[$index]}"
  echo "[pod_eval] canonical_memory rho=${RHO} positive_gain=${POSITIVE_GAIN}"
  "${PYTHON_BIN}" run_math_benchmark_vllm.py \
    --methods canonical_memory \
    "${COMMON_ARGS[@]}" \
    --rho "${RHO}" \
    --positive-gain "${POSITIVE_GAIN}" \
    --selector-max-examples 16 \
    --selector-max-new-tokens 96 \
    --selector-diagnostic-max-decode-steps 96 \
    --selector-debug-top-heads 20 \
    --run-name "hardmath30_seed11_qwen3_cot_cm_hc_rho${RHO}_pg${POSITIVE_GAIN}_t${MAX_NEW_TOKENS}_h100"
done

echo "[pod_eval] complete"
