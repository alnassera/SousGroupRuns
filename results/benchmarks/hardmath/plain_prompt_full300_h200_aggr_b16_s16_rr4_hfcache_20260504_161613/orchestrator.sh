#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/qwen3_moe_math_copy/results/raw/plain_prompt_full300_h200_aggr_b16_s16_rr4_hfcache_20260504_161613
OUT=$ROOT/qwen3_moe_math_outputs_vllm_shards
cd /workspace/qwen3_moe_math_copy
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export HF_HOME=/workspace/hf_home
export HF_HUB_CACHE=/workspace/hf_cache
export HUGGINGFACE_HUB_CACHE=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME=/workspace/.cache
export TMPDIR=/workspace/tmp
export TMP=/workspace/tmp
export TEMP=/workspace/tmp
export TORCHINDUCTOR_CACHE_DIR=/workspace/torchinductor_cache
export TRITON_CACHE_DIR=/workspace/triton_cache
export VLLM_CACHE_ROOT=/workspace/vllm_cache
mkdir -p /workspace/tmp /workspace/hf_home /workspace/torchinductor_cache /workspace/triton_cache /workspace/vllm_cache $OUT
PY=/workspace/qwen3_eval_bundle/.venv_vllm/bin/python3
COMMON=(
  $PY /workspace/qwen3_moe_math_copy/run_math_benchmark_vllm.py
  --methods plain prompt
  --math-dataset-path /workspace/qwen3_moe_math_copy/HARDMath/evaluation/data/HARDMath_mini.json
  --max-examples 300
  --qwen-enable-thinking
  --max-new-tokens 16384
  --max-model-len 20480
  --no-enforce-eager
  --no-auto-select-heads
  --enable-chunked-prefill
  --enable-prefix-caching
  --routing-mode hybrid
  --bank-profile heuristic_completion
  --planner-bank-gain 1.5
  --math-hygiene-bank-gain 2.5
  --checker-bank-gain 2.0
  --completion-bank-gain 1.0
  --layer-ids 12 16 20 24 28 32 36
  --max-layers 9
  --max-heads-per-layer 4
  --layer-selection-metric top_head_score_sum
  --batch-size 16
  --max-num-seqs 16
  --gpu-memory-utilization 0.90
  --dtype bfloat16
  --output-dir $OUT
)
echo [orchestrator] root=$ROOT
echo [orchestrator] started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo [orchestrator] HF_HUB_CACHE=$HF_HUB_CACHE HF_HOME=$HF_HOME TMPDIR=$TMPDIR TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR
echo [orchestrator] methods=plain,prompt max_examples=300 batch_size=16 max_num_seqs=16 gpu_memory_utilization=0.90 max_model_len=20480 max_new_tokens=16384
pids=()
for gpu in 0 1 2 3; do
  ids_file=$ROOT/shards/rr_shard_${gpu}.ids
  mapfile -t ids < $ids_file
  run_name=hardmath_mini300_plain_prompt_h200_aggr_b16_s16_rr4_hfcache_shard${gpu}_t16384
  log=$ROOT/logs/plain_prompt_rr_shard_${gpu}_gpu${gpu}.log
  (
    export CUDA_VISIBLE_DEVICES=$gpu
    echo [worker] gpu=$gpu run_name=$run_name ids=$(tr '\n' ' ' < $ids_file)
    echo [worker] cache HF_HUB_CACHE=$HF_HUB_CACHE HF_HOME=$HF_HOME TMPDIR=$TMPDIR TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR
    ${COMMON[@]} --example-ids ${ids[@]} --run-name $run_name
  ) > $log 2>&1 &
  pid=$!
  echo $pid > $ROOT/logs/plain_prompt_rr_shard_${gpu}_gpu${gpu}.pid
  pids+=($pid)
  echo [orchestrator] launched gpu=$gpu pid=$pid log=$log
done
status=0
for pid in ${pids[@]}; do
  if ! wait $pid; then
    echo [orchestrator] worker_failed pid=$pid >&2
    status=1
  fi
done
echo [orchestrator] workers_done_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) status=$status
if [[ $status -eq 0 ]]; then
  $PY /workspace/qwen3_moe_math_copy/score_hardmath_outputs.py $OUT --output-path $ROOT/plain_prompt_300_score_summary.json > $ROOT/plain_prompt_300_score_summary.stdout.json
  echo [orchestrator] aggregate_score_done_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
fi
exit $status
