#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-shared12x6_d640_kv2}"

export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"
export TIE_EMBEDDINGS="${TIE_EMBEDDINGS:-1}"
export USE_COMPILE="${USE_COMPILE:-1}"
export SDP_BACKEND="${SDP_BACKEND:-flash}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-512}"
export EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-1024}"
export EVAL_GRAD_ACCUM_STEPS="${EVAL_GRAD_ACCUM_STEPS:-1}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-524288}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-262144}"
export ITERATIONS="${ITERATIONS:-200000}"
export WARMUP_STEPS="${WARMUP_STEPS:-20}"
export WARMDOWN_ITERS="${WARMDOWN_ITERS:-4000}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-0}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-50}"
export RUN_ID="${RUN_ID:-${CONFIG}_$(date +%Y%m%d_%H%M%S)}"

case "${CONFIG}" in
  baseline9x512)
    export NUM_LAYERS=9
    export NUM_SHARED_BLOCKS=0
    export SHARED_BLOCK_PATTERN=mirror
    export MODEL_DIM=512
    export NUM_HEADS=8
    export NUM_KV_HEADS=4
    export MLP_MULT=2
    ;;
  shared10x5_d640_kv5)
    export NUM_LAYERS=10
    export NUM_SHARED_BLOCKS=5
    export SHARED_BLOCK_PATTERN=mirror
    export MODEL_DIM=640
    export NUM_HEADS=10
    export NUM_KV_HEADS=5
    export MLP_MULT=2
    ;;
  shared12x6_d544_kv4_mlp3)
    export NUM_LAYERS=12
    export NUM_SHARED_BLOCKS=6
    export SHARED_BLOCK_PATTERN=mirror
    export MODEL_DIM=544
    export NUM_HEADS=8
    export NUM_KV_HEADS=4
    export MLP_MULT=3
    ;;
  shared12x6_d608_kv4)
    export NUM_LAYERS=12
    export NUM_SHARED_BLOCKS=6
    export SHARED_BLOCK_PATTERN=mirror
    export MODEL_DIM=608
    export NUM_HEADS=8
    export NUM_KV_HEADS=4
    export MLP_MULT=2
    ;;
  shared12x6_d640_kv2)
    export NUM_LAYERS=12
    export NUM_SHARED_BLOCKS=6
    export SHARED_BLOCK_PATTERN=mirror
    export MODEL_DIM=640
    export NUM_HEADS=10
    export NUM_KV_HEADS=2
    export MLP_MULT=2
    ;;
  *)
    echo "Unknown config: ${CONFIG}" >&2
    echo "Known configs: baseline9x512 shared10x5_d640_kv5 shared12x6_d544_kv4_mlp3 shared12x6_d608_kv4 shared12x6_d640_kv2" >&2
    exit 1
    ;;
esac

echo "Launching ${RUN_ID}"
echo "  config=${CONFIG}"
echo "  data=${DATA_PATH}"
echo "  tokenizer=${TOKENIZER_PATH}"
echo "  layers=${NUM_LAYERS} shared=${NUM_SHARED_BLOCKS} dim=${MODEL_DIM} heads=${NUM_HEADS} kv=${NUM_KV_HEADS} mlp=${MLP_MULT}"
echo "  train_seq=${TRAIN_SEQ_LEN} eval_seq=${EVAL_SEQ_LEN} train_batch_tokens=${TRAIN_BATCH_TOKENS} val_batch_size=${VAL_BATCH_SIZE}"
echo "  iterations=${ITERATIONS} wallclock=${MAX_WALLCLOCK_SECONDS} compile=${USE_COMPILE} sdp=${SDP_BACKEND}"

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-1}" train_gpt.py
