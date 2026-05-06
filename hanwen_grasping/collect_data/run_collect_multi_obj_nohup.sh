#!/usr/bin/env bash
# Run collect_multi_obj_data_for_student.py with multiple nohup workers.
#
# Defaults:
# - NUM_GPUS=1, WORKERS_PER_GPU=1, EPISODES=1 per worker
# - OUT=output/multi_obj (relative to hanwen_grasping)
# - --flat_output enabled (no timestamp subdir; timestamp in filename)
#
# Example:
#   NUM_GPUS=3 WORKERS_PER_GPU=2 EPISODES=500 OUT=output/multi_obj/20260430 \
#     bash collect_data/run_collect_multi_obj_nohup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HANWEN_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$HANWEN_ROOT"

NUM_GPUS="${NUM_GPUS:-1}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-1}"
GPU_START="${GPU_START:-0}"
EPISODES="${EPISODES:-1}"
MAX_PLAN="${MAX_PLAN:-10}"
OUT="${OUT:-output/multi_obj}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]] || [ "$NUM_GPUS" -le 0 ]; then
  echo "Error: NUM_GPUS must be a positive integer." >&2
  exit 1
fi
if ! [[ "$WORKERS_PER_GPU" =~ ^[0-9]+$ ]] || [ "$WORKERS_PER_GPU" -le 0 ]; then
  echo "Error: WORKERS_PER_GPU must be a positive integer." >&2
  exit 1
fi
if ! [[ "$GPU_START" =~ ^[0-9]+$ ]] || [ "$GPU_START" -lt 0 ]; then
  echo "Error: GPU_START must be a non-negative integer." >&2
  exit 1
fi

mkdir -p "$SCRIPT_DIR/logs"
if [[ "$OUT" = /* ]]; then
  OUT_ABS="$OUT"
else
  OUT_ABS="$HANWEN_ROOT/$OUT"
fi
mkdir -p "$OUT_ABS"

export PYTHONUNBUFFERED=1

echo "Starting multi-worker nohup collection"
echo "  cwd:              $HANWEN_ROOT"
echo "  output dir:       $OUT_ABS"
echo "  num_gpus:         $NUM_GPUS"
echo "  workers_per_gpu:  $WORKERS_PER_GPU"
echo "  episodes/worker:  $EPISODES"
echo "  max_plan:         $MAX_PLAN"
echo ""

PIDS=()
for ((g=0; g<NUM_GPUS; g++)); do
  GPU=$((GPU_START + g))
  for ((w=0; w<WORKERS_PER_GPU; w++)); do
    TS="$(date '+%Y%m%d_%H%M%S')"
    LOG="$SCRIPT_DIR/logs/nohup_multi_obj_gpu${GPU}_w${w}_${TS}.log"
    PID_FILE="$SCRIPT_DIR/logs/nohup_multi_obj_gpu${GPU}_w${w}_${TS}.pid"

    # shellcheck disable=SC2206
    EXTRA_ARR=($EXTRA_ARGS)

    CMD=(python -u collect_data/collect_multi_obj_data_for_student.py
      --headless
      --num_episodes "$EPISODES"
      --max_plan_attempts "$MAX_PLAN"
      --env_id "$GPU"
      --output_dir "$OUT_ABS"
      --flat_output
    )

    echo "Launching GPU $GPU worker $w"
    echo "  log: $LOG"

    CUDA_VISIBLE_DEVICES="$GPU" nohup "${CMD[@]}" "${EXTRA_ARR[@]}" </dev/null >>"$LOG" 2>&1 &
    echo $! >"$PID_FILE"
    PIDS+=("$!")
    echo "  pid: $(cat "$PID_FILE")"
  done
done

echo ""
echo "Workers started: ${#PIDS[@]}"
echo "Check logs: ls -1 $SCRIPT_DIR/logs/nohup_multi_obj_gpu*_w*_*.log"
