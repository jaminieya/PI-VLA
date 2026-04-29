#!/bin/bash
# Run collect_data.py 10,000 times across 3 GPUs (parallel).
# Each worker writes to its own output root to avoid timestamp collisions.
#
# Usage:
#   cd PI-VLA/hanwen_grasping/collect_data
#   bash run_collect_10000.sh
#   bash run_collect_10000.sh <base_output_dir> <total_runs>
#
# Example:
#   bash run_collect_10000.sh ../output/data_collection/20260421 10000

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HANWEN_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$HANWEN_ROOT"

BASE_OUTPUT_DIR="${1:-../output/data_collection/20260421}"
TOTAL_RUNS="${2:-1000}"
NUM_GPUS=3

if ! [[ "$TOTAL_RUNS" =~ ^[0-9]+$ ]] || [ "$TOTAL_RUNS" -le 0 ]; then
  echo "Error: total_runs must be a positive integer."
  exit 1
fi

mkdir -p "$BASE_OUTPUT_DIR"
mkdir -p "$SCRIPT_DIR/logs"

BASE_PER_GPU=$((TOTAL_RUNS / NUM_GPUS))
REMAINDER=$((TOTAL_RUNS % NUM_GPUS))

echo "Starting collection:"
echo "  total runs:      $TOTAL_RUNS"
echo "  GPUs:            $NUM_GPUS (0,1,2)"
echo "  base output dir: $BASE_OUTPUT_DIR"
echo "  logs dir:        $SCRIPT_DIR/logs"
echo ""

PIDS=()

cleanup() {
  echo ""
  echo "Stopping all workers..."
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup INT TERM

for gpu in 0 1 2; do
  RUNS_FOR_GPU=$BASE_PER_GPU
  if [ "$gpu" -lt "$REMAINDER" ]; then
    RUNS_FOR_GPU=$((RUNS_FOR_GPU + 1))
  fi

  WORKER_OUTPUT_DIR="${BASE_OUTPUT_DIR}/gpu${gpu}"
  LOG_FILE="${SCRIPT_DIR}/logs/collect_gpu${gpu}.log"

  echo "GPU $gpu -> $RUNS_FOR_GPU runs, output: $WORKER_OUTPUT_DIR"

  (
    FAILURES=0
    for i in $(seq 1 "$RUNS_FOR_GPU"); do
      echo "[$(date '+%F %T')] [GPU $gpu] run ${i}/${RUNS_FOR_GPU}"
      if ! CUDA_VISIBLE_DEVICES="$gpu" python collect_data/collect_data.py \
        --headless \
        --export_train_arm_scene \
        --train_arm_output_dir "$WORKER_OUTPUT_DIR"; then
        FAILURES=$((FAILURES + 1))
        echo "[$(date '+%F %T')] [GPU $gpu] WARNING: run ${i}/${RUNS_FOR_GPU} failed"
      fi
    done
    echo "[$(date '+%F %T')] [GPU $gpu] done. failures=$FAILURES"
  ) >"$LOG_FILE" 2>&1 &

  PIDS+=("$!")
done

echo ""
echo "Workers started:"
for idx in 0 1 2; do
  echo "  GPU $idx log: $SCRIPT_DIR/logs/collect_gpu${idx}.log"
done
echo "Use: tail -f $SCRIPT_DIR/logs/collect_gpu0.log"
echo ""

TOTAL_FAIL=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || true
done

for gpu in 0 1 2; do
  LOG_FILE="${SCRIPT_DIR}/logs/collect_gpu${gpu}.log"
  if [ -f "$LOG_FILE" ]; then
    FAIL_LINE="$(rg "failures=" "$LOG_FILE" | tail -n 1)"
    FAIL_COUNT="$(echo "$FAIL_LINE" | sed -n 's/.*failures=\([0-9]\+\).*/\1/p')"
    FAIL_COUNT="${FAIL_COUNT:-0}"
    TOTAL_FAIL=$((TOTAL_FAIL + FAIL_COUNT))
  fi
done

echo "All workers finished. Total failures reported: $TOTAL_FAIL"
exit 0
