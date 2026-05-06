#!/bin/bash
# Run collect_multi_obj_data_for_student.py across multiple GPUs/workers (parallel).
# Uses --num_episodes per worker (one long process) for lower overhead.
# Each worker writes to its own output subtree to avoid timestamp/path collisions.
#
# Usage:
#   cd PI-VLA/hanwen_grasping/collect_data
#   bash run_multi_obj_collect_10000.sh
#   bash run_multi_obj_collect_10000.sh <base_output_dir> <total_runs> <num_gpus> <workers_per_gpu>
#
# Example:
#   bash run_multi_obj_collect_10000.sh output/multi_obj/20260421 10000
#   bash run_multi_obj_collect_10000.sh output/multi_obj/20260421 10000 3 2

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HANWEN_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$HANWEN_ROOT"

BASE_OUTPUT_DIR="${1:-output/multi_obj/20260421}"
TOTAL_RUNS="${2:-10000}"
NUM_GPUS="${3:-3}"
WORKERS_PER_GPU="${4:-1}"

if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]] || [ "$NUM_GPUS" -le 0 ]; then
  echo "Error: num_gpus must be a positive integer."
  exit 1
fi

if ! [[ "$WORKERS_PER_GPU" =~ ^[0-9]+$ ]] || [ "$WORKERS_PER_GPU" -le 0 ]; then
  echo "Error: workers_per_gpu must be a positive integer."
  exit 1
fi

TOTAL_WORKERS=$((NUM_GPUS * WORKERS_PER_GPU))

if ! [[ "$TOTAL_RUNS" =~ ^[0-9]+$ ]] || [ "$TOTAL_RUNS" -le 0 ]; then
  echo "Error: total_runs must be a positive integer."
  exit 1
fi

mkdir -p "$SCRIPT_DIR/logs"
mkdir -p "$BASE_OUTPUT_DIR"

BASE_PER_WORKER=$((TOTAL_RUNS / TOTAL_WORKERS))
REMAINDER=$((TOTAL_RUNS % TOTAL_WORKERS))

echo "Starting multi-object collection:"
echo "  total runs:      $TOTAL_RUNS"
echo "  GPUs:            $NUM_GPUS (0..$((NUM_GPUS - 1)))"
echo "  workers/gpu:     $WORKERS_PER_GPU"
echo "  total workers:   $TOTAL_WORKERS"
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

WORKER_IDX=0
for gpu in $(seq 0 $((NUM_GPUS - 1))); do
  for worker in $(seq 0 $((WORKERS_PER_GPU - 1))); do
    RUNS_FOR_WORKER=$BASE_PER_WORKER
    if [ "$WORKER_IDX" -lt "$REMAINDER" ]; then
      RUNS_FOR_WORKER=$((RUNS_FOR_WORKER + 1))
    fi

    WORKER_OUTPUT_DIR="${BASE_OUTPUT_DIR}/gpu${gpu}/w${worker}"
    LOG_FILE="${SCRIPT_DIR}/logs/multi_obj_collect_gpu${gpu}_w${worker}.log"

    echo "GPU $gpu worker $worker -> $RUNS_FOR_WORKER episodes, output: $WORKER_OUTPUT_DIR"

    (
      echo "[$(date '+%F %T')] [GPU $gpu][W $worker] start: episodes=${RUNS_FOR_WORKER}"
      if [ "$RUNS_FOR_WORKER" -eq 0 ]; then
        echo "[$(date '+%F %T')] [GPU $gpu][W $worker] done. failures=0 (no assigned episodes)"
        exit 0
      fi
      if CUDA_VISIBLE_DEVICES="$gpu" nohup python collect_data/collect_multi_obj_data_for_student.py \
        --headless \
        --num_episodes "$RUNS_FOR_WORKER" \
        --output_dir "$WORKER_OUTPUT_DIR" \
        --env_id "$gpu" </dev/null; then
        echo "[$(date '+%F %T')] [GPU $gpu][W $worker] done. failures=0"
      else
        echo "[$(date '+%F %T')] [GPU $gpu][W $worker] done. failures=1"
      fi
    ) >"$LOG_FILE" 2>&1 &

    PIDS+=("$!")
    WORKER_IDX=$((WORKER_IDX + 1))
  done
done

echo ""
echo "Workers started:"
for gpu in $(seq 0 $((NUM_GPUS - 1))); do
  for worker in $(seq 0 $((WORKERS_PER_GPU - 1))); do
    echo "  GPU $gpu worker $worker log: $SCRIPT_DIR/logs/multi_obj_collect_gpu${gpu}_w${worker}.log"
  done
done
echo "Use: tail -f $SCRIPT_DIR/logs/multi_obj_collect_gpu0_w0.log"
echo ""

TOTAL_FAIL=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || true
done

for gpu in $(seq 0 $((NUM_GPUS - 1))); do
  for worker in $(seq 0 $((WORKERS_PER_GPU - 1))); do
    LOG_FILE="${SCRIPT_DIR}/logs/multi_obj_collect_gpu${gpu}_w${worker}.log"
    if [ -f "$LOG_FILE" ]; then
      FAIL_LINE="$(rg "failures=" "$LOG_FILE" | tail -n 1)"
      FAIL_COUNT="$(echo "$FAIL_LINE" | sed -n 's/.*failures=\([0-9]\+\).*/\1/p')"
      FAIL_COUNT="${FAIL_COUNT:-0}"
      TOTAL_FAIL=$((TOTAL_FAIL + FAIL_COUNT))
    fi
  done
done

echo "All workers finished. Total failures reported: $TOTAL_FAIL"
exit 0
