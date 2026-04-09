#!/bin/bash
# Run collect_100.sh jobs in parallel on 3 GPUs.
# Usage: ./run_3_gpu_parallel.sh <object_idx> [episodes_per_worker]
#   object_idx: 0=sugar_box, 1=mustard_bottle, 2=banana (required)
#   episodes_per_worker: episodes per worker (default: 334); total = 3 x instances_per_gpu x episodes_per_worker
#
# Examples:
#   ./run_3_gpu_parallel.sh 0           # object 0, 334 episodes per GPU (= 1002 total)
#   ./run_3_gpu_parallel.sh 1 100         # object 1, 100 episodes per GPU (= 300 total)
#
# Run from starter_code: ./run_3_gpu_parallel.sh 2 150000

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OBJECT_IDX="${1:?Usage: $0 <object_idx> [episodes_per_worker]   object_idx: 0=sugar_box, 1=mustard_bottle, 2=banana}"
EPISODES_PER_WORKER="${2:-334}"
NUM_GPUS=3
INSTANCES_PER_GPU=2

if [[ ! "$OBJECT_IDX" =~ ^[012]$ ]]; then
  echo "Error: object_idx must be 0, 1, or 2 (0=sugar_box, 1=mustard_bottle, 2=banana)"
  exit 1
fi

echo "One object: object_idx=$OBJECT_IDX | $NUM_GPUS GPUs x $INSTANCES_PER_GPU instances, $EPISODES_PER_WORKER episodes each (= $((NUM_GPUS * INSTANCES_PER_GPU * EPISODES_PER_WORKER)) total)"
echo "Logs: collect_gpu<gpu>_worker<idx>.log"
echo ""
for gpu in 0 1 2; do
  for worker in 0 1; do
    # Inline the variable definition right before the command
    CUDA_VISIBLE_DEVICES=$gpu nohup bash collect_100.sh "$OBJECT_IDX" "$EPISODES_PER_WORKER" > "collect_gpu${gpu}_worker${worker}.log" 2>&1 &
    
    echo "Started GPU $gpu worker $worker (object_idx=$OBJECT_IDX) PID=$!"
  done
done

echo ""
echo "All $((NUM_GPUS * INSTANCES_PER_GPU)) workers started. Check logs: tail -f collect_gpu0_worker0.log"
echo "Wait for all: wait"
