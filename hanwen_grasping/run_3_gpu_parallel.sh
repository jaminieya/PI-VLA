#!/bin/bash
# Run 3 collect_100.sh jobs in parallel on 3 GPUs, all for one selected object_idx.
# Usage: ./run_3_gpu_parallel.sh <object_idx> [episodes_per_worker]
#   object_idx: 0=sugar_box, 1=mustard_bottle, 2=banana (required)
#   episodes_per_worker: episodes per GPU (default: 334); total = 3 x episodes_per_worker
#
# Examples:
#   ./run_3_gpu_parallel.sh 0           # object 0, 334 episodes per GPU (= 1002 total)
#   ./run_3_gpu_parallel.sh 1 100         # object 1, 100 episodes per GPU (= 300 total)
#
# Run from starter_code: ./run_3_gpu_parallel.sh 0

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OBJECT_IDX="${1:?Usage: $0 <object_idx> [episodes_per_worker]   object_idx: 0=sugar_box, 1=mustard_bottle, 2=banana}"
EPISODES_PER_WORKER="${2:-334}"
NUM_GPUS=3

if [[ ! "$OBJECT_IDX" =~ ^[012]$ ]]; then
  echo "Error: object_idx must be 0, 1, or 2 (0=sugar_box, 1=mustard_bottle, 2=banana)"
  exit 1
fi

echo "One object: object_idx=$OBJECT_IDX | $NUM_GPUS GPUs, $EPISODES_PER_WORKER episodes each (= $((NUM_GPUS * EPISODES_PER_WORKER)) total)"
echo "  GPU 0 -> log: collect_gpu0.log"
echo "  GPU 1 -> log: collect_gpu1.log"
echo "  GPU 2 -> log: collect_gpu2.log"
echo ""

for gpu in 0 1 2; do
  export CUDA_VISIBLE_DEVICES=$gpu
  nohup bash collect_100.sh "$OBJECT_IDX" "$EPISODES_PER_WORKER" > "collect_gpu${gpu}.log" 2>&1 &
  echo "Started GPU $gpu (object_idx=$OBJECT_IDX) PID=$!"
  unset CUDA_VISIBLE_DEVICES
done

echo ""
echo "All $NUM_GPUS workers started. Check logs: tail -f collect_gpu0.log"
echo "Wait for all: wait"
