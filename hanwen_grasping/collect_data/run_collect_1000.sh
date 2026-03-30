#!/bin/bash
# Run new_setup_dataset_collect.py --headless 1000 times with banana only (object_idx=5).
# Each run saves to collected_data/grasp_6dof_demo_YYYYMMDD_HHMMSS.h5
#
# Usage: cd hanwen_grasping && ./run_collect_1000.sh
# Or:    bash run_collect_1000.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

N_RUNS=1000
OBJECT_IDX=5  # 5 = banana (011_banana)

echo "Running dataset collection $N_RUNS times (banana only, headless)"
echo "Output: ../collected_data/grasp_6dof_demo_*.h5"
echo ""

for i in $(seq 1 $N_RUNS); do
    echo "[$i/$N_RUNS] Starting run..."
    python new_setup_dataset_collect.py --headless --object_idx $OBJECT_IDX
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "[$i/$N_RUNS] WARNING: Run failed with exit code $exit_code"
    fi
done

echo ""
echo "Completed $N_RUNS runs."
