#!/bin/bash
# Collect grasp trajectories for one object type
# Usage: ./collect_100.sh [object_idx] [num_episodes]
#   object_idx: 0=sugar_box, 1=mustard_bottle, 2=banana (default: 0)
#   num_episodes: number of episodes to collect (default: 100)
#
# Run from starter_code: ./collect_100.sh 0 100
# Or: bash collect_100.sh 1 50

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OBJECT_IDX="${1:-0}"
NUM_EPISODES="${2:-100}"

echo "Collecting $NUM_EPISODES samples for object type $OBJECT_IDX"
echo "  0 = sugar_box, 1 = mustard_bottle, 2 = banana"
echo "Output: ../collected_data/grasp_6dof_demo_*.h5"
echo ""

for i in $(seq 1 $NUM_EPISODES); do
    echo "=============================================="
    echo "Episode $i/$NUM_EPISODES"
    echo "=============================================="
    python collect_data.py --headless --object_idx $OBJECT_IDX || true
done

echo ""
echo "Done: collected $NUM_EPISODES episodes"
ls -lh ../collected_data/grasp_6dof_demo_*.h5 2>/dev/null | tail -5
