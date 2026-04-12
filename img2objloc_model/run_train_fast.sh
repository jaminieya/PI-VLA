#!/bin/bash

# ---------------------------------------------------------------------------
# run_train_fast.sh
# Runs train_fast.py in the background via nohup.
# Logs are written to OUTPUT_DIR/train.log
# Usage:
#   chmod +x run_train_fast.sh
#   ./run_train_fast.sh
# ---------------------------------------------------------------------------

CONDA_ENV="rlgpu"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$SCRIPT_DIR/train_fast.py"

DATA_DIR="/home/hojinsohn/VLM-NT/PI-VLA/output/multi_obj_layout_feat_shards"
OUTPUT_DIR="/home/hojinsohn/VLM-NT/PI-VLA/output/runs/exp_fast"
LOG_FILE="$OUTPUT_DIR/train.log"

EPOCHS=100
BATCH_SIZE=512
LR=3e-4
NUM_WORKERS=4
SAVE_EVERY=10

# ---------------------------------------------------------------------------
mkdir -p "$OUTPUT_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "Starting train_fast at $(date)"
echo "Logging to $LOG_FILE"

nohup python "$SCRIPT" \
    --data-dir      "$DATA_DIR" \
    --output-dir    "$OUTPUT_DIR" \
    --epochs        "$EPOCHS" \
    --batch-size    "$BATCH_SIZE" \
    --lr            "$LR" \
    --num-workers   "$NUM_WORKERS" \
    --save-every    "$SAVE_EVERY" \
    --preload \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "Launched with PID $PID"
echo "$PID" > "$OUTPUT_DIR/train.pid"
echo "To monitor: tail -f $LOG_FILE"
echo "To stop:    kill \$(cat $OUTPUT_DIR/train.pid)"