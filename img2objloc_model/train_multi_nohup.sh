#!/bin/bash

# ---------------------------------------------------------------------------
# train_multi_nohup.sh
# Runs train_multi.py in the background via nohup.
# Logs are written to LOG_FILE.
# Usage:
#   chmod +x train_multi_nohup.sh
#   ./train_multi_nohup.sh
# ---------------------------------------------------------------------------

CONDA_ENV="rlgpu"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$SCRIPT_DIR/train_multi.py"

DATA_DIR="/home/hojinsohn/VLM-NT/PI-VLA/output/multi_obj_layout_shards"
OUTPUT_DIR="/home/hojinsohn/VLM-NT/PI-VLA/output/runs/multi_obj_layout_exp01"
LOG_FILE="$OUTPUT_DIR/train.log"

EPOCHS=50
BATCH_SIZE=64
LR=3e-4
NUM_WORKERS=4
SAVE_EVERY=10
MAX_SHARDS=30 # for debugging

# ---------------------------------------------------------------------------
mkdir -p "$OUTPUT_DIR"

# Activate conda env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "Starting training at $(date)"
echo "Logging to $LOG_FILE"

# -u: unbuffered stdout/stderr so train.log updates live under nohup redirection
nohup python -u "$SCRIPT" \
    --data-dir      "$DATA_DIR" \
    --output-dir    "$OUTPUT_DIR" \
    --epochs        "$EPOCHS" \
    --batch-size    "$BATCH_SIZE" \
    --lr            "$LR" \
    --num-workers   "$NUM_WORKERS" \
    --save-every    "$SAVE_EVERY" \
    --max-shards    "$MAX_SHARDS" \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "Launched with PID $PID"
echo "$PID" > "$OUTPUT_DIR/train.pid"
echo "To monitor: tail -f $LOG_FILE"
echo "To stop:    kill \$(cat $OUTPUT_DIR/train.pid)"