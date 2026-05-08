#!/usr/bin/env bash
set -euo pipefail

# Ensure conda is available in non-interactive shells.
# You can override with CONDA_SH=/path/to/conda.sh and/or CONDA_ENV=your_env
CONDA_ENV="${CONDA_ENV:-rlgpu}"
CONDA_SH="${CONDA_SH:-}"

if [[ -z "${CONDA_SH}" ]]; then
  if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    CONDA_SH="${HOME}/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    CONDA_SH="${HOME}/anaconda3/etc/profile.d/conda.sh"
  elif command -v conda >/dev/null 2>&1; then
    CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
  fi
fi

if [[ -z "${CONDA_SH}" || ! -f "${CONDA_SH}" ]]; then
  echo "ERROR: Could not find conda.sh. Set CONDA_SH to your conda.sh path." >&2
  echo "Example: CONDA_SH=~/miniconda3/etc/profile.d/conda.sh bash train_config_multi.sh" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

# Runs full_train_multi.py under nohup and writes logs + PID.
# Usage:
#   bash train_config_multi.sh
#   CUDA_VISIBLE_DEVICES=0 bash train_config_multi.sh
#   RUN_NAME=my_multi_run bash train_config_multi.sh
#   RUN_NAME=exp1 WANDB_RUN_NAME=multi_v1 TRAIN_EPOCHS=50 TRAIN_LOSS_TYPE=mse bash train_config_multi.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

# ---------------------------------------------------------------------------
# Training config defaults (edit here)
# You can still override any of these from CLI, e.g.:
#   TRAIN_EPOCHS=50 TRAIN_LOSS_TYPE=mse bash train_config_multi.sh
# ---------------------------------------------------------------------------
RUN_NAME="train_config_multi"
TRAIN_SCRIPT="full_train_multi.py"
WANDB_ENABLED="true"
WANDB_LOG_BATCHES="true"
WANDB_BATCH_LOG_EVERY="10"
TRAIN_EPOCHS="100"
TRAIN_EVAL_EVERY="10"
TRAIN_BATCH_SIZE="128"
TRAIN_LR="1e-4"
TRAIN_LOSS_TYPE="hybrid"   # hybrid | mse | cos
TRAIN_LOSS_ALPHA="0.5"     # used when TRAIN_LOSS_TYPE=hybrid
TRAIN_CKPT_NAME="best_z_goal_model_multi_hybrid_0.5_loc_head2_{run_name}.pth" # supports {run_name}
TRAIN_PCA_OUTPUT_DIR="/home/hojinsohn/VLM-NT/PI-VLA/output/pca_training_plots_multi_loc_head2"
TRAIN_INFONCE_WEIGHT="0.0"
TRAIN_INFONCE_TEMP="0.1"
TRAIN_MAX_PROMPT_LEN="8"

TS="$(date +%Y%m%d_%H%M%S)"
if [[ -z "${WANDB_RUN_NAME:-}" ]]; then
  lr_tag="$(printf "%s" "${TRAIN_LR}" | sed 's/\./p/g; s/-/m/g')"
  WANDB_RUN_NAME="${TRAIN_LOSS_TYPE}_bs${TRAIN_BATCH_SIZE}_lr${lr_tag}_ep${TRAIN_EPOCHS}_${TS}"
fi

# Export so full_train_multi.py can read via os.getenv(...)
export WANDB_ENABLED WANDB_RUN_NAME WANDB_LOG_BATCHES WANDB_BATCH_LOG_EVERY
export TRAIN_EPOCHS TRAIN_EVAL_EVERY TRAIN_BATCH_SIZE TRAIN_LR
export TRAIN_LOSS_TYPE TRAIN_LOSS_ALPHA TRAIN_CKPT_NAME TRAIN_PCA_OUTPUT_DIR
export TRAIN_INFONCE_WEIGHT TRAIN_INFONCE_TEMP TRAIN_MAX_PROMPT_LEN

RUN_DIR="${REPO_ROOT}/PI-VLA/output/runs/${RUN_NAME}_${TS}"
LOG_FILE="${RUN_DIR}/train.log"
PID_FILE="${RUN_DIR}/train.pid"

mkdir -p "${RUN_DIR}"

echo "Repo root: ${REPO_ROOT}"
echo "Script dir: ${SCRIPT_DIR}"
echo "Run dir:    ${RUN_DIR}"
echo "Log file:   ${LOG_FILE}"
echo "Train file: ${TRAIN_SCRIPT}"
echo "Config:     WANDB_RUN_NAME=${WANDB_RUN_NAME} WANDB_ENABLED=${WANDB_ENABLED} WANDB_LOG_BATCHES=${WANDB_LOG_BATCHES} WANDB_BATCH_LOG_EVERY=${WANDB_BATCH_LOG_EVERY}"
echo "Config:     TRAIN_EPOCHS=${TRAIN_EPOCHS} TRAIN_EVAL_EVERY=${TRAIN_EVAL_EVERY} TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} TRAIN_LR=${TRAIN_LR}"
echo "Config:     TRAIN_LOSS_TYPE=${TRAIN_LOSS_TYPE} TRAIN_LOSS_ALPHA=${TRAIN_LOSS_ALPHA}"
echo "Config:     TRAIN_CKPT_NAME=${TRAIN_CKPT_NAME}"
echo "Config:     TRAIN_PCA_OUTPUT_DIR=${TRAIN_PCA_OUTPUT_DIR}"
echo "Config:     TRAIN_MAX_PROMPT_LEN=${TRAIN_MAX_PROMPT_LEN}"

cd "${SCRIPT_DIR}"

# shellcheck disable=SC2086
nohup python -u "${TRAIN_SCRIPT}" > "${LOG_FILE}" 2>&1 &
echo $! > "${PID_FILE}"

echo "Started training. PID=$(cat "${PID_FILE}")"
echo "Tail logs with: tail -f \"${LOG_FILE}\""
