#!/usr/bin/env bash
set -euo pipefail

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
  echo "ERROR: Could not find conda.sh. Set CONDA_SH=/path/to/conda.sh" >&2
  exit 1
fi
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

# ── Config ────────────────────────────────────────────────────────────────────
RUN_NAME="full_train_multi_regression"
TRAIN_SCRIPT="full_train_multi_regression.py"

# ── Loss config ───────────────────────────────────────────────────────────────
# Options: mse | cosine | contrastive | hybrid
TRAIN_LOSS_TYPE="hybrid"

# Weights for hybrid mode (ignored for other modes)
TRAIN_MSE_WEIGHT="0.5"
TRAIN_COS_WEIGHT="0.5"
TRAIN_CONTRA_WEIGHT="0.5"   # set > 0 to enable contrastive term inside hybrid
TRAIN_TEMPERATURE="0.07"    # contrastive softmax temperature

# ── Wandb config ───────────────────────────────────────────────────────────────
WANDB_ENABLED="true"
WANDB_LOG_BATCHES="true"
WANDB_BATCH_LOG_EVERY="10"

TRAIN_EPOCHS="40"
TRAIN_EVAL_EVERY="3"
TRAIN_BATCH_SIZE="256"
TRAIN_LR="3e-4"
TRAIN_DROPOUT="0.2"

TRAIN_CKPT_NAME="best_z_goal_model_regression_{run_name}.pth"
TRAIN_PCA_OUTPUT_BASE="/home/hojinsohn/VLM-NT/PI-VLA/output/pca_training_plots_multi_regression"
TRAIN_PCA_OUTPUT_DIR="${TRAIN_PCA_OUTPUT_BASE}/${TRAIN_LOSS_TYPE}"

# Integration eval — only runs when val MAE improves (safest default)
RUN_INTEGRATION_EVERY="90" # Do not run...
INTEGRATION_NUM_TRIALS="10" # Do not run...
INTEGRATION_MAX_WORKERS="1" # Do not run...

TS="$(date +%Y%m%d_%H%M%S)"
if [[ -z "${WANDB_RUN_NAME:-}" ]]; then
  lr_tag="$(printf "%s" "${TRAIN_LR}" | sed 's/\./p/g; s/-/m/g')"
  WANDB_RUN_NAME="${TRAIN_LOSS_TYPE}_bs${TRAIN_BATCH_SIZE}_lr${lr_tag}_ep${TRAIN_EPOCHS}_${TS}"
fi

export WANDB_ENABLED WANDB_RUN_NAME WANDB_LOG_BATCHES WANDB_BATCH_LOG_EVERY
export TRAIN_EPOCHS TRAIN_EVAL_EVERY TRAIN_BATCH_SIZE TRAIN_LR TRAIN_DROPOUT
export TRAIN_LOSS_TYPE TRAIN_MSE_WEIGHT TRAIN_COS_WEIGHT TRAIN_CONTRA_WEIGHT TRAIN_TEMPERATURE
export TRAIN_CKPT_NAME TRAIN_PCA_OUTPUT_DIR
export RUN_INTEGRATION_EVERY INTEGRATION_NUM_TRIALS INTEGRATION_MAX_WORKERS

RUN_DIR="${REPO_ROOT}/PI-VLA/output/runs/${RUN_NAME}_${TS}"
LOG_FILE="${RUN_DIR}/train.log"
PID_FILE="${RUN_DIR}/train.pid"
mkdir -p "${RUN_DIR}"

echo "Run dir:  ${RUN_DIR}"
echo "Log file: ${LOG_FILE}"
echo "Config:   WANDB_RUN_NAME=${WANDB_RUN_NAME}"
echo "Config:   TRAIN_EPOCHS=${TRAIN_EPOCHS} TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} TRAIN_LR=${TRAIN_LR}"
echo "Config:   RUN_INTEGRATION_EVERY=${RUN_INTEGRATION_EVERY} INTEGRATION_NUM_TRIALS=${INTEGRATION_NUM_TRIALS}"

cd "${SCRIPT_DIR}"
nohup python -u "${TRAIN_SCRIPT}" > "${LOG_FILE}" 2>&1 &
echo $! > "${PID_FILE}"
echo "Started regression training. PID=$(<"${PID_FILE}")"
echo "Tail logs: tail -f \"${LOG_FILE}\""