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
RUN_NAME="train_config_multi_mdn"
TRAIN_SCRIPT="train_config_multi_mdn.py"

WANDB_ENABLED="true"
WANDB_LOG_BATCHES="true"
WANDB_BATCH_LOG_EVERY="10"

TRAIN_EPOCHS="90"
TRAIN_EVAL_EVERY="3"
TRAIN_BATCH_SIZE="256"
TRAIN_LR="3e-4"
TRAIN_DROPOUT="0.2"
TRAIN_MAX_PROMPT_LEN="8"

# MDN-specific: number of Gaussian mixture components
# 8 is a good default. Increase to 16 if PCA shows the model
# still under-covers the z_goal distribution after epoch 20.
MDN_N_COMPONENTS="8"

TRAIN_CKPT_NAME="best_z_goal_model_mdn_{run_name}.pth"
TRAIN_PCA_OUTPUT_DIR="/home/hojinsohn/VLM-NT/PI-VLA/output/pca_training_plots_multi_mdn"

# Integration eval — only runs when val MAE improves (safest default)
RUN_INTEGRATION_EVERY="${RUN_INTEGRATION_EVERY:-5}"
INTEGRATION_NUM_TRIALS="${INTEGRATION_NUM_TRIALS:-10}"
INTEGRATION_MAX_WORKERS="${INTEGRATION_MAX_WORKERS:-1}"

TS="$(date +%Y%m%d_%H%M%S)"
if [[ -z "${WANDB_RUN_NAME:-}" ]]; then
  lr_tag="$(printf "%s" "${TRAIN_LR}" | sed 's/\./p/g; s/-/m/g')"
  WANDB_RUN_NAME="mdn_K${MDN_N_COMPONENTS}_bs${TRAIN_BATCH_SIZE}_lr${lr_tag}_ep${TRAIN_EPOCHS}_${TS}"
fi

export WANDB_ENABLED WANDB_RUN_NAME WANDB_LOG_BATCHES WANDB_BATCH_LOG_EVERY
export TRAIN_EPOCHS TRAIN_EVAL_EVERY TRAIN_BATCH_SIZE TRAIN_LR
export TRAIN_DROPOUT TRAIN_MAX_PROMPT_LEN
export TRAIN_CKPT_NAME TRAIN_PCA_OUTPUT_DIR
export MDN_N_COMPONENTS
export RUN_INTEGRATION_EVERY INTEGRATION_NUM_TRIALS INTEGRATION_MAX_WORKERS

RUN_DIR="${REPO_ROOT}/PI-VLA/output/runs/${RUN_NAME}_${TS}"
LOG_FILE="${RUN_DIR}/train.log"
PID_FILE="${RUN_DIR}/train.pid"
mkdir -p "${RUN_DIR}"

echo "Run dir:  ${RUN_DIR}"
echo "Log file: ${LOG_FILE}"
echo "Config:   WANDB_RUN_NAME=${WANDB_RUN_NAME}"
echo "Config:   TRAIN_EPOCHS=${TRAIN_EPOCHS} TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} TRAIN_LR=${TRAIN_LR}"
echo "Config:   MDN_N_COMPONENTS=${MDN_N_COMPONENTS}"
echo "Config:   RUN_INTEGRATION_EVERY=${RUN_INTEGRATION_EVERY} INTEGRATION_NUM_TRIALS=${INTEGRATION_NUM_TRIALS}"

cd "${SCRIPT_DIR}"
nohup python -u "${TRAIN_SCRIPT}" > "${LOG_FILE}" 2>&1 &
echo $! > "${PID_FILE}"
echo "Started MDN training. PID=$(<"${PID_FILE}")"
echo "Tail logs: tail -f \"${LOG_FILE}\""