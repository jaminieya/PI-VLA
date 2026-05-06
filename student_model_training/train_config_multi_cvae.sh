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
  echo "Example: CONDA_SH=~/miniconda3/etc/profile.d/conda.sh bash train_config_multi_cvae.sh" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

# Runs full_train_multi_cvae.py under nohup and writes logs + PID.
# Usage:
#   bash train_config_multi_cvae.sh
#   CUDA_VISIBLE_DEVICES=0 bash train_config_multi_cvae.sh
#   WANDB_RUN_NAME=my_custom_name bash train_config_multi_cvae.sh
#   TRAIN_EPOCHS=50 TRAIN_BATCH_SIZE=64 TRAIN_LR=5e-5 bash train_config_multi_cvae.sh
#   TRAIN_KL_BETA_SCHEDULE=linear TRAIN_KL_BETA_END=2.0 TRAIN_KL_ANNEAL_STEPS=10000 ...
#   RUN_INTEGRATION_EVERY=10 INTEGRATION_NUM_TRIALS=5 INTEGRATION_MAX_WORKERS=2 bash train_config_multi_cvae.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

# ---------------------------------------------------------------------------
# Training config defaults (edit here)
# You can still override any of these from CLI, e.g.:
#   TRAIN_EPOCHS=50 TRAIN_BATCH_SIZE=64 bash train_config_multi_cvae.sh
# ---------------------------------------------------------------------------
RUN_NAME="train_config_multi_cvae"
TRAIN_SCRIPT="full_train_multi_cvae.py"
WANDB_ENABLED="true"
WANDB_LOG_BATCHES="true"
WANDB_BATCH_LOG_EVERY="10"
TRAIN_EPOCHS="60"
TRAIN_EVAL_EVERY="3"
TRAIN_BATCH_SIZE="256"
TRAIN_LR="3e-4"
TRAIN_MAX_PROMPT_LEN="8"
TRAIN_DROPOUT="0.2"

# KL schedule (CVAE) — linear warmup so decoder stabilizes before full KL pressure
TRAIN_KL_BETA_START="0.0"
TRAIN_KL_BETA_END="1.0"
TRAIN_KL_BETA_SCHEDULE="linear"
TRAIN_KL_ANNEAL_STEPS="15000"
TRAIN_KL_CYCLE_STEPS="3000"
TRAIN_CKPT_NAME="best_z_goal_model_multi_cvae_{run_name}.pth" # supports {run_name}
TRAIN_PCA_OUTPUT_DIR="/home/hojinsohn/VLM-NT/PI-VLA/output/pca_training_plots_multi_cvae"

# Integrated Isaac Gym + NTField eval during training (full_train_multi_cvae.py)
RUN_INTEGRATION_EVERY="${RUN_INTEGRATION_EVERY:-3}"
INTEGRATION_NUM_TRIALS="${INTEGRATION_NUM_TRIALS:-10}"
# W&B: upload first N trial mp4s per integration (keys include epoch so runs stay browsable).
INTEGRATION_VIDEOS_PER_EVAL="${INTEGRATION_VIDEOS_PER_EVAL:-8}"
# Playback fps passed to wandb.Video (Isaac mp4 is often 60; 30 is a reasonable default for UI).
INTEGRATION_WANDB_VIDEO_FPS="${INTEGRATION_WANDB_VIDEO_FPS:-30}"
# >1 often segfaults Isaac Gym (GPU PhysX) on one GPU (subprocess exit=-11). Override only if safe.
INTEGRATION_MAX_WORKERS="${INTEGRATION_MAX_WORKERS:-1}"

TS="$(date +%Y%m%d_%H%M%S)"

# Build a unique wandb run name from key settings unless user already set one.
if [[ -z "${WANDB_RUN_NAME:-}" ]]; then
  lr_tag="$(printf "%s" "${TRAIN_LR}" | sed 's/\./p/g; s/-/m/g')"
  WANDB_RUN_NAME="cvae_bs${TRAIN_BATCH_SIZE}_lr${lr_tag}_ep${TRAIN_EPOCHS}_${TS}"
fi

# Export so full_train_multi_cvae.py can read via os.getenv(...)
export WANDB_ENABLED WANDB_RUN_NAME WANDB_LOG_BATCHES WANDB_BATCH_LOG_EVERY
export TRAIN_EPOCHS TRAIN_EVAL_EVERY TRAIN_BATCH_SIZE TRAIN_LR
export TRAIN_CKPT_NAME TRAIN_PCA_OUTPUT_DIR
export TRAIN_MAX_PROMPT_LEN TRAIN_DROPOUT
export TRAIN_KL_BETA_START TRAIN_KL_BETA_END TRAIN_KL_BETA_SCHEDULE
export TRAIN_KL_CYCLE_STEPS TRAIN_KL_ANNEAL_STEPS
export RUN_INTEGRATION_EVERY INTEGRATION_NUM_TRIALS INTEGRATION_MAX_WORKERS
export INTEGRATION_VIDEOS_PER_EVAL INTEGRATION_WANDB_VIDEO_FPS

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
echo "Config:     TRAIN_KL_BETA_START=${TRAIN_KL_BETA_START} TRAIN_KL_BETA_END=${TRAIN_KL_BETA_END} TRAIN_KL_BETA_SCHEDULE=${TRAIN_KL_BETA_SCHEDULE}"
echo "Config:     TRAIN_KL_ANNEAL_STEPS=${TRAIN_KL_ANNEAL_STEPS} TRAIN_KL_CYCLE_STEPS=${TRAIN_KL_CYCLE_STEPS}"
echo "Config:     TRAIN_CKPT_NAME=${TRAIN_CKPT_NAME}"
echo "Config:     TRAIN_PCA_OUTPUT_DIR=${TRAIN_PCA_OUTPUT_DIR}"
echo "Config:     TRAIN_MAX_PROMPT_LEN=${TRAIN_MAX_PROMPT_LEN} TRAIN_DROPOUT=${TRAIN_DROPOUT}"
echo "Config:     RUN_INTEGRATION_EVERY=${RUN_INTEGRATION_EVERY} INTEGRATION_NUM_TRIALS=${INTEGRATION_NUM_TRIALS} INTEGRATION_MAX_WORKERS=${INTEGRATION_MAX_WORKERS}"
echo "Config:     INTEGRATION_VIDEOS_PER_EVAL=${INTEGRATION_VIDEOS_PER_EVAL} INTEGRATION_WANDB_VIDEO_FPS=${INTEGRATION_WANDB_VIDEO_FPS}"

cd "${SCRIPT_DIR}"

nohup python -u "${TRAIN_SCRIPT}" > "${LOG_FILE}" 2>&1 &
echo $! > "${PID_FILE}"

echo "Started training. PID=$(<"${PID_FILE}")"
echo "Tail logs with: tail -f \"${LOG_FILE}\""
