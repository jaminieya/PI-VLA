#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_goal_rep_training_nohup.sh <checkpoint.pt> <h5_glob> [output.pt]
#
# Example:
  # ./run_goal_rep_training_nohup.sh \
  #   ../ntrl-demo/Experiments/UR5_trajectory/trajectory_03_09_20_10/Model_Epoch_04300_ValLoss_6.635179e-01.pt \
  #   "./collected_data/grasp_6dof_demo_*.h5" \
  #   goal_rep_student_film_1_1.pt

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <checkpoint.pt> <h5_glob> [output.pt]"
  exit 1
fi

CHECKPOINT="$1"
H5_GLOB="$2"
OUT_FILE="${3:-goal_rep_student_film.pt}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/train_goal_rep_alignment_${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/train_goal_rep_alignment_${TIMESTAMP}.pid"

# =========================
# Training config (edit here)
# =========================
PYTHON_BIN="python"
CONDA_ENV_NAME="rlgpu"
EPOCHS=30
BATCH_SIZE=16
LR="1e-3"
NUM_WORKERS=0
# GPU: either set DEVICE to a specific index, e.g. cuda:1, or pin a physical GPU
# with CUDA_VISIBLE_DEVICES (then usually DEVICE="cuda" — it becomes the only visible GPU).
DEVICE="cuda"
# Example: use physical GPU 1 only — uncomment:
# CUDA_VISIBLE_DEVICES="1"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
# Line-buffer stdout/stderr to the log file (helps tail -f; use with PYTHONUNBUFFERED=1 below).
USE_STDBUF=1
LOSS="mse"
IMAGE_KEY="images"
NORMALIZE_COORDS=0          # set to 1 to add --normalize_coords
# teacher_z = NTField E(q_start); joints = legacy MLP on 6D q_start
START_COND="teacher_z"
TRIPLET_WEIGHT="0.5"
TRIPLET_MARGIN="0.5"
WARMUP_PCT="0.1"
ALIGN_WEIGHT="0.0"
CONTRASTIVE_WEIGHT="1.0"
USE_INFONCE=1               # set to 1 to add --use_infonce, 0 for triplet loss
TEMPERATURE="0.07"

CMD=(
  env PYTHONUNBUFFERED=1
  "${PYTHON_BIN}" -u train_goal_rep_alignment.py
  --checkpoint "${CHECKPOINT}"
  --h5_glob "${H5_GLOB}"
  --out "${OUT_FILE}"
  --epochs "${EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --lr "${LR}"
  --num_workers "${NUM_WORKERS}"
  --device "${DEVICE}"
  --loss "${LOSS}"
  --image_key "${IMAGE_KEY}"
  --triplet_weight "${TRIPLET_WEIGHT}"
  --triplet_margin "${TRIPLET_MARGIN}"
  --warmup_pct "${WARMUP_PCT}"
  --start_cond "${START_COND}"
  --align_weight "${ALIGN_WEIGHT}"
  --contrastive_weight "${CONTRASTIVE_WEIGHT}"
  --temperature "${TEMPERATURE}"
)

if [[ "${NORMALIZE_COORDS}" == "1" ]]; then
  CMD+=(--normalize_coords)
fi

if [[ "${USE_INFONCE}" == "1" ]]; then
  CMD+=(--use_infonce)
fi

# Activate conda env (default: rlgpu) before launching nohup process.
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
else
  echo "Error: conda not found in PATH."
  exit 1
fi

cd "${SCRIPT_DIR}"

if [[ -n "${CUDA_VISIBLE_DEVICES}" ]]; then
  export CUDA_VISIBLE_DEVICES
fi

NOHUP_PREFIX=()
if [[ "${USE_STDBUF}" == "1" ]] && command -v stdbuf >/dev/null 2>&1; then
  NOHUP_PREFIX=(stdbuf -oL -eL)
fi

# Record launch configuration at the top of the run log.
{
  echo "=== train_goal_rep_alignment launch ==="
  echo "timestamp: ${TIMESTAMP}"
  echo "checkpoint: ${CHECKPOINT}"
  echo "h5_glob: ${H5_GLOB}"
  echo "out_file: ${OUT_FILE}"
  echo "conda_env: ${CONDA_ENV_NAME}"
  echo "python_bin: ${PYTHON_BIN}"
  echo "device: ${DEVICE}"
  echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-<unset>}"
  echo "epochs: ${EPOCHS}"
  echo "batch_size: ${BATCH_SIZE}"
  echo "lr: ${LR}"
  echo "num_workers: ${NUM_WORKERS}"
  echo "loss: ${LOSS}"
  echo "image_key: ${IMAGE_KEY}"
  echo "normalize_coords: ${NORMALIZE_COORDS}"
  echo "start_cond: ${START_COND}"
  echo "triplet_weight: ${TRIPLET_WEIGHT}"
  echo "triplet_margin: ${TRIPLET_MARGIN}"
  echo "warmup_pct: ${WARMUP_PCT}"
  echo "align_weight: ${ALIGN_WEIGHT}"
  echo "contrastive_weight: ${CONTRASTIVE_WEIGHT}"
  echo "use_infonce: ${USE_INFONCE}"
  echo "temperature: ${TEMPERATURE}"
  echo "command: ${CMD[*]}"
  echo "======================================"
} > "${LOG_FILE}"

nohup "${NOHUP_PREFIX[@]}" "${CMD[@]}" >> "${LOG_FILE}" 2>&1 &
PID=$!
echo "${PID}" > "${PID_FILE}"

echo "Started training."
echo "Conda env: ${CONDA_ENV_NAME}"
echo "PID: ${PID}"
echo "Log: ${LOG_FILE}"
echo "PID file: ${PID_FILE}"
echo
echo "Watch logs:"
echo "  tail -f \"${LOG_FILE}\""
echo
echo "Stop training:"
echo "  kill ${PID}"
