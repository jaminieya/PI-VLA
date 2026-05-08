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
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PI_VLA_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

TEST_DATASET="/home/hojinsohn/VLM-NT/PI-VLA/hanwen_grasping/output/multi_obj/test_run_pt_shards"
NTFIELD_CHECKPOINT="/home/hojinsohn/VLM-NT/PI-VLA/teacher_model.pt"

NTFIELD_STEP_SIZE="0.02"
NTFIELD_MAX_STEPS="200"
NTFIELD_TOL="0.01"
NTFIELD_DEVICE="cuda:0"
# Success thresholds in metres
# EE-to-target (wrist proxy)
EE_SUCCESS_THRESH="0.10"
# Finger-midpoint XY distance and |Z diff| to target center
FINGER_MID_XY_THRESH="0.10"
FINGER_MID_Z_THRESH="0.10"

# Set > 0 for a quick sanity check before running the full dataset
MAX_RECORDS="15"

TS="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${PI_VLA_ROOT}/output/eval_results/ntfield_oracle_${TS}"
mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/eval.log"

echo "Test dataset : ${TEST_DATASET}"
echo "Output dir   : ${OUTPUT_DIR}"
echo "Log          : ${LOG_FILE}"

cd "${PI_VLA_ROOT}"
nohup python -u model_evaluation/evaluate_ntfield_oracle.py \
  --test_dataset       "${TEST_DATASET}" \
  --ntfield_checkpoint "${NTFIELD_CHECKPOINT}" \
  --output_dir         "${OUTPUT_DIR}" \
  --ntfield_device     "${NTFIELD_DEVICE}" \
  --ntfield_step_size  "${NTFIELD_STEP_SIZE}" \
  --ntfield_max_steps  "${NTFIELD_MAX_STEPS}" \
  --ntfield_tol        "${NTFIELD_TOL}" \
  --ee_success_thresh  "${EE_SUCCESS_THRESH}" \
  --finger_mid_xy_success_thresh_m "${FINGER_MID_XY_THRESH}" \
  --finger_mid_z_success_thresh_m  "${FINGER_MID_Z_THRESH}" \
  --max_records        "${MAX_RECORDS}" \
  > "${LOG_FILE}" 2>&1 &

echo "Started. PID=$!"
echo "Tail: tail -f \"${LOG_FILE}\""