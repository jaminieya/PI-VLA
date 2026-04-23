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
  echo "ERROR: Could not find conda.sh. Set CONDA_SH to your conda.sh path." >&2
  exit 1
fi

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

RUN_NAME="${RUN_NAME:-train_multi_obj}"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${REPO_ROOT}/PI-VLA/output/runs/${RUN_NAME}_${TS}"
LOG_FILE="${RUN_DIR}/train.log"
PID_FILE="${RUN_DIR}/train.pid"

mkdir -p "${RUN_DIR}"

echo "Repo root: ${REPO_ROOT}"
echo "Script dir: ${SCRIPT_DIR}"
echo "Run dir:    ${RUN_DIR}"
echo "Log file:   ${LOG_FILE}"

cd "${SCRIPT_DIR}"

nohup python -u train_multi_obj.py > "${LOG_FILE}" 2>&1 &
echo $! > "${PID_FILE}"

echo "Started training. PID=$(cat "${PID_FILE}")"
echo "Tail logs with: tail -f \"${LOG_FILE}\""
