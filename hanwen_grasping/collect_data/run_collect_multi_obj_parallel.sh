#!/usr/bin/env bash

set -euo pipefail

# Run collect_multi_obj_data_for_student.py in parallel across GPUs.
# Usage:
#   ./run_collect_multi_obj_parallel.sh
#   ./run_collect_multi_obj_parallel.sh --total-episodes 10000 --gpus 0,1,2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/collect_multi_obj_data_for_student.py"

TOTAL_EPISODES=10000
MAX_PLAN_ATTEMPTS=10
GPUS="0,1,2"
HEADLESS=1
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --total-episodes)
      TOTAL_EPISODES="$2"
      shift 2
      ;;
    --max-plan-attempts)
      MAX_PLAN_ATTEMPTS="$2"
      shift 2
      ;;
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --no-headless)
      HEADLESS=0
      shift
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      echo "Unknown option: $1"
      echo "Supported options: --total-episodes N --max-plan-attempts N --gpus 0,1,2 --no-headless -- [extra args]"
      exit 1
      ;;
  esac
done

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
  echo "Cannot find script: ${PYTHON_SCRIPT}"
  exit 1
fi

IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
NUM_GPUS="${#GPU_LIST[@]}"
if [[ "${NUM_GPUS}" -lt 1 ]]; then
  echo "No GPUs specified."
  exit 1
fi

BASE=$((TOTAL_EPISODES / NUM_GPUS))
REM=$((TOTAL_EPISODES % NUM_GPUS))

declare -a PIDS=()
declare -a JOBS=()

echo "Launching ${NUM_GPUS} jobs for ${TOTAL_EPISODES} total episodes"
echo "GPUs: ${GPUS}"
echo "max_plan_attempts=${MAX_PLAN_ATTEMPTS}, headless=${HEADLESS}"

for i in "${!GPU_LIST[@]}"; do
  GPU_ID="${GPU_LIST[$i]}"
  EPISODES="${BASE}"
  if [[ "${i}" -lt "${REM}" ]]; then
    EPISODES=$((EPISODES + 1))
  fi

  # Skip empty slices when total episodes < number of GPUs.
  if [[ "${EPISODES}" -le 0 ]]; then
    continue
  fi

  CMD=(python "${PYTHON_SCRIPT}" --num_episodes "${EPISODES}" --max_plan_attempts "${MAX_PLAN_ATTEMPTS}" --env_id "${i}")
  if [[ "${HEADLESS}" -eq 1 ]]; then
    CMD+=(--headless)
  fi
  if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
    CMD+=("${EXTRA_ARGS[@]}")
  fi

  LOG_FILE="${SCRIPT_DIR}/collect_gpu${GPU_ID}_env${i}.log"
  echo "[job ${i}] GPU=${GPU_ID} env_id=${i} episodes=${EPISODES} log=${LOG_FILE}"

  (
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
    "${CMD[@]}"
  ) >"${LOG_FILE}" 2>&1 &

  PIDS+=("$!")
  JOBS+=("gpu=${GPU_ID},env_id=${i},episodes=${EPISODES}")
done

if [[ "${#PIDS[@]}" -eq 0 ]]; then
  echo "No jobs launched (check --total-episodes and --gpus)."
  exit 1
fi

FAIL=0
for idx in "${!PIDS[@]}"; do
  PID="${PIDS[$idx]}"
  JOB="${JOBS[$idx]}"
  if wait "${PID}"; then
    echo "[done] ${JOB}"
  else
    echo "[fail] ${JOB}"
    FAIL=1
  fi
done

if [[ "${FAIL}" -ne 0 ]]; then
  echo "One or more jobs failed. Check per-job logs in ${SCRIPT_DIR}."
  exit 1
fi

echo "All jobs completed successfully."
