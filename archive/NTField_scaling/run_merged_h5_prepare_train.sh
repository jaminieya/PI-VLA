#!/usr/bin/env bash
# Run prepare_trajectory_dataset.py then train_arm_trajectory.py for each cumulative merged H5 level.
#
# Usage (from anywhere):
#   bash /path/to/NTField_scaling/run_merged_h5_prepare_train.sh
#
# Env overrides:
#   NUM_PAIRS=100000 PREPARE_SEED=42 EPOCHS=5000 DEVICE=cuda:0 PREPARE_ONLY=1 TRAIN_ONLY=0
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PI_VLA_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
NTRL="${PI_VLA_ROOT}/ntrl-demo"
MERGED_ROOT="${MERGED_ROOT:-${SCRIPT_DIR}/merged_h5/seed0}"
DATASET_ROOT="${DATASET_ROOT:-${NTRL}/datasets/scaling_merged/seed0}"
EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT:-${NTRL}/Experiments/UR5_scaling_merged_seed0}"

NUM_PAIRS="${NUM_PAIRS:-100000}"
PREPARE_SEED="${PREPARE_SEED:-42}"
EPOCHS="${EPOCHS:-5000}"
BATCH_SIZE="${BATCH_SIZE:-2000}"
DEVICE="${DEVICE:-cuda:0}"
SAVE_EVERY="${SAVE_EVERY:-100}"

PREPARE_ONLY="${PREPARE_ONLY:-0}"
TRAIN_ONLY="${TRAIN_ONLY:-0}"

if [[ ! -d "${MERGED_ROOT}" ]]; then
  echo "MERGED_ROOT not found: ${MERGED_ROOT}"
  echo "Run: python ${SCRIPT_DIR}/build_cumulative_merged_h5.py"
  exit 1
fi

mapfile -t CUM_DIRS < <(find "${MERGED_ROOT}" -maxdepth 1 -type d -name 'cumulative_*' | sort)

if [[ ${#CUM_DIRS[@]} -eq 0 ]]; then
  echo "No cumulative_* under ${MERGED_ROOT}"
  exit 1
fi

for CUM in "${CUM_DIRS[@]}"; do
  NAME="$(basename "${CUM}")"
  OUT_NPY="${DATASET_ROOT}/${NAME}"

  if [[ "${TRAIN_ONLY}" != "1" ]]; then
    echo "=== prepare: ${NAME} (${CUM}) -> ${OUT_NPY} ==="
    mkdir -p "${OUT_NPY}"
    python "${NTRL}/dataprocessing/prepare_trajectory_dataset.py" \
      --data_dir "${CUM}" \
      --output_dir "${OUT_NPY}" \
      --num_pairs "${NUM_PAIRS}" \
      --seed "${PREPARE_SEED}"
  fi

  if [[ "${PREPARE_ONLY}" != "1" ]]; then
    echo "=== train: ${NAME} data=${OUT_NPY} ==="
    (
      cd "${NTRL}"
      python train/train_arm_trajectory.py \
        --data_path "${OUT_NPY}" \
        --model_path "${EXPERIMENTS_ROOT}/${NAME}" \
        --device "${DEVICE}" \
        --epochs "${EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --save_every "${SAVE_EVERY}"
    )
  fi
done

echo "All done."
