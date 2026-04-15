#!/usr/bin/env bash
set -euo pipefail

# Parallel launcher for new_clean_data_collect/collect_ntfield_rrt_episodes.py
#
# Usage:
#   bash new_clean_data_collect/run_3_gpu_parallel_collect.sh [total_episodes] [output_root]
#
# Example:
#   bash new_clean_data_collect/run_3_gpu_parallel_collect.sh 1000 output/data_collection/20260408_parallel
#
# Notes:
# - Spawns 3 workers on CUDA_VISIBLE_DEVICES=0,1,2
# - Each worker writes to its own folder: <output_root>/worker_gpu{0,1,2}
# - Passes --resume so workers can be restarted safely.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PI_VLA_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PI_VLA_ROOT}"

TOTAL_EPISODES="${1:-1000}"
OUTPUT_ROOT="${2:-output/data_collection/20260408_parallel}"
NUM_WORKERS=3

if ! [[ "${TOTAL_EPISODES}" =~ ^[0-9]+$ ]]; then
  echo "Error: total_episodes must be a non-negative integer"
  exit 1
fi

BASE=$(( TOTAL_EPISODES / NUM_WORKERS ))
REM=$(( TOTAL_EPISODES % NUM_WORKERS ))

echo "Parallel NTField-RRT collection"
echo "  total episodes: ${TOTAL_EPISODES}"
echo "  workers: ${NUM_WORKERS} (GPU 0,1,2)"
echo "  output root: ${OUTPUT_ROOT}"
echo

for gpu in 0 1 2; do
  extra=0
  if [[ ${gpu} -lt ${REM} ]]; then
    extra=1
  fi
  worker_eps=$(( BASE + extra ))
  worker_dir="${OUTPUT_ROOT}/worker_gpu${gpu}"
  log_file="${OUTPUT_ROOT}/collect_gpu${gpu}.log"
  mkdir -p "${worker_dir}"
  mkdir -p "$(dirname "${log_file}")"

  if [[ ${worker_eps} -eq 0 ]]; then
    echo "GPU ${gpu}: skip (0 episodes assigned)"
    continue
  fi

  echo "GPU ${gpu}: ${worker_eps} episodes -> ${worker_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" nohup python new_clean_data_collect/collect_ntfield_rrt_episodes.py \
    --num_episodes "${worker_eps}" \
    --output_dir "${worker_dir}" \
    --seed "$((42 + gpu))" \
    --resume \
    > "${log_file}" 2>&1 &
  echo "  started pid=$! log=${log_file}"
done

echo
echo "All workers launched."
echo "Monitor: tail -f ${OUTPUT_ROOT}/collect_gpu0.log"
echo "After completion, merge H5s:"
echo "  python new_clean_data_collect/merge_worker_h5.py --input_root ${OUTPUT_ROOT} --output_dir output/data_collection/20260408"
