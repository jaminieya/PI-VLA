#!/usr/bin/env bash
set -euo pipefail

# Isaac Gym: NTField planning with student-predicted goal latent.
# Run from a machine with Isaac Gym, assets under hanwen_grasping/, etc.
#
# Usage:
#   ./run_plan_ntfield_with_student.sh <teacher_checkpoint.pt> <student.pt> <demo.h5> [-- extra python / isaac args]
#
# Example:
#   ./run_plan_ntfield_with_student.sh \
#     ntrl-demo/Experiments/UR5_trajectory/.../Model_Epoch_04300.pt \
#     hanwen_grasping/checkpoints/goal_rep_student.pt \
#     collected_data/grasp_6dof_demo_001.h5
#
# Common extras (after --):
#   --headless
#   --record --record_output out.mp4
#   --torch_device cuda:1   (force PyTorch/NTField onto another GPU; PhysX stays on --compute_device_id)
#   --torch_device cpu      (slow; if single-GPU VRAM is full after Isaac)
#
# By default, if torch sees 2+ GPUs, the script uses (compute_device_id+1) % num_gpus for teacher/student.
#
# Env:
#   PYTHON_BIN   default: python
#   HEADLESS=1   if set (default), prepends --headless unless already in extra args

# ./run_plan_ntfield_with_student.sh /home/hojinsohn/VLM-NT/PI-VLA/ntrl-demo/Experiments/UR5_trajectory/trajectory_03_09_20_10/Model_Epoch_04300_ValLoss_6.635179e-01.pt /home/hojinsohn/VLM-NT/PI-VLA/hanwen_grasping/checkpoints/goal_rep_student_0.5_weight_epoch002.pt /home/hojinsohn/VLM-NT/collected_data/grasp_6dof_demo_20260315_190955.h5 

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PI_VLA_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PI_VLA_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
HEADLESS="${HEADLESS:-1}"

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <teacher_checkpoint.pt> <student.pt> <demo.h5> [-- extra args to plan_ntfield_with_student.py]" >&2
  exit 1
fi

TEACHER_CKPT="$1"
STUDENT="$2"
H5_PATH="$3"
shift 3

EXTRA=()
if [[ $# -gt 0 && "$1" == "--" ]]; then
  shift
  EXTRA=("$@")
else
  EXTRA=("$@")
fi

prepend_headless=0
if [[ "${HEADLESS}" == "1" ]]; then
  has_headless=0
  for a in "${EXTRA[@]}"; do
    if [[ "$a" == "--headless" ]]; then has_headless=1; break; fi
  done
  if [[ "${has_headless}" -eq 0 ]]; then
    prepend_headless=1
  fi
fi

CMD=( env PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u student_model_evaluation/plan_ntfield_with_student.py )
if [[ "${prepend_headless}" -eq 1 ]]; then
  CMD+=( --headless )
fi
CMD+=(
  --ntfield
  --checkpoint "${TEACHER_CKPT}"
  --student "${STUDENT}"
  --h5_path "${H5_PATH}"
)

exec "${CMD[@]}" "${EXTRA[@]}"
