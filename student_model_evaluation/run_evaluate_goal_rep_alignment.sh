#!/usr/bin/env bash
set -euo pipefail

# Offline eval: student z_hat vs teacher z_goal on HDF5 grasp demos.
#
# Usage:
#   ./run_evaluate_goal_rep_alignment.sh <student.pt> <h5_glob> [-- extra python args]
#
# Example:
#   ./run_evaluate_goal_rep_alignment.sh \
#     hanwen_grasping/checkpoints/goal_rep_student.pt \
#     "collected_data/grasp_6dof_demo_*.h5"
#
# Env:
#   PYTHON_BIN  default: python
#   DEVICE      default: cuda

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PI_VLA_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PI_VLA_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <student.pt> <h5_glob> [-- extra args to evaluate_goal_rep_alignment.py]" >&2
  exit 1
fi

STUDENT="$1"
H5_GLOB="$2"
shift 2

EXTRA=()
if [[ $# -gt 0 && "$1" == "--" ]]; then
  shift
  EXTRA=("$@")
else
  EXTRA=("$@")
fi

exec env PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u student_model_evaluation/evaluate_goal_rep_alignment.py \
  --student "${STUDENT}" \
  --h5_glob "${H5_GLOB}" \
  --device "${DEVICE}" \
  "${EXTRA[@]}"
