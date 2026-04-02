#!/usr/bin/env bash
set -euo pipefail

# Single-image inference: prompt + RGB + q_start -> z_goal_hat (optional teacher compare via q_goal).
#
# Usage:
#   ./run_infer_goal_rep_latent.sh <student.pt> <image> <prompt> <q_start> [-- extra python args]
#
# Example:
#   ./run_infer_goal_rep_latent.sh checkpoints/student.pt frame.png "grasp the apple" "0.1,-0.2,0,1.57,1.57,0"
#
# Optional env:
#   Q_GOAL    if set, passed as --q_goal (6 comma-separated radians)
#   OUT_Z     if set, passed as --out_z (path for .npy)
#   DEVICE    default: cuda
#   PYTHON_BIN

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PI_VLA_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PI_VLA_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <student.pt> <image> <prompt> <q_start> [-- extra args to infer_goal_rep_latent.py]" >&2
  exit 1
fi

STUDENT="$1"
IMAGE="$2"
PROMPT="$3"
Q_START="$4"
shift 4

EXTRA=()
if [[ $# -gt 0 && "$1" == "--" ]]; then
  shift
  EXTRA=("$@")
else
  EXTRA=("$@")
fi

CMD=(
  env PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u student_model_evaluation/infer_goal_rep_latent.py
  --student "${STUDENT}"
  --image "${IMAGE}"
  --prompt "${PROMPT}"
  --q_start "${Q_START}"
  --device "${DEVICE}"
)

if [[ -n "${OUT_Z:-}" ]]; then
  CMD+=( --out_z "${OUT_Z}" )
fi
if [[ -n "${Q_GOAL:-}" ]]; then
  CMD+=( --q_goal "${Q_GOAL}" )
fi

exec "${CMD[@]}" "${EXTRA[@]}"
