#!/usr/bin/env bash
# Replay an HDF5 trajectory vs NTField-planned trajectory in Isaac Gym; save MP4s under output/trajectory_evaluation/.
#
# Typical flow: trajectory_evaluation/ntfield/collect_data.py saves a new .h5 under
#   output/trajectory_evaluation/YYYYMMDD_HHMMSS/, then runs this script (unless --no_run_ntfield_demo).
#
# Usage (from anywhere):
#   bash trajectory_evaluation/ntfield/run_isaac_ntfield_demo.sh
#   bash trajectory_evaluation/ntfield/run_isaac_ntfield_demo.sh /path/to/demo.h5 /path/to/Model_RRT_train.pt
# Optional 3rd arg: checkpoint trained WITHOUT RRT labels (e.g. straight-line dataset) -> ntfield_straightline.mp4
#   bash trajectory_evaluation/ntfield/run_isaac_ntfield_demo.sh /path/to/demo.h5 /path/to/rrt_ckpt.pt /path/to/straightline_ckpt.pt
#
# Requires: Isaac Gym, conda env with isaacgym + torch + imageio[ffmpeg] or opencv-python.
# Interactive: needs DISPLAY (open viewer until recording finishes).
# Headless:   HEADLESS=1 bash ...  (no window; fixed-length replay; may still need GPU/EGL for camera rendering on some hosts)
#
# Run from PI-VLA root if using default relative paths.

set -euo pipefail

PI_VLA_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_DIR="${PI_VLA_ROOT}/output/trajectory_evaluation"
mkdir -p "${OUT_DIR}"

DEFAULT_H5="${PI_VLA_ROOT}/output/data_collection/test/grasp_6dof_demo_20260315_220503.h5"
DEFAULT_CKPT="${PI_VLA_ROOT}/ntrl-demo/Experiments/UR5_trajectory_no_wall_accuracy_check/trajectory_03_25_20_28/Model_Epoch_05000_ValLoss_7.820605e-01.pt"

H5="${1:-$DEFAULT_H5}"
CKPT="${2:-$DEFAULT_CKPT}"
CKPT_STRAIGHTLINE="${3:-}"

if [[ ! -f "$H5" ]]; then
  echo "HDF5 not found: $H5"
  echo "Pass an existing file, e.g. under output/data_collection/test/*.h5"
  exit 1
fi
if [[ ! -f "$CKPT" ]]; then
  echo "Checkpoint not found: $CKPT"
  exit 1
fi

STEM="$(basename "$H5" .h5)"
SESSION_DIR="${OUT_DIR}"
if [[ "$STEM" =~ [0-9]{8}_[0-9]{6} ]]; then
  STAMP="${BASH_REMATCH[0]}"
  SESSION_DIR="${OUT_DIR}/${STAMP}"
fi
mkdir -p "${SESSION_DIR}"
ORIG_MP4="${SESSION_DIR}/original.mp4"
NT_MP4="${SESSION_DIR}/ntfield.mp4"

cd "${PI_VLA_ROOT}/hanwen_grasping"

HEADLESS_ARGS=()
if [[ "${HEADLESS:-0}" == "1" ]]; then
  HEADLESS_ARGS=(--headless)
  echo "HEADLESS=1: passing --headless (no Isaac viewer)"
fi

# Default --record_output in Python uses this session dir + episode_meta.txt when the h5 basename contains YYYYMMDD_HHMMSS.
RC_ARGS=(--h5_path "$H5" --record --interpolate 4 --no_walls "${HEADLESS_ARGS[@]}")
NS_ARGS=(--ntfield --checkpoint "$CKPT" --h5_path "$H5" --record --no_walls "${HEADLESS_ARGS[@]}")
if [[ "$SESSION_DIR" != "$OUT_DIR" ]]; then
  echo "=== Session directory: ${SESSION_DIR}"
else
  RC_ARGS+=(--record_output "$ORIG_MP4")
  NS_ARGS+=(--record_output "$NT_MP4")
  echo "=== No YYYYMMDD_HHMMSS in h5 basename; writing flat under output/trajectory_evaluation/"
fi

echo "=== Original trajectory replay -> ${ORIG_MP4}"
python run_collected_trajectory.py "${RC_ARGS[@]}"

echo "=== NTField plan (RRT / trajectory-supervised checkpoint) -> ${NT_MP4}"
python new_setup.py "${NS_ARGS[@]}"

if [[ -n "${CKPT_STRAIGHTLINE}" ]]; then
  if [[ ! -f "${CKPT_STRAIGHTLINE}" ]]; then
    echo "Straight-line NTField checkpoint not found: ${CKPT_STRAIGHTLINE}"
    exit 1
  fi
  NT_SL_MP4="${SESSION_DIR}/ntfield_straightline.mp4"
  NS_SL_ARGS=(--ntfield --checkpoint "$CKPT_STRAIGHTLINE" --h5_path "$H5" --record --no_walls "${HEADLESS_ARGS[@]}"
    --record_output "$NT_SL_MP4")
  echo "=== NTField plan (NOT RRT-supervised, e.g. straight-line dataset) -> ${NT_SL_MP4}"
  python new_setup.py "${NS_SL_ARGS[@]}"
fi

echo "Done. Videos:"
echo "  $ORIG_MP4"
echo "  $NT_MP4"
if [[ -n "${CKPT_STRAIGHTLINE}" ]]; then
  echo "  ${NT_SL_MP4}"
fi
if [[ -f "${SESSION_DIR}/episode_meta.txt" ]]; then
  echo "  ${SESSION_DIR}/episode_meta.txt"
fi
