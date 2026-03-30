#!/usr/bin/env bash
# Delegates to canonical script under trajectory_evaluation/ntfield/.
set -euo pipefail
exec "$(cd "$(dirname "$0")" && pwd)/../ntfield/run_isaac_ntfield_demo.sh" "$@"
