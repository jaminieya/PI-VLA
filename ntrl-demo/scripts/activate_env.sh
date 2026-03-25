#!/bin/bash
# Source this before running when using arm config (torch_kdtree needs libpython).
# Usage: source scripts/activate_env.sh
# Or: export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
if [ -n "$CONDA_PREFIX" ]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
fi
