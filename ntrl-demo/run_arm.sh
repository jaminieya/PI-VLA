#!/bin/bash
# Run arm (UR5) preprocessing, training, and testing with visualization.
# Requires: conda activate ntrl-demo, and for preprocess: source scripts/activate_env.sh
#
# Usage:
#   ./run_arm.sh preprocess   # Preprocess arm data (needs LD_LIBRARY_PATH)
#   ./run_arm.sh train        # Train arm model
#   ./run_arm.sh test [ckpt]  # Test with visualization (ckpt: checkpoint path, or latest)
#   ./run_arm.sh all          # preprocess + train + test
#   ./run_arm.sh all --no-viz # all but skip visualization (headless test)

set -e
cd "$(dirname "$0")"

MODEL_PATH="./Experiments/UR5"
DATA_PATH="./datasets/arm/UR5"
CONFIG="configs/arm.txt"

# Set LD_LIBRARY_PATH for torch_kdtree (arm preprocess)
export_ld_path() {
    if [ -n "$CONDA_PREFIX" ]; then
        export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
    fi
}

find_latest_checkpoint() {
    local latest=""
    local latest_epoch=0
    # Training creates arm_MM_DD_HH_MM (from DataPath datasets/arm/UR5)
    for d in "$MODEL_PATH"/*/; do
        [ -d "$d" ] || continue
        for f in "$d"Model_Epoch_*.pt; do
            [ -f "$f" ] || continue
            epoch=$(echo "$f" | sed -n 's/.*Epoch_\([0-9]*\).*/\1/p')
            if [ -n "$epoch" ] && [ "$epoch" -gt "$latest_epoch" ] 2>/dev/null; then
                latest_epoch=$epoch
                latest=$f
            fi
        done
    done
    echo "$latest"
}

cmd_preprocess() {
    echo "=== Preprocessing arm data ==="
    export_ld_path
    python dataprocessing/preprocess.py --config "$CONFIG"
}

cmd_train() {
    # Default: Agg (headless) - works without display. Use --display for X11.
    local use_display=false
    [ "$2" = "--display" ] && use_display=true
    if [ "$use_display" = false ]; then
        export MPLBACKEND=Agg
        echo "=== Training arm model (headless) ==="
    else
        unset MPLBACKEND
        echo "=== Training arm model (X11 display) ==="
    fi
    python train/train_arm.py
}

cmd_test() {
    local ckpt="$2"
    local save_path=""
    local use_display=false
    if [ "$2" = "--save" ]; then
        ckpt=""
        save_path="${3:-arm_path.html}"
    elif [ "$2" = "--display" ]; then
        use_display=true
    elif [ "$3" = "--save" ]; then
        save_path="${4:-arm_path.html}"
    elif [ "$3" = "--display" ]; then
        use_display=true
    fi
    [ -z "$ckpt" ] && ckpt=$(find_latest_checkpoint)
    if [ -z "$ckpt" ]; then
        echo "Error: No checkpoint found. Train first with: ./run_arm.sh train"
        exit 1
    fi
    if [ ! -f "$ckpt" ]; then
        echo "Error: Checkpoint not found: $ckpt"
        exit 1
    fi
    # Default: save to HTML (works without display). Use --display to try opening window.
    [ "$use_display" = false ] && [ -z "$save_path" ] && save_path="arm_path.html"
    echo "=== Testing with visualization ==="
    if [ -n "$save_path" ]; then
        python tests/arm_plan_stat.py --checkpoint "$ckpt" --save "$save_path"
    else
        python tests/arm_plan_stat.py --checkpoint "$ckpt"
    fi
}

cmd_all() {
    local skip_viz=false
    local use_display=false
    for arg in "$@"; do
        [ "$arg" = "--no-viz" ] && skip_viz=true
        [ "$arg" = "--display" ] && use_display=true
    done
    cmd_preprocess
    if [ "$use_display" = true ]; then
        cmd_train "" "--display"
    else
        cmd_train
    fi
    if [ "$skip_viz" = false ]; then
        cmd_test
    else
        echo "Skipping visualization (--no-viz)"
    fi
}

# Install trimesh if missing (for visualization)
install_deps() {
    echo "=== Installing dependencies (trimesh, pyglet<2, etc.) ==="
    pip install trimesh "pyglet<2" libigl pytorch_kinematics -q
}

case "${1:-}" in
    preprocess) cmd_preprocess ;;
    train)      cmd_train "" "$2" ;;
    test)       cmd_test "$2" "$3" ;;
    all)        cmd_all "$@" ;;
    install)    install_deps ;;
    *)
        echo "Usage: $0 {preprocess|train|test|all|install} [options]"
        echo ""
        echo "  preprocess       Preprocess arm data (datasets/arm/UR5)"
        echo "  train [--display]   Train UR5 model (default: headless; use --display for X11)"
        echo "  test [ckpt] [--display]  Test (default: saves to arm_path.html); use --display for window"
        echo "  all [--no-viz] [--display]  Run preprocess + train + test"
        echo "  install         Install trimesh, libigl, pytorch_kinematics"
        echo ""
        echo "Example: $0 all"
        exit 1
        ;;
esac
