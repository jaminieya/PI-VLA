#!/bin/bash
# Setup script for grasping_py38 environment
# Run after: conda activate grasping_py38

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$(cd "$SCRIPT_DIR/../../config" && pwd)"

echo "=== Setting up grasping_py38 ==="

# 1. Create library symlinks (OMPL custom bindings expect specific versions)
for lib in libode.so.0.16.2 libode.so.0.16.6; do
    if [ -f "$CONDA_PREFIX/lib/$lib" ] && [ ! -f "$CONDA_PREFIX/lib/libode.so.8" ]; then
        ln -sf "$CONDA_PREFIX/lib/$lib" "$CONDA_PREFIX/lib/libode.so.8"
        echo "Created libode.so.8 symlink"
        break
    fi
done

# libboost_python38.so.1.71.0 (OMPL custom bindings) -> conda's version
for lib in libboost_python38.so.1.90.0 libboost_python38.so.1.74.0; do
    if [ -f "$CONDA_PREFIX/lib/$lib" ] && [ ! -f "$CONDA_PREFIX/lib/libboost_python38.so.1.71.0" ]; then
        ln -sf "$CONDA_PREFIX/lib/$lib" "$CONDA_PREFIX/lib/libboost_python38.so.1.71.0"
        echo "Created libboost_python38.so.1.71.0 symlink"
        break
    fi
done

# libboost_serialization.so.1.71.0 (OMPL custom bindings)
for lib in libboost_serialization.so.1.74.0 libboost_serialization.so.1.90.0 libboost_serialization.so.1.82.0; do
    if [ -f "$CONDA_PREFIX/lib/$lib" ] && [ ! -L "$CONDA_PREFIX/lib/libboost_serialization.so.1.71.0" ]; then
        ln -sf "$CONDA_PREFIX/lib/$lib" "$CONDA_PREFIX/lib/libboost_serialization.so.1.71.0"
        echo "Created libboost_serialization.so.1.71.0 symlink"
        break
    fi
done

# 2. Install Isaac Gym (use python -m pip to avoid picking up ~/.local pip)
if [ -d "$CONFIG_DIR/isaacgym/python" ]; then
    python -m pip install -e "$CONFIG_DIR/isaacgym/python"
    echo "Installed Isaac Gym"
else
    echo "WARNING: Isaac Gym not found at $CONFIG_DIR/isaacgym"
fi

# 3. trac_ik: run setup_ros_trac_ik.sh first (builds from source via catkin)
if ! (source /opt/ros/noetic/setup.bash 2>/dev/null && source "$CONFIG_DIR/trac_ik_catkin_ws/devel/setup.bash" 2>/dev/null && python -c "from trac_ik_python.trac_ik import IK" 2>/dev/null); then
    if ! (source /opt/ros/noetic/setup.bash 2>/dev/null && python -c "from trac_ik_python.trac_ik import IK" 2>/dev/null); then
        echo "WARNING: trac_ik not found. Run setup_ros_trac_ik.sh first:"
        echo "  conda activate grasping_py38"
        echo "  bash setup_ros_trac_ik.sh"
    fi
fi

# 4. Update robot_arm_configuration to use conda OMPL (no custom path for py38)
# The conda ompl may not have Python bindings - keep custom path as fallback
# Custom OMPL needs libboost_python38 - conda provides it for py38

echo ""
echo "=== Setup complete ==="
echo "Run with:"
echo "  source /opt/ros/noetic/setup.bash"
if [ -f "$CONFIG_DIR/trac_ik_catkin_ws/devel/setup.bash" ]; then
    echo "  source $CONFIG_DIR/trac_ik_catkin_ws/devel/setup.bash"
fi
echo "  conda activate grasping_py38"
echo "  export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH"
echo "  python new_setup.py"
echo ""
