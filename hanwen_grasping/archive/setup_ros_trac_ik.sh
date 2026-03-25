#!/bin/bash
# Setup trac_ik for grasping_py38 (original from ROS/catkin only, no tracikpy)
# Run with: conda activate grasping_py38 && bash setup_ros_trac_ik.sh
#
# Requires: ROS Noetic installed on system
# Uses: apt trac_ik if installed, else builds from source via catkin

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$(cd "$SCRIPT_DIR/../../config" && pwd)"

if [ -z "$CONDA_PREFIX" ]; then
    echo "ERROR: Activate grasping_py38 first:"
    echo "  conda activate grasping_py38"
    echo "  bash setup_ros_trac_ik.sh"
    exit 1
fi

if [ ! -f /opt/ros/noetic/setup.bash ]; then
    echo "ERROR: ROS Noetic required. Install at /opt/ros/noetic"
    echo "  (or ask cluster admin to install ros-noetic-desktop)"
    exit 1
fi

# Remove any tracikpy compatibility layer
rm -rf "$SCRIPT_DIR/trac_ik_python"

# Check if already working (with ROS + catkin workspace sourced)
if [ -f "$CONFIG_DIR/trac_ik_catkin_ws/devel/setup.bash" ]; then
    if (source /opt/ros/noetic/setup.bash && source "$CONFIG_DIR/trac_ik_catkin_ws/devel/setup.bash" && python -c "from trac_ik_python.trac_ik import IK" 2>/dev/null); then
        echo "trac_ik already available (catkin-built)."
        exit 0
    fi
fi
if (source /opt/ros/noetic/setup.bash && python -c "from trac_ik_python.trac_ik import IK" 2>/dev/null); then
    echo "trac_ik already available (apt)."
    exit 0
fi

echo "=== Setting up trac_ik (original ROS) ==="

# Try 1: apt-installed
if (source /opt/ros/noetic/setup.bash && python -c "from trac_ik_python.trac_ik import IK" 2>/dev/null); then
    echo ""
    echo "=== Using system ROS Noetic trac_ik (apt) ==="
    echo "Run with: source /opt/ros/noetic/setup.bash && conda activate grasping_py38"
    echo ""
    exit 0
fi

# Try 2: catkin-built
if [ -f "$CONFIG_DIR/trac_ik_catkin_ws/devel/setup.bash" ]; then
    if (source /opt/ros/noetic/setup.bash && source "$CONFIG_DIR/trac_ik_catkin_ws/devel/setup.bash" && python -c "from trac_ik_python.trac_ik import IK" 2>/dev/null); then
        echo ""
        echo "=== Using trac_ik (catkin-built) ==="
        echo "Run with: source /opt/ros/noetic/setup.bash && source $CONFIG_DIR/trac_ik_catkin_ws/devel/setup.bash && conda activate grasping_py38"
        echo ""
        exit 0
    fi
fi

# Build from source
echo "Building trac_ik from source..."
bash "$SCRIPT_DIR/build_trac_ik_catkin.sh"
