#!/bin/bash
# Build trac_ik from source in a catkin workspace (no sudo required)
# Uses: system ROS Noetic + conda grasping_py38 for Python 3.8 and build deps
#
# Run with: conda activate grasping_py38 && bash build_trac_ik_catkin.sh
#
# After building, source the workspace before running new_setup.py:
#   source /opt/ros/noetic/setup.bash
#   source /media/corallab-s1/4tbhdd/junheelim/config/trac_ik_catkin_ws/devel/setup.bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$(cd "$SCRIPT_DIR/../../config" && pwd)"
CATKIN_WS="$CONFIG_DIR/trac_ik_catkin_ws"

if [ -z "$CONDA_PREFIX" ]; then
    echo "ERROR: Activate grasping_py38 first:"
    echo "  conda activate grasping_py38"
    exit 1
fi

if [ ! -f /opt/ros/noetic/setup.bash ]; then
    echo "ERROR: ROS Noetic not found. Need /opt/ros/noetic/setup.bash"
    exit 1
fi

# Install Python deps for catkin build (if missing)
python -c "import empy" 2>/dev/null || python -m pip install empy
python -c "import catkin_pkg" 2>/dev/null || python -m pip install catkin_pkg rospkg

echo "=== Building trac_ik from source (catkin) ==="

mkdir -p "$CATKIN_WS/src"
if [ ! -d "$CATKIN_WS/src/trac_ik" ]; then
    echo "Cloning trac_ik (master/ROS 1)..."
    git clone --branch master --depth 1 https://bitbucket.org/traclabs/trac_ik.git "$CATKIN_WS/src/trac_ik"
fi

# Build with conda deps (nlopt, eigen from conda)
cd "$CATKIN_WS"
rm -rf build devel

export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
source /opt/ros/noetic/setup.bash
export CMAKE_PREFIX_PATH=/opt/ros/noetic:$CONDA_PREFIX

catkin_make --only-pkg-with-deps trac_ik_python

echo ""
echo "=== trac_ik build complete ==="
echo ""
echo "Before running new_setup.py:"
echo "  source /opt/ros/noetic/setup.bash"
echo "  source $CATKIN_WS/devel/setup.bash"
echo "  conda activate grasping_py38"
echo "  export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH"
echo "  python new_setup.py"
echo ""
