# Python 3.8 Environment for new_setup.py

This environment uses Python 3.8 to resolve the OMPL/libboost_python38 dependency.
Uses **trac_ik** (original ROS version): apt if installed, else built from source via catkin. No tracikpy.

**Requires:** ROS Noetic installed on system (`/opt/ros/noetic`)

## Quick Transfer (to another machine)

### 1. Copy these files to the target machine:
- `environment_py38.yml`
- `setup_ros_trac_ik.sh`
- `build_trac_ik_catkin.sh`
- `setup_grasping_py38.sh`
- `config/` (isaacgym)

### 2. Create the conda environment:
```bash
conda env create -f environment_py38.yml
conda activate grasping_py38
```

### 3. Install trac_ik (requires ROS Noetic):
```bash
cd /path/to/PI-VLA/hanwen_grasping
bash setup_ros_trac_ik.sh
```
Uses apt trac_ik if installed, else builds from source via catkin (no sudo).

### 4. Run setup script:
```bash
bash setup_grasping_py38.sh
```

### 5. Run new_setup.py:
```bash
source /opt/ros/noetic/setup.bash
source /path/to/config/trac_ik_catkin_ws/devel/setup.bash   # if catkin-built
conda activate grasping_py38
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python new_setup.py
```

## Optional: Export exact env (for identical transfer)

```bash
conda activate grasping_py38
conda env export > environment_py38_full.yml
# On target: conda env create -f environment_py38_full.yml
```

Note: Full export includes paths - use `conda env export --no-builds` for more portable export.
