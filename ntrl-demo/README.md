## About
This is a minimal example. 

## Setup (Conda)
1. `conda env create -f environment.yml`
2. `conda activate ntrl-demo`
3. For **gibson** config: `python dataprocessing/preprocess.py --config configs/gibson.txt`
4. For **arm** config: `source scripts/activate_env.sh` (sets LD_LIBRARY_PATH for torch_kdtree), then run preprocess
5. `python train/train_gib.py` to start training

## Arm (UR5) workflow
```bash
# Install visualization deps (trimesh, libigl, pytorch_kinematics)
./run_arm.sh install

# Full pipeline: preprocess → train → test with visualization
./run_arm.sh all

# Or step by step:
./run_arm.sh preprocess   # Preprocess datasets/arm/UR5
./run_arm.sh train       # Train model (headless, no display needed)
./run_arm.sh train --display   # Train with X11 display (ssh -X)
./run_arm.sh test        # Test with trimesh visualization (uses latest checkpoint)
./run_arm.sh test /path/to/Model_Epoch_XXXXX_ValLoss_*.pt  # Test with specific checkpoint
```

**Display:** Training defaults to headless (no display). Use `--display` with X11 forwarding (`ssh -X`) when you want the display backend.

## Setup (Docker, legacy)
1. git clone this repo
2. run `docker build -t ntrl:demo .` under the root directory of this repo, once you built the docker image, you don't need to build it again unless you change the dockerfile.
3. run `docker run -u $(id -u):$(id -g) --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --volume="/home/n/Eikonal_Planning/ntrl-demo:/workspace" --volume="/usr/lib/x86_64-linux-gnu/:/glu" --volume="/home/n/.local:/.local" --env="QT_X11_NO_MITSHM=1"  --gpus all -ti --rm ntrl:demo` to start the docker container.
4. run `pip install scipy` inside the container to install the KD-tree dependency
5. run `python dataprocessing/preprocess.py --config configs/gibson.txt ` to sample training data
6. run `python train/train_gib.py` to start the training.
