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

## UR5 trajectory NTField (straight-line dataset)

Dataset generation (writes `points.npy` and `tau_obs.npy` only—no checkpoint) is documented in [`dataprocessing/README.md`](dataprocessing/README.md). Train **after** the dataset exists:

```bash
cd ntrl-demo   # PI-VLA/ntrl-demo

python train/train_arm_trajectory.py --data_path ./datasets/arm/UR5_straightline/
```

**Checkpoint location:** each run creates a timestamped folder under `Experiments/UR5_trajectory/`:

`Experiments/UR5_trajectory/trajectory_MM_DD_HH_MM/Model_Epoch_XXXXX_ValLoss_*.pt`

Example log (`python train/train_arm_trajectory.py --data_path ./datasets/arm/UR5_straightline/`):

```text
torch.Size([128, 6])
(100000, 12)
(100000,)
Training on 100000 samples, 50 batches/epoch
Output: /media/corallab-s1/4tbhdd/junheelim/PI-VLA/ntrl-demo/Experiments/UR5_trajectory/trajectory_03_30_12_17
Epoch 1 -- Loss = 1.9081e+00 -- TrajLoss = 1.5660e+00
Epoch 2 -- Loss = 1.5871e+00 -- TrajLoss = 1.3908e+00
Epoch 3 -- Loss = 1.5517e+00 -- TrajLoss = 1.3475e+00
Epoch 4 -- Loss = 1.4989e+00 -- TrajLoss = 1.2856e+00
Epoch 5 -- Loss = 1.4251e+00 -- TrajLoss = 1.2029e+00
Epoch 6 -- Loss = 1.3560e+00 -- TrajLoss = 1.1230e+00
Epoch 7 -- Loss = 1.2998e+00 -- TrajLoss = 1.0562e+00
Epoch 8 -- Loss = 1.2786e+00 -- TrajLoss = 1.0329e+00
Epoch 9 -- Loss = 1.2318e+00 -- TrajLoss = 9.7739e-01
```

Use that folder’s `Model_Epoch_*.pt` for `hanwen_grasping/new_setup.py --ntfield --checkpoint ...` or `trajectory_evaluation/ntfield/eval_trajectory_ntfield.py`.

### Evaluation example (straight-line trained run)

From **PI-VLA** root, after training on `datasets/arm/UR5_straightline`:

```bash
python trajectory_evaluation/ntfield/eval_trajectory_ntfield.py \
  --checkpoint ntrl-demo/Experiments/UR5_trajectory/trajectory_03_30_12_17/Model_Epoch_05000_ValLoss_2.422709e-01.pt \
  --data_path ntrl-demo/datasets/arm/UR5_straightline/ \
  --device cuda:0
```

**Captured output** (checkpoint epoch 5000, val loss at save `2.422709e-01`; default `val_ratio=0.2`, `max_plan_samples=200`):

```text
Dataset: n=100000, val holdout: 20000 (val_ratio=0.2, seed=42)
Checkpoint: .../Model_Epoch_05000_ValLoss_2.422709e-01.pt
Device: cuda:0

--- Field metrics (held-out val) ---
  tau RMSE: 3.671103e-01
  tau MAE:  3.176630e-01
  mean |rel err|: 1.887645e-01

--- Planner metrics (n=200, goal_eps_rad=0.062832) ---
  success rate: 134/200 (67.0%)
  mean final L2 error (rad):   0.880655
  median final L2 error (rad): 0.057527
  mean final Linf error (rad): 0.593707
  mean steps (before appended goal): 115.97
  mean path length (rad, sum of segments): 10.959216
```

**Interpretation**

- **Field metrics:** On **20k** validation samples, **τ** prediction matches labels with RMSE ≈ **0.37** rad and mean relative error ≈ **19%**. Labels here are straight-line joint-space lengths, so this is direct **regression quality** on the synthetic supervision, not Isaac execution.
- **Planner success (67%):** **134 / 200** runs reach ‖q_final − q_goal‖₂ below **0.0628** rad (`tol=0.01` in normalized space × `SCALE = π/0.5`). The rest miss under the default planner budget.
- **Mean vs median L2:** **Median** ≈ **0.058** rad is close to the success threshold; **mean** ≈ **0.88** rad is much larger, so a **heavy tail** of bad plans dominates the mean. Report **both** when comparing checkpoints; lowering the tail usually matters more than the median for manipulation.
- **Steps / path length:** Mean **~116** steps and **~11** rad summed segment length suggest many trajectories are **long or circuitous** versus a single straight joint step (τ often ≤ 2). That is consistent with gradient descent on an imperfect field and `max_steps=200`.

`InstanceNorm1d` warnings during planner evaluation (batch size 1) are expected and do not invalidate these metrics.

**Visualize in Isaac** (`hanwen_grasping`, interactive viewer):

```bash
cd /path/to/PI-VLA/hanwen_grasping
python new_setup.py --ntfield \
  --checkpoint ../ntrl-demo/Experiments/UR5_trajectory/trajectory_03_30_12_17/Model_Epoch_05000_ValLoss_2.422709e-01.pt \
  --h5_path ../output/data_collection/test/grasp_6dof_demo_20260315_220503.h5 \
  --no_walls
```

**Headless** (no `DISPLAY`; add **`--record`** or no MP4 is written). Parent directories for **`--record_output`** are created automatically (`new_setup.py` calls `os.makedirs`).

```bash
python new_setup.py --ntfield \
  --checkpoint ../ntrl-demo/Experiments/UR5_trajectory/trajectory_03_30_12_17/Model_Epoch_05000_ValLoss_2.422709e-01.pt \
  --h5_path ../output/data_collection/test/grasp_6dof_demo_20260315_220503.h5 \
  --no_walls --headless \
  --record --record_output ../output/trajectory_evaluation/03301217/ntfield_model.mp4
```

If saving the MP4 throws before a clean Isaac shutdown, you may see a **segmentation fault** on exit; fixing the path (or updating `new_setup.py` as above) avoids that.

The `InstanceNorm1d` batch-size warning during training is expected for small batches and is harmless for the printed losses.

**Note:** If you pass `--output_dir ntrl-demo/datasets/...` to the generator while the shell cwd is already `ntrl-demo`, checkpoints are unchanged—but data lands under `ntrl-demo/ntrl-demo/datasets/...`. Prefer `--data_path ./datasets/arm/UR5_straightline/` when data lives next to `train/`.

## Setup (Docker, legacy)
1. git clone this repo
2. run `docker build -t ntrl:demo .` under the root directory of this repo, once you built the docker image, you don't need to build it again unless you change the dockerfile.
3. run `docker run -u $(id -u):$(id -g) --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --volume="/home/n/Eikonal_Planning/ntrl-demo:/workspace" --volume="/usr/lib/x86_64-linux-gnu/:/glu" --volume="/home/n/.local:/.local" --env="QT_X11_NO_MITSHM=1"  --gpus all -ti --rm ntrl:demo` to start the docker container.
4. run `pip install scipy` inside the container to install the KD-tree dependency
5. run `python dataprocessing/preprocess.py --config configs/gibson.txt ` to sample training data
6. run `python train/train_gib.py` to start the training.
