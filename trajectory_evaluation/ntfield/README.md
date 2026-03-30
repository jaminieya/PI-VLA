# NTField tooling (`trajectory_evaluation/ntfield`)

Scripts here share the **same Isaac Gym scene and flags** as `hanwen_grasping/new_setup.py` where applicable (`--no_walls`, cameras, UR5 asset layout). RRTConnect-based collection lives under `trajectory_evaluation/rrtconnect/`.

| File | Role |
|------|------|
| `collect_data.py` | Collect grasp HDF5; plans arm motion with **NTField** (`--ntfield_checkpoint` required). |
| `eval_trajectory_ntfield.py` | Held-out **τ** metrics + gradient planner stats on `points.npy` / `tau_obs.npy`. |
| `run_isaac_ntfield_demo.sh` | Replay **logged** joint trajectory vs **NTField plan** in Isaac; writes `original.mp4` / `ntfield.mp4`. |

## Train (straight-line dataset, no RRT labels)

Generate data (FCL collision-checked joint segments; no OMPL planning step):

```bash
python ntrl-demo/dataprocessing/generate_straightline_collision_dataset.py \
  --output_dir ntrl-demo/datasets/arm/UR5_straightline \
  --num_pairs 100000
```

Train NTField:

```bash
cd ntrl-demo && python train/train_arm_trajectory.py --data_path ./datasets/arm/UR5_straightline
```

Use the resulting `Model_Epoch_*.pt` with `collect_data.py`, `eval_trajectory_ntfield.py`, and `new_setup.py --ntfield` (via the demo script).

## Visualize in the same setup as `new_setup.py`

**Yes, with one caveat.**

- **`run_isaac_ntfield_demo.sh`** runs `hanwen_grasping/run_collected_trajectory.py` and **`hanwen_grasping/new_setup.py --ntfield`** from the `hanwen_grasping` directory. That **is** the same entry points as an interactive `new_setup.py` session: same default scene style, `--no_walls`, and recording path.
- Your **checkpoint must match the scenario** the field was trained for (e.g. no-wall training + `--no_walls` at test). A model trained only on straight-line/clearance in a **minimal table** scene may not match cluttered `new_setup` with many obstacles unless you train with matching geometry.

**Ways to visualize**

1. After `ntfield/collect_data.py`, the demo script runs automatically (unless `--no_run_ntfield_demo`).
2. Manually:  
   `bash trajectory_evaluation/ntfield/run_isaac_ntfield_demo.sh /path/to/demo.h5 /path/to/Model_Epoch_*.pt`
3. Directly: from `hanwen_grasping/`, run `new_setup.py --ntfield --checkpoint ... --h5_path ...` (same as the shell script).

Use **`HEADLESS=1`** for batch recording without a viewer.

## Quick commands (from PI-VLA root)

```bash
python trajectory_evaluation/ntfield/eval_trajectory_ntfield.py \
  --checkpoint ntrl-demo/Experiments/.../Model_Epoch_*.pt \
  --data_path ntrl-demo/datasets/arm/UR5_straightline

python trajectory_evaluation/ntfield/collect_data.py \
  --ntfield_checkpoint ntrl-demo/Experiments/.../Model_Epoch_*.pt
```
