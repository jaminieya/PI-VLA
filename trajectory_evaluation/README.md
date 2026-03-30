# Trajectory NTField evaluation

Evaluate checkpoints produced by `train_arm_trajectory.py` on trajectory data (`points.npy`, `tau_obs.npy`).

## Usage

From the **PI-VLA** repository root (with conda env `ntrl-demo` or equivalent):

```bash
python trajectory_evaluation/eval_trajectory_ntfield.py \
  --checkpoint ntrl-demo/Experiments/UR5_trajectory/trajectory_MM_DD_HH_MM/Model_Epoch_00500_ValLoss_*.pt \
  --data_path ntrl-demo/datasets/arm/UR5_trajectory_vertical_train \
  --device cuda:0
```

`--data_path` must be the folder that actually contains **`points.npy`** and **`tau_obs.npy`** (whatever name you used with `prepare_trajectory_dataset.py --output_dir`). If those files are missing, create them with `prepare_trajectory_dataset.py` or point `--data_path` at the correct subdirectory; the eval script error message lists sibling folders that already contain both files.

Options:

- `--val_ratio` — fraction of data for held-out metrics (default `0.2`).
- `--max_plan_samples` — cap planner runs for speed (default `200`).
- `--skip_planner` — only report τ RMSE / MAE / relative error.
- `--step_size`, `--max_steps`, `--tol` — passed to `planning/gradient_planner_trajectory.plan`.
- `--goal_success_eps_rad` — success threshold in radians (default `tol * SCALE` with `SCALE = π/0.5`).

## Example: `eval_trajectory_ntfield.py` output

Command (from repo root, example checkpoint and dataset on `corallabs1`):

```bash
python trajectory_evaluation/eval_trajectory_ntfield.py \
  --checkpoint ntrl-demo/Experiments/UR5_trajectory_no_wall_accuracy_check/trajectory_03_25_20_28/Model_Epoch_05000_ValLoss_7.820605e-01.pt \
  --data_path ntrl-demo/datasets/arm/UR5_trajectory_vertical_train \
  --device cuda:0
```

Captured console log (same run; minor TTY line-wrap artifacts removed):

```text
torch.Size([128, 6])
Dataset: n=100000, val holdout: 20000 (val_ratio=0.2, seed=42)
Checkpoint: /media/corallab-s1/4tbhdd/junheelim/PI-VLA/ntrl-demo/Experiments/UR5_trajectory_no_wall_accuracy_check/trajectory_03_25_20_28/Model_Epoch_05000_ValLoss_7.820605e-01.pt
Device: cuda:0
/home/junheelim/.conda/envs/grasping_py38/lib/python3.8/site-packages/torch/nn/modules/instancenorm.py:88: UserWarning: input's size at dim=0 does not match num_features. You can silence this warning by not passing in num_features, which is not used because affine=False
  warnings.warn(f"input's size at dim={feature_dim} does not match num_features. "

--- Field metrics (held-out val) ---
  tau RMSE: 8.349088e-01
  tau MAE:  7.291639e-01
  mean |rel err|: 7.119177e-01
/home/junheelim/.conda/envs/grasping_py38/lib/python3.8/site-packages/torch/nn/modules/instancenorm.py:88: UserWarning: input's size at dim=0 does not match num_features. You can silence this warning by not passing in num_features, which is not used because affine=False
  warnings.warn(f"input's size at dim={feature_dim} does not match num_features. "

--- Planner metrics (n=200, goal_eps_rad=0.062832) ---
  success rate: 200/200 (100.0%)
  mean final L2 error (rad):   0.049828
  median final L2 error (rad): 0.052955
  mean final Linf error (rad): 0.035346
  mean steps (before appended goal): 16.61
  mean path length (rad, sum of segments): 1.119688
```

The **`InstanceNorm1d` UserWarning** lines are expected when the network runs with batch size 1 during planning; they do not invalidate the printed metrics.

## Isaac Gym demo (recorded MP4s)

Compare **logged joint trajectory** vs **NTField gradient plan** on the same episode (`q_start` = first `joint_configs`, `q_goal` = `final_joint_config` if present, else last row). Videos are written under **`output/trajectory_evaluation/`**.

```bash
# Defaults: first test H5 + no-wall checkpoint; override with two args: H5_PATH CKPT_PATH
bash trajectory_evaluation/run_isaac_ntfield_demo.sh
```

Manual equivalent (from `hanwen_grasping/`):

```bash
mkdir -p ../output/trajectory_evaluation
python run_collected_trajectory.py \
  --h5_path ../output/data_collection/test/grasp_6dof_demo_20260315_220503.h5 \
  --record --record_output ../output/trajectory_evaluation/original_demo.mp4 \
  --interpolate 4 --no_walls

python new_setup.py --ntfield \
  --checkpoint ../ntrl-demo/Experiments/.../Model_Epoch_05000_ValLoss_....pt \
  --h5_path ../output/data_collection/test/grasp_6dof_demo_20260315_220503.h5 \
  --record --record_output ../output/trajectory_evaluation/ntfield_demo.mp4 \
  --no_walls
```

Use **`--no_walls`** for both so the table scene matches typical **no-wall** training checkpoints. **`imageio[ffmpeg]`** or **OpenCV** is needed to write MP4s.

**Interactive (default):** opens the Isaac viewer; uses **`DISPLAY`** / X11.

**Headless / no window:** pass **`--headless`** to both Python scripts, or run the demo with:

```bash
HEADLESS=1 bash trajectory_evaluation/run_isaac_ntfield_demo.sh
```

That skips `create_viewer` and runs a **fixed number of physics frames** (replay length + a short hold at the goal), then saves the video. Isaac still runs **`step_graphics`** and camera sensors; on some Linux servers you may need **`EGL` / OSMesa** (or a virtual framebuffer) for GPU rendering without X11—if frames are black or the process fails, check Isaac Gym docs for your GPU/driver setup.

**UR5 assets:** Isaac must load `hanwen_grasping/assets/urdf/ur5e/ur5e_mimic_real_gripper_test.urdf` and its **mesh** files. If the URDF is missing, truncated, or meshes are absent, Isaac logs *Failed to parse/import URDF* and the viewer may show **only the table (white box)** with no arm—that is not a planning bug; restore the full **`assets/ur5e/`** tree (e.g. from the machine where you collected data, or git-lfs if assets are tracked that way). `run_collected_trajectory.py` now checks this **before** `load_asset` where possible.

---

# Trajectory NTField only — accuracy evaluation (plan)

**Scope:** Models trained with [`ntrl-demo/train/train_arm_trajectory.py`](../ntrl-demo/train/train_arm_trajectory.py) (dataset: `points.npy` + `tau_obs.npy`). **Not in scope:** `train_arm.py` / `metric_arm`, [`ntrl-demo/tests/arm_plan_stat.py`](../ntrl-demo/tests/arm_plan_stat.py) MPPI pipeline, `run_arm.sh`, grasping stack.

If you instead trained with [`ntrl-demo/trajectory_collision/train_arm_trajectory.py`](../ntrl-demo/trajectory_collision/train_arm_trajectory.py), loading and losses differ slightly (`trajectory_collision.model_function_metric`), but the **same evaluation ideas** apply; optional collision/clearance metrics become relevant.

---

## What the model is

A **travel-time field** τ(q_start, q_goal) over 12D concatenated joints (6+6), with gradients used for planning. Training supervision pairs each sample with **τ_obs** from [`ntrl-demo/dataprocessing/prepare_trajectory_dataset.py`](../ntrl-demo/dataprocessing/prepare_trajectory_dataset.py) output.

At test time, **planning** is [`ntrl-demo/planning/gradient_planner_trajectory.plan`](../ntrl-demo/planning/gradient_planner_trajectory.py)(model, q_start, q_goal, …): integrate **q_start** along `model.function.Gradient(XP)` until within **tol** in **normalized** space (`SCALE = π/0.5`).

Evaluation stacks three optional layers: fit of τ to supervision, then whether gradient planning reaches **q_goal**, then (if you execute on hardware/sim) tracking.

```mermaid
flowchart LR
  subgraph layerA [Layer_A_field_fit]
    tauEval["tau_pred vs tau_obs RMSE_MAE"]
  end
  subgraph layerB [Layer_B_planning]
    planEval["q_final vs q_g success path_metrics"]
  end
  subgraph layerC [Layer_C_execution_optional]
    execEval["q_actual vs q_planned"]
  end
  tauEval --> planEval
  planEval --> execEval
```

**Data flow (conceptual):** `points.npy` rows `(q_start, q_goal)` and `tau_obs.npy` feed **Layer A**; the same pairs (or held-out test split) feed `plan()` for **Layer B**; **Layer C** only applies if you run the planned trajectory on a sim or robot.

---

## Layer A — Field accuracy (direct model fit)

**Data:** Rows of `points.npy` (first 12 = q_start ∥ q_goal in the same layout as training) and `tau_obs.npy`.

- Hold out a fraction of indices (or a separate directory of test `.npy`).
- Forward each 12D point through the trained network (same normalization as training in `train_arm_trajectory.py`).
- Report **RMSE**, **MAE**, and optionally **relative error** on τ vs τ_obs.

This measures “does τ match the RRT / sampling supervision?”—independent of whether gradient planning reaches q_g.

---

## Layer B — Planning accuracy (given q_g)

For each test pair (q_start, q_goal):

1. Run `plan(...)` with fixed hyperparameters (step_size, max_steps, tol).
2. **Success:** final configuration within ε of q_goal — use **radians** for reporting; note `plan()`’s `tol` is in **normalized** joint space.
3. **Goal error:** L2 and L∞ over 6 joints (rad and optionally deg).
4. **Efficiency:** iteration count at termination, path length ∑‖Δq‖ in joint space (or normalized space for comparability).

Optional: compare planned path length or shape to a **reference** trajectory if you store one; not required for a minimal eval script.

---

## Layer C — Execution (optional, only if you command a sim/robot)

If the arm physically moves along the planned path: **tracking error** q_actual vs q_planned and **task success** at the end. This is outside the NTField code path but can be a separate metric.

---

## Implementation sketch (trajectory-only)

- Reuse the same **model build + `Function` + checkpoint load** pattern as [`train_arm_trajectory.py`](../ntrl-demo/train/train_arm_trajectory.py) (lines loading `DatabaseTrajectory`, instantiating `model_network.NN` and `model_function.Function`, and saving/loading `.pt` — load the inverse of save).
- **Field eval:** batch forward on test tensors, compare to `tau_obs`.
- **Planner eval:** loop `(q_start, q_g)` from test rows or sampled goals, call `plan()`, aggregate statistics.

No need to integrate `arm_plan_stat.py` or MPPI for this scope.

---

## Primary recommendation

1. **Primary:** planning **success rate** + **median final ‖q − q_g‖₂** over N held-out (q_start, q_goal) pairs.
2. **Secondary:** **τ RMSE** on the same or a separate held-out τ dataset.
3. **Collision/clearance:** only if you trained with `trajectory_collision/` and care about obstacles.

---

## Checklist (from original plan todos)

- [ ] Mirror `train_arm_trajectory.py`: load `NN`, `Function`, restore `.pt` from `Experiments/.../trajectory_*`.
- [ ] Held-out τ: RMSE/MAE between `TravelTimes` and `tau_obs`.
- [ ] For `(q_start, q_g)`, run `gradient_planner_trajectory.plan`; log success, final L2/Linf vs q_g, steps, path length.
- [ ] Only if trained with `trajectory_collision/`: add clearance / collision metrics.
