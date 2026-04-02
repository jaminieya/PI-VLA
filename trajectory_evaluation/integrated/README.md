# Integrated triple planning comparison

**`run_triple_plan_compare.py`** builds **one** Isaac scene (same object layout as `rrtconnect/collect_data.py`), picks a grasp with **RRTConnect**, snapshots **`q_start_live`** (arm joints before any comparison motion) and **`grasp_target_q`** (grasp IK), then on that fixed scene runs:

1. **RRT** — replay normalized RRTConnect path → **`rrt.mp4`**
2. **RRT-trajectory NTField** — gradient plan from `q_start_live` to `grasp_target_q` → **`rrt_ntfield.mp4`**
3. **Non–RRT-expert NTField** (e.g. straight-line-trained checkpoint) — same start/goal → **`ntfield.mp4`**

The arm is **reset** toward `q_start_live` between passes (objects are not respawned; small object drift from physics may still occur).

Also writes **`scene_snapshot.json`** (`q_start_live`, `grasp_target_q`, `object_location`, checkpoint paths, video paths). Unless **`--skip_hdf5`**, saves **`grasp_6dof_demo_*.h5`** whose **`joint_configs`** come from **one** RRT replay (same as pass 1), suitable for off-line tools.

## Run (from PI-VLA root)

```bash
python trajectory_evaluation/integrated/run_triple_plan_compare.py \
  --ntfield_checkpoint_rrt_trajectory ntrl-demo/Experiments/.../RRT_trajectory_Model.pt \
  --ntfield_checkpoint_straightline ntrl-demo/Experiments/.../straightline_Model.pt
```

Optional: `--use_viewer`, `--skip_hdf5`, `--ntfield_device cpu`, `--ntfield_step_size`, etc.

Outputs go under **`output/trajectory_evaluation/YYYYMMDD_HHMMSS/`**.
