# Student model & evaluation

This directory holds **goal-latent student** tooling and **trajectory / NTField evaluation** code that used to live under `hanwen_grasping/` and `trajectory_evaluation/`.

## Layout

| Path | Role |
|------|------|
| `evaluate_goal_rep_alignment.py` | Offline eval: student `z_hat` vs teacher `z_goal` on HDF5 demos |
| `infer_goal_rep_latent.py` | Single-image + prompt + `q_start` → `z_goal_hat` |
| `plan_ntfield_with_student.py` | Isaac Gym: NTField planning driven by student-predicted goal latent |
| `trajectory_evaluation/` | NTField trajectory metrics, Isaac collection, `run_isaac_ntfield_demo.sh`, RRTConnect variant |

**Training** the student is still in `../hanwen_grasping/train_goal_rep_alignment.py` (and `run_goal_rep_training_nohup.sh`).

## Examples (from `PI-VLA` root)

```bash
# Eval student (adds hanwen_grasping to path automatically)
python student_model_evaluation/evaluate_goal_rep_alignment.py \
  --student hanwen_grasping/checkpoints/goal_rep_student.pt \
  --h5_glob "collected_data/grasp_6dof_demo_*.h5"

# NTField trajectory eval (dataset metrics / planning)
python student_model_evaluation/trajectory_evaluation/ntfield/eval_trajectory_ntfield.py \
  --checkpoint ntrl-demo/Experiments/UR5_trajectory/.../Model_Epoch_*.pt \
  --data_path ntrl-demo/datasets/arm/UR5_trajectory

# Isaac replay + NTField demo videos
bash student_model_evaluation/trajectory_evaluation/ntfield/run_isaac_ntfield_demo.sh /path/to/demo.h5 /path/to/Model_Epoch_*.pt
```

## Imports

Python modules under `trajectory_evaluation/` are imported as `trajectory_evaluation.ntfield.*`. That requires **`student_model_evaluation`** on `PYTHONPATH`, or run scripts that insert it (e.g. `ntfield/collect_data.py`).
