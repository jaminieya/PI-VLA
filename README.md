# PI-VLA (Quick Guide)

This repository combines NTField training, grasping simulation, goal-latent student modeling, and evaluation utilities.

## Main folders

- `final_integrate/`  
  End-to-end integration runners for combined pipeline execution (`run_integrated_pipeline*.py`).

- `goal_embedding_visualization/`  
  Tools for extracting teacher goal embeddings from HDF5 and visualizing them (for example with 3D PCA).

- `hanwen_grasping/`  
  Isaac Gym grasping environment and data/training utilities.  
  Includes data collection scripts, simulation setup, and goal-representation training scripts.

- `ntrl-demo/`  
  Core NTField codebase: preprocessing, model training, planning, and experiment outputs.  
  Includes UR5 trajectory training and baseline planning components.

- `student_model_evaluation/`  
  Student goal-latent inference/evaluation tools and scripts that connect student predictions with NTField planning.

- `trajectory_evaluation/`  
  Trajectory-level evaluation and benchmarking utilities (NTField and RRTConnect variants), including demo/collection scripts.

## Minimal workflow (from `PI-VLA` root)

1. Train or prepare NTField checkpoints in `ntrl-demo/`.
2. Train/evaluate goal-representation student models in `hanwen_grasping/` and `student_model_evaluation/`.
3. Run trajectory metrics/benchmarks in `trajectory_evaluation/`.
4. Use `final_integrate/` scripts for integrated pipeline runs.
5. Use `goal_embedding_visualization/` to inspect embedding quality.

## `ntrl-demo`: trajectory-based NTField training (RRTConnect data)

This is the pipeline for training a trajectory NTField from collected RRTConnect trajectories and their interpolated waypoints.

1. **Collect trajectory episodes** (HDF5) with RRTConnect trajectories in `joint_configs`.
2. **Convert trajectories to NTField supervision** using `ntrl-demo/dataprocessing/prepare_trajectory_dataset.py`.
   - The script loads each trajectory, samples `(q_start, q_goal)` pairs from within trajectories, and computes `tau_obs` as trajectory path length between sampled waypoints.
   - Output files:
     - `points.npy` with shape `(N, 12)` where each row is `[q_start(6), q_goal(6)]`
     - `tau_obs.npy` with shape `(N,)`
3. **Train trajectory NTField** with `ntrl-demo/train/train_arm_trajectory.py` using those `.npy` files.

Example (from `PI-VLA` root):

```bash
# 1) Build trajectory dataset from collected RRTConnect episodes
python ntrl-demo/dataprocessing/prepare_trajectory_dataset.py \
  --data_dir collected_data \
  --output_dir ntrl-demo/datasets/arm/UR5_trajectory \
  --num_pairs 100000 \
  --tau_min 0.01 \
  --tau_max 2.0

# 2) Train NTField on the generated trajectory dataset
python ntrl-demo/train/train_arm_trajectory.py \
  --data_path ntrl-demo/datasets/arm/UR5_trajectory \
  --device cuda:0
```

Checkpoint output pattern:

`ntrl-demo/Experiments/UR5_trajectory/trajectory_MM_DD_HH_MM/Model_Epoch_XXXXX_ValLoss_*.pt`

## `trajectory_evaluation/` structure

```text
trajectory_evaluation/
├── collect_data_ntfield.py
├── ntfield/
│   └── eval_trajectory_ntfield.py
├── rrtconnect/
│   ├── README.md
│   ├── collect_data.py
│   ├── collect_data_ntfield_hdf5.py
│   └── run_isaac_ntfield_demo.sh
└── comparison/
    ├── run_rrt_ntfield_benchmark.py
    └── run_rrt_ntfield_benchmark_batch.py
```

- `ntfield/`: evaluates trajectory NTField checkpoints on trajectory datasets (`points.npy`, `tau_obs.npy`).
- `rrtconnect/`: data collection and replay/demo utilities for RRTConnect vs NTField trajectories.
- `comparison/`: direct RRTConnect-vs-NTField benchmark scripts; batch runner sweeps a fixed `(x, y)` object grid.

## Running `comparison/run_rrt_ntfield_benchmark_batch.py`

From `PI-VLA` root:

```bash
python trajectory_evaluation/comparison/run_rrt_ntfield_benchmark_batch.py \
  --checkpoint ntrl-demo/Experiments/UR5_trajectory/trajectory_MM_DD_HH_MM/Model_Epoch_XXXXX_ValLoss_*.pt
```

Useful options:

- `--planner-playback {direct,settle}`: waypoint playback style (`settle` dwells multiple sim steps per waypoint).
- `--ntfield-waypoint-mode {full,two_point}`: forward waypoint mode to each benchmark run.
- `--ntfield-fixed-waypoints N`: resample to a fixed waypoint count (`0` disables resampling).
- `--object-z Z`: z value while `(x, y)` is swept over the predefined grid.
- `--out-root PATH`: output folder (default is `output/trajectory_evaluation/batch_<timestamp>/`).
- `--dry-run`: print all generated commands without executing.

Pass additional benchmark args after `--`:

```bash
python trajectory_evaluation/comparison/run_rrt_ntfield_benchmark_batch.py \
  --checkpoint ntrl-demo/Experiments/UR5_trajectory/trajectory_MM_DD_HH_MM/Model_Epoch_XXXXX_ValLoss_*.pt \
  --planner-playback settle \
  -- --sim_device cuda:0 --no_video
```

Each grid run writes into a subdirectory like:

`output/trajectory_evaluation/batch_<timestamp>/run_00_x0.5000_y0.3000/`

and the batch summary is saved to:

`output/trajectory_evaluation/batch_<timestamp>/batch_summary.json`

## Notes

- Each subfolder may have its own environment/dependency assumptions.
- Start with folder-level scripts/READMEs for command details.
