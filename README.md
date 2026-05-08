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

- `student_model_training/`  
  Student latent-goal training pipelines, preprocessing scripts for `.pt` shards, and train configs (MDN/regression).

- `trajectory_evaluation/`  
  Trajectory-level evaluation and benchmarking utilities (NTField and RRTConnect variants), including demo/collection scripts.

## Minimal workflow (from `PI-VLA` root)

1. Train or prepare NTField checkpoints in `ntrl-demo/`.
2. Train/evaluate goal-representation student models in `hanwen_grasping/` and `student_model_evaluation/`.
3. Run trajectory metrics/benchmarks in `trajectory_evaluation/`.
4. Use `final_integrate/` scripts for integrated pipeline runs.
5. Use `goal_embedding_visualization/` to inspect embedding quality.

## Student model + integration quick commands

### Student Model Training (Short)

Before training, prepare data in this order: collect multi-object data with `hanwen_grasping/collect_data/run_multi_obj_collect_10000.sh`, preprocess H5 outputs into `.pt` shards (for example with `student_model_training/preprocess_dataset/preprocess_grasp_multi3_to_pt_shards.py`), then run training scripts below.

Run from `PI-VLA` root:

```bash
cd student_model_training
bash train_config_multi_mdn.sh
```

- `train_config_multi_mdn.sh` launches `full_train_multi_mdn.py` in the `rlgpu` conda env.
- It saves logs/checkpoints under `output/runs/full_train_multi_mdn_<timestamp>/`.
- Default MDN config in script: `epochs=90`, `batch_size=256`, `lr=3e-4`, `n_components=8`.

For a regression baseline:

```bash
cd student_model_training
bash train_config_multi_regression.sh
```

- `train_config_multi_regression.sh` launches `full_train_multi_regression.py`.
- It saves logs/checkpoints under `output/runs/full_train_multi_regression_<timestamp>/`.
- Default regression config in script: `loss=hybrid`, `epochs=40`, `batch_size=256`, `lr=3e-4`.

### Integration Planning (Short)

Run from `PI-VLA` root (headless by default unless `--use_viewer` is set):

```bash
python final_integrate/run_integrated_pipeline_latent_multi_obj_mdn.py \
  --ntfield_checkpoint teacher_model.pt \
  --latent_checkpoint /path/to/best_z_goal_model_mdn_*.pth \
  --num_trials 1
```

- Uses top-view image -> student latent prediction -> NTField gradient planning.
- Outputs go to `output/final_integrate/<timestamp>/` unless `--output_dir` is provided.

For detailed sanity-check/evaluation run:

```bash
python final_integrate/run_integrated_pipeline_latent_multi_obj_check.py \
  --ntfield_checkpoint teacher_model.pt \
  --latent_checkpoint /path/to/best_z_goal_model_mdn_*.pth \
  --output_dir output/manual_check_run \
  --seed 1002
```

- Writes videos and `pipeline_summary.json` under `<output_dir>/<timestamp>/`.
- Includes extra checking metrics (for example collision/between-finger checks).

### Multi-Object Evaluation (Short)

Run from `PI-VLA` root. This evaluates multiple object XY placements (12-grid) and saves one `result.json` per run plus `batch_summary.json`.

Student-model benchmark batch:

```bash
python trajectory_evaluation/multi_comparison/run_student_multi_benchmark_batch.py \
  --student-checkpoint /path/to/best_z_goal_model_*.pth \
  --checkpoint teacher_model.pt \
  --seed 123 \
  --out-root output/trajectory_evaluation/student_multi_batch
```

RRT + NTField baseline batch:

```bash
python trajectory_evaluation/multi_comparison/run_rrt_ntfield_multi_benchmark_batch.py \
  --checkpoint teacher_model.pt \
  --seed 123 \
  --out-root output/trajectory_evaluation/rrt_ntfield_multi_batch
```

Analyze one or more batch outputs:

```bash
python trajectory_evaluation/multi_comparison/analyze_batch_results.py \
  --summary /path/to/student_multi_batch/batch_summary.json \
  --summary /path/to/rrt_ntfield_multi_batch/batch_summary.json \
  --labels student baseline
```

- Use `--no-video` in batch scripts for faster runs.
- Use `--save-final-geometric-debug` when you want per-run debug images.

### Student Model Evaluation (Short)

Run from `PI-VLA` root to evaluate a trained student checkpoint on pt shards:

```bash
python student_model_evaluation/evaluate_student_model.py \
  --checkpoint /path/to/best_z_goal_model_*.pth \
  --dataset-root student_model_training/data/pt_shards_multi \
  --split val \
  --batch-size 256
```

- Prints JSON metrics including `mse`, `mae`, `cos_distance`, `l2_mean`, and threshold accuracies.
- Add `--save-json output/student_eval.json` to save results.

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
