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

## Notes

- Each subfolder may have its own environment/dependency assumptions.
- Start with folder-level scripts/READMEs for command details.
