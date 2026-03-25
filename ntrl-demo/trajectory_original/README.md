# Trajectory NTField (Original)

NTField training and planning **without** collision avoidance.

- **Training**: Trajectory supervision + Eikonal (F=1) loss
- **Planning**: Gradient descent from start to goal configuration

## Usage

### Training

```bash
cd ntrl-demo
python trajectory_original/train_arm_trajectory.py --data_path ./datasets/arm/UR5_trajectory
```

### Planning

The original plan (no obstacle) is in `planning/` and used by hanwen_grasping:

```python
from planning import plan
path = plan(model, q_start, q_goal, step_size=0.02, max_steps=200, tol=0.01, device="cuda")
```
