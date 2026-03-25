# Trajectory NTField with Collision Avoidance

NTField training and planning **with** collision avoidance.

- **Training**: Trajectory + Eikonal + collision loss (requires obstacle mesh)
- **Planning**: Gradient descent with obstacle repulsion

## Usage

### Training

```bash
cd ntrl-demo
python trajectory_collision/train_arm_trajectory.py \
  --data_path ./datasets/arm/UR5_trajectory \
  --obstacle_mesh datasets/arm/UR5/realpc_scaled.off
```

### Planning

```python
from trajectory_collision.gradient_planner_trajectory import plan

path = plan(model, q_start, q_goal,
            obstacle_mesh="datasets/arm/UR5/realpc_scaled.off",
            repulsion_weight=0.1, repulsion_margin=0.05)
```
