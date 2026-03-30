"""
Gradient descent planner for trajectory NTField WITH collision avoidance.

Plans path from q_start to q_goal using obstacle mesh for repulsion near obstacles.
"""

import numpy as np
import torch

SCALE = np.pi / 0.5


def plan(model, q_start, q_goal, step_size=0.02, max_steps=200, tol=0.01, device="cuda",
         obstacle_mesh=None, repulsion_weight=0.1, repulsion_margin=0.05):
    """
    Plan path with optional obstacle repulsion.

    Args:
        model: NTField model with model.function.Gradient(Xp)
        q_start, q_goal: (6,) joint configs in radians
        obstacle_mesh: Path to .off mesh for collision-aware planning
        repulsion_weight: Weight for obstacle repulsion
        repulsion_margin: Distance threshold (meters) for repulsion

    Returns:
        List of (6,) joint configs in radians.
    """
    dim = 6
    q_start = np.asarray(q_start, dtype=np.float64).reshape(6)
    q_goal = np.asarray(q_goal, dtype=np.float64).reshape(6)

    collision_fn = None
    if obstacle_mesh is not None:
        try:
            from trajectory_collision.collision_utils import (
                build_obstacle_kdtree,
                build_arm_chain,
                arm_obstacle_distance_batch,
            )
            kdtree, v_obs = build_obstacle_kdtree(obstacle_mesh, device=device)
            chain, mesh_list = build_arm_chain()

            def collision_fn(q_norm):
                th = torch.tensor(q_norm * SCALE, dtype=torch.float32, device=device).unsqueeze(0)
                dist, normal = arm_obstacle_distance_batch(th, chain, mesh_list, kdtree, v_obs, device)
                return dist.item(), normal[0].detach().cpu().numpy()
        except Exception as e:
            import warnings
            warnings.warn(f"Obstacle mesh failed ({e}), planning without collision avoidance")

    q_start_norm = q_start / SCALE
    q_goal_norm = q_goal / SCALE

    XP = np.concatenate([q_start_norm, q_goal_norm]).astype(np.float32)
    XP = torch.tensor(XP, dtype=torch.float32, device=device).unsqueeze(0)
    XP.requires_grad_(True)

    path_norm = [q_start_norm.copy()]

    for step in range(max_steps):
        q_current = XP[0, :dim].detach().cpu().numpy()
        dist = np.linalg.norm(q_current - q_goal_norm)
        if dist < tol:
            break

        grad = model.function.Gradient(XP)
        grad_step = grad[:, :dim].detach().clone()

        if collision_fn is not None:
            try:
                obs_dist, obs_normal = collision_fn(q_current)
                if obs_dist < repulsion_margin and obs_dist > 0:
                    repulsion = torch.tensor(obs_normal, dtype=torch.float32, device=device)
                    repulsion = torch.nn.functional.normalize(repulsion.unsqueeze(0), dim=1).squeeze(0)
                    grad_step = grad_step + repulsion_weight * repulsion.unsqueeze(0)
            except Exception:
                pass

        with torch.no_grad():
            XP_new = XP.clone().detach()
            XP_new[:, :dim] = XP[:, :dim].detach() + step_size * grad_step.squeeze(0)
            XP = XP_new.requires_grad_(True)

        path_norm.append(XP[0, :dim].detach().cpu().numpy().copy())

    path_norm.append(q_goal_norm.copy())
    path_rad = [p * SCALE for p in path_norm]
    return path_rad
