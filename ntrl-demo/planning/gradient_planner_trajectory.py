"""
Gradient descent planner for trajectory-trained NTField.

Plans path from q_start to q_goal using the trajectory NTField (no obstacle/speed_kdtree).
Uses model.function.Gradient(XP) which returns descent direction toward goal.
"""

import numpy as np
import torch

SCALE = np.pi / 0.5


def plan(model, q_start, q_goal, step_size=0.02, max_steps=200, tol=0.01, device="cuda"):
    """
    Plan path from q_start to q_goal using gradient descent on the trajectory NTField.

    Args:
        model: NTField model with model.function.Gradient(Xp) where Xp is (1, 12) tensor
        q_start: (6,) numpy array, start joint config in radians
        q_goal: (6,) numpy array, goal joint config in radians
        step_size: gradient step size in normalized space
        max_steps: maximum planning iterations
        tol: convergence threshold (L2 distance in normalized space)
        device: torch device

    Returns:
        List of (6,) joint configs in radians, from q_start to q_goal.
    """
    dim = 6
    q_start = np.asarray(q_start, dtype=np.float64).reshape(6)
    q_goal = np.asarray(q_goal, dtype=np.float64).reshape(6)

    # Normalize to NTField input space
    q_start_norm = q_start / SCALE
    q_goal_norm = q_goal / SCALE

    # XP = [q_start, q_goal] concatenated, shape (1, 12)
    XP = np.concatenate([q_start_norm, q_goal_norm]).astype(np.float32)
    XP = torch.tensor(XP, dtype=torch.float32, device=device).unsqueeze(0)
    XP.requires_grad_(True)

    path_norm = [q_start_norm.copy()]

    for step in range(max_steps):
        # Check convergence
        q_current = XP[0, :dim].detach().cpu().numpy()
        dist = np.linalg.norm(q_current - q_goal_norm)
        if dist < tol:
            break

        # Get gradient: Gradient returns (Ypred0, Ypred1) for q_start and q_goal
        # Ypred0 points toward decreasing travel time (toward goal)
        grad = model.function.Gradient(XP)

        # Update q_start in normalized space (detach to avoid graph build-up)
        with torch.no_grad():
            XP_new = XP.clone().detach()
            XP_new[:, :dim] = XP[:, :dim].detach() + step_size * grad[:, :dim].detach()
            XP = XP_new.requires_grad_(True)

        path_norm.append(XP[0, :dim].detach().cpu().numpy().copy())

    # Add goal
    path_norm.append(q_goal_norm.copy())

    # Convert to radians
    path_rad = [p * SCALE for p in path_norm]
    return path_rad
