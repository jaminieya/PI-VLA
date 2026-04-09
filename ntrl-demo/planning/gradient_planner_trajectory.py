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


def plan_with_goal_latent(
    model,
    q_start,
    z_goal,
    step_size=0.02,
    max_steps=200,
    grad_tol=1e-3,
    device="cuda",
):
    """
    Plan path from q_start using a fixed goal latent z_goal.

    This uses ``model.network.out_with_goal_latent(q_start_norm, z_goal)``, so the
    optimization is done directly against the latent goal representation instead of
    explicit q_goal joints.

    Args:
        model: NTField model with ``model.network.out_with_goal_latent``.
        q_start: (6,) numpy array, start joint config in radians.
        z_goal: (H,) or (1, H) goal latent from the same network latent space.
        step_size: gradient step size in normalized space.
        max_steps: maximum planning iterations.
        grad_tol: stop when ||descent_direction||_2 falls below this threshold.
        device: torch device.

    Returns:
        List of (6,) joint configs in radians, starting from q_start.
    """
    q_start = np.asarray(q_start, dtype=np.float64).reshape(6)
    q_start_norm = q_start / SCALE

    q_current = torch.tensor(q_start_norm, dtype=torch.float32, device=device).unsqueeze(0)
    q_current.requires_grad_(True)

    if isinstance(z_goal, np.ndarray):
        z_goal_t = torch.from_numpy(z_goal).to(device=device, dtype=torch.float32)
    elif torch.is_tensor(z_goal):
        z_goal_t = z_goal.to(device=device, dtype=torch.float32)
    else:
        z_goal_t = torch.tensor(z_goal, dtype=torch.float32, device=device)
    if z_goal_t.dim() == 1:
        z_goal_t = z_goal_t.unsqueeze(0)
    z_goal_t = z_goal_t.detach()

    path_norm = [q_start_norm.copy()]

    for _ in range(max_steps):
        tau, _, q_for_grad = model.network.out_with_goal_latent(q_current, z_goal_t)
        grad = torch.autograd.grad(tau, q_for_grad, torch.ones_like(tau), create_graph=False)[0]

        direction = -grad
        direction = direction / (torch.norm(direction, dim=1, keepdim=True) ** 2 + 1e-12)
        step_norm = torch.norm(direction, dim=1).item()
        if step_norm < grad_tol:
            break

        with torch.no_grad():
            q_next = q_for_grad.detach() + step_size * direction.detach()
            q_current = q_next.requires_grad_(True)

        path_norm.append(q_current[0].detach().cpu().numpy().copy())

    path_rad = [p * SCALE for p in path_norm]
    return path_rad
