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
    Plan path from q_start to a target latent representation using gradient descent.
    """
    dim = len(q_start)
    
    # Assuming SCALE is a global constant as implied in the original codebase
    global SCALE 
    
    q_start = np.asarray(q_start, dtype=np.float64).reshape(dim)
    q_start_norm = q_start / SCALE

    # Set up start tensor
    q_tensor = torch.tensor(q_start_norm, dtype=torch.float32, device=device).unsqueeze(0)

    # Ensure z_goal is correctly formatted
    if not isinstance(z_goal, torch.Tensor):
        z_goal = torch.tensor(z_goal, dtype=torch.float32, device=device)
    if z_goal.dim() == 1:
        z_goal = z_goal.unsqueeze(0)

    path_norm = [q_start_norm.copy()]

    # Access the network (handling potential wrapper layers)
    network = getattr(model, 'network', getattr(model, 'function', model))
    if hasattr(network, 'network'):
        network = network.network

    for step in range(max_steps):
        
        # q_tensor must be a fresh leaf each iteration
        q_inp = q_tensor.detach().requires_grad_(True)
        
        # Forward pass
        tau, w, coords = network.out_with_goal_latent(q_inp, z_goal)
        
        # Check convergence based on proximity (tau approaches 0)
        if tau.item() < 1e-4:
            break

        # Differentiate w.r.t. coords (the intermediate node inside the network)
        # This returns a (1, 12) gradient tensor
        dtau_tuple = torch.autograd.grad(
            outputs=tau, 
            inputs=coords, 
            grad_outputs=torch.ones_like(tau),
            only_inputs=True,
            allow_unused=True # Good practice when checking for graph disconnects
        )
        
        dtau = dtau_tuple[0]

        # Guard against disconnected graph or entirely flat gradients
        if dtau is None or dtau.abs().max().item() < 1e-12:
            print(f"[plan_with_goal_latent] Warning: zero gradient at step {step}, graph may be disconnected or flat.")
            break
        
        # Isolate the gradient for the start configuration (first half)
        grad_q = dtau[:, :dim]
        
        # Calculate the gradient magnitude
        grad_mag = torch.norm(grad_q, dim=1).view(-1, 1)
        
        # Check convergence based on a flat gradient
        if grad_mag.item() < grad_tol:
            break

        # Reproduce the exact inverse-squared normalization
        update_dir = -grad_q / (grad_mag**2 + 1e-8)

        # Update q_start in normalized space 
        with torch.no_grad():
            q_tensor = q_inp.detach() + step_size * update_dir

        path_norm.append(q_tensor[0].detach().cpu().numpy().copy())

    # Convert normalized path back to radians
    path_rad = [p * SCALE for p in path_norm]
    
    return path_rad