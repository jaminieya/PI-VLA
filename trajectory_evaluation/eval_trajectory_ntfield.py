#!/usr/bin/env python3
"""
Evaluate trajectory-trained NTField: field fit (tau vs tau_obs) and planning (gradient plan to q_goal).

Run from PI-VLA root (recommended):
  python trajectory_evaluation/eval_trajectory_ntfield.py \\
    --checkpoint ntrl-demo/Experiments/UR5_trajectory/trajectory_XX_XX_XX_XX/Model_Epoch_00500_*.pt \\
    --data_path ntrl-demo/datasets/arm/UR5_trajectory \\
    --device cuda:0

Expects points.npy (N,12) and tau_obs.npy (N,) with q_start,q_goal in NTField normalized space (see trajectory_sampler.SCALE).
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch

# ntrl-demo is a sibling of trajectory_evaluation/ under PI-VLA
_NTRL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ntrl-demo"))
if _NTRL_ROOT not in sys.path:
    sys.path.insert(0, _NTRL_ROOT)

from models.metric_arm import model_function_metric as model_function
from models.metric_arm import model_network_metric as model_network
from planning.gradient_planner_trajectory import SCALE, plan

torch.backends.cudnn.benchmark = True


@dataclass
class PlannerStats:
    n: int
    n_success: int
    mean_final_l2_rad: float
    median_final_l2_rad: float
    mean_final_linf_rad: float
    mean_steps: float
    mean_path_length_rad: float


class _ModelShim:
    """gradient_planner_trajectory.plan expects model.function."""

    def __init__(self, function: model_function.Function):
        self.function = function


def load_network_and_function(
    checkpoint_path: str, experiment_dir: str | None, device: torch.device, dim: int = 6
) -> Tuple[model_network.NN, model_function.Function]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    B = ckpt["B_state_dict"]
    if not isinstance(B, torch.Tensor):
        raise TypeError("Checkpoint missing tensor B_state_dict")
    B = B.to(device)
    network = model_network.NN(device, dim, B)
    network.load_state_dict(ckpt["model_state_dict"])
    network.to(device)
    network.eval()

    if experiment_dir is None:
        experiment_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    function = model_function.Function(experiment_dir, device, network, dim)
    return network, function


def evaluate_field(
    function: model_function.Function,
    points: torch.Tensor,
    tau_obs: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> Tuple[float, float, float]:
    """RMSE, MAE, mean relative error |tau_pred - tau| / (|tau|+1e-6)."""
    n = points.shape[0]
    sq_err = 0.0
    abs_err = 0.0
    rel_err = 0.0
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            pb = points[start:end].to(device)
            tb = tau_obs[start:end].to(device)
            tau_pred = function.TravelTimes(pb)
            d = tau_pred - tb
            sq_err += float(torch.sum(d * d).item())
            abs_err += float(torch.sum(torch.abs(d)).item())
            rel_err += float(
                torch.sum(torch.abs(d) / (torch.abs(tb) + 1e-6)).item()
            )
    rmse = float(np.sqrt(sq_err / n))
    mae = float(abs_err / n)
    mre = float(rel_err / n)
    return rmse, mae, mre


def evaluate_planner(
    function: model_function.Function,
    points_np: np.ndarray,
    indices: np.ndarray,
    device_str: str,
    step_size: float,
    max_steps: int,
    tol: float,
    goal_success_eps_rad: float,
) -> PlannerStats:
    """points_np rows: [q_s_norm, q_g_norm] concatenated."""
    model = _ModelShim(function)
    dim = 6
    final_l2: List[float] = []
    final_linf: List[float] = []
    steps_list: List[int] = []
    path_lens: List[float] = []
    n_success = 0

    for i in indices:
        row = points_np[i]
        q_s_n = row[:dim]
        q_g_n = row[dim:]
        q_start = (q_s_n * SCALE).astype(np.float64)
        q_goal = (q_g_n * SCALE).astype(np.float64)

        path_rad = plan(
            model,
            q_start,
            q_goal,
            step_size=step_size,
            max_steps=max_steps,
            tol=tol,
            device=device_str,
        )
        # Last point is exact q_goal appended by plan(); use second-to-last as final planner state
        q_final = np.asarray(path_rad[-2], dtype=np.float64).reshape(6)
        err = q_final - q_goal
        l2 = float(np.linalg.norm(err))
        linf = float(np.max(np.abs(err)))
        final_l2.append(l2)
        final_linf.append(linf)
        if l2 < goal_success_eps_rad:
            n_success += 1

        # steps: interior waypoints excluding initial q_start and appended q_goal duplicate
        n_steps = max(0, len(path_rad) - 2)
        steps_list.append(n_steps)
        seg = 0.0
        for a, b in zip(path_rad[:-1], path_rad[1:]):
            seg += float(np.linalg.norm(np.asarray(a) - np.asarray(b)))
        path_lens.append(seg)

    arr_l2 = np.array(final_l2, dtype=np.float64)
    return PlannerStats(
        n=len(indices),
        n_success=n_success,
        mean_final_l2_rad=float(arr_l2.mean()),
        median_final_l2_rad=float(np.median(arr_l2)),
        mean_final_linf_rad=float(np.mean(final_linf)),
        mean_steps=float(np.mean(steps_list)),
        mean_path_length_rad=float(np.mean(path_lens)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trajectory NTField checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model_Epoch_*.pt from train_arm_trajectory")
    parser.add_argument("--data_path", type=str, required=True, help="Directory with points.npy and tau_obs.npy")
    parser.add_argument("--experiment_dir", type=str, default=None, help="Function log dir (default: checkpoint parent)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Fraction of data for held-out eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_plan_samples", type=int, default=200, help="Max pairs for planner metrics (subset of val)")
    parser.add_argument("--step_size", type=float, default=0.02)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--tol", type=float, default=0.01, help="plan() convergence threshold in normalized space")
    parser.add_argument(
        "--goal_success_eps_rad",
        type=float,
        default=None,
        help="Success if ||q_final-q_goal||_2 < this (rad); default tol*SCALE",
    )
    parser.add_argument("--skip_planner", action="store_true", help="Only compute field metrics")
    args = parser.parse_args()

    if args.device != "cpu" and not torch.cuda.is_available():
        print("Warning: CUDA unavailable; using cpu")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    if args.goal_success_eps_rad is None:
        goal_eps = float(args.tol * SCALE)
    else:
        goal_eps = args.goal_success_eps_rad

    data_path = os.path.abspath(args.data_path)
    points_path = os.path.join(data_path, "points.npy")
    tau_path = os.path.join(data_path, "tau_obs.npy")
    if not os.path.isfile(points_path) or not os.path.isfile(tau_path):
        raise FileNotFoundError(f"Need {points_path} and {tau_path}")

    points_np = np.load(points_path)
    tau_np = np.load(tau_path).astype(np.float64)
    if points_np.shape[0] != tau_np.shape[0]:
        raise ValueError("points.npy and tau_obs.npy length mismatch")
    n = points_np.shape[0]

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_val = max(1, int(round(n * args.val_ratio)))
    val_idx = perm[:n_val]

    points_t = torch.tensor(points_np, dtype=torch.float32)
    tau_t = torch.tensor(tau_np, dtype=torch.float32)

    _, function = load_network_and_function(
        os.path.abspath(args.checkpoint),
        args.experiment_dir,
        device,
        dim=6,
    )

    print(f"Dataset: n={n}, val holdout: {n_val} (val_ratio={args.val_ratio}, seed={args.seed})")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")

    val_points = points_t[val_idx]
    val_tau = tau_t[val_idx]
    rmse, mae, mre = evaluate_field(function, val_points, val_tau, args.batch_size, device)
    print("\n--- Field metrics (held-out val) ---")
    print(f"  tau RMSE: {rmse:.6e}")
    print(f"  tau MAE:  {mae:.6e}")
    print(f"  mean |rel err|: {mre:.6e}")

    if args.skip_planner:
        return

    n_plan = min(args.max_plan_samples, len(val_idx))
    plan_idx = val_idx[:n_plan]
    dev_str = str(device) if device.type == "cuda" else "cpu"
    stats = evaluate_planner(
        function,
        points_np,
        plan_idx,
        device_str=dev_str,
        step_size=args.step_size,
        max_steps=args.max_steps,
        tol=args.tol,
        goal_success_eps_rad=goal_eps,
    )
    print(f"\n--- Planner metrics (n={stats.n}, goal_eps_rad={goal_eps:.6f}) ---")
    print(f"  success rate: {stats.n_success}/{stats.n} ({100.0 * stats.n_success / stats.n:.1f}%)")
    print(f"  mean final L2 error (rad):   {stats.mean_final_l2_rad:.6f}")
    print(f"  median final L2 error (rad): {stats.median_final_l2_rad:.6f}")
    print(f"  mean final Linf error (rad): {stats.mean_final_linf_rad:.6f}")
    print(f"  mean steps (before appended goal): {stats.mean_steps:.2f}")
    print(f"  mean path length (rad, sum of segments): {stats.mean_path_length_rad:.6f}")


if __name__ == "__main__":
    main()
