#!/usr/bin/env python3
"""
Generate NTField trajectory-style data (points.npy, tau_obs.npy) without RRTConnect / OMPL.

For each accepted sample:
  - Sample random q_start, q_goal (radians, 6-DoF UR5).
  - tau_obs = joint-space path length along the straight line: ||q_goal - q_start||_2
    (equivalent to arc length of linear interpolation, matching trajectory_sampler semantics).
  - Keep the pair only if the segment is collision-free on a fine grid, using the same
    FCL pipeline as hanwen_grasping.robot_arm_configuration.arm_collision_free.

Scene geometry matches create_static_collision_model() (table + side/upper boxes), i.e. the
same static obstacle model used by new_setup.py / collect_data-style pipelines — not the full
Isaac YCB clutter. For obstacle-rich data, extend with target_mesh / extra static objects
similar to get_path2grasp.

Usage (recommended from PI-VLA root):

  python ntrl-demo/dataprocessing/generate_straightline_collision_dataset.py \\
    --output_dir ntrl-demo/datasets/arm/UR5_straightline \\
    --num_pairs 100000

Default ``--sample_mode gaussian`` draws poses near the collect_data UR5 home (high acceptance).
Wide ``--sample_mode uniform`` is collision-sparse; use a large ``--max_tries_factor``.

The script temporarily changes cwd to hanwen_grasping so URDF/STL paths resolve like new_setup.py.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_NTRL_DEMO = os.path.dirname(_SCRIPT_DIR)
_PI_VLA_ROOT = os.path.dirname(_NTRL_DEMO)
_HANWEN_ROOT = os.path.join(_PI_VLA_ROOT, "hanwen_grasping")

if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from trajectory_sampler import SCALE

if not os.path.isdir(_HANWEN_ROOT):
    raise FileNotFoundError(f"Expected hanwen_grasping at {_HANWEN_ROOT}")

# Matches hanwen_grasping get_path2grasp / collect_data nominal home configuration (radians).
UR5E_HOME_JOINTS = np.array([0.7, -2.0, 2.5, -0.3, 0.7, 0.0], dtype=np.float64)


def segment_collision_free(rac, q_s, q_g, plane_obj, static_env_models, n_checks: int) -> bool:
    """True iff linearly interpolated configs along [q_s, q_g] are all collision-free."""
    q_s = np.asarray(q_s, dtype=np.float64).reshape(6)
    q_g = np.asarray(q_g, dtype=np.float64).reshape(6)
    flex: list = []
    for alpha in np.linspace(0.0, 1.0, n_checks, dtype=np.float64):
        q = (1.0 - alpha) * q_s + alpha * q_g
        if not rac.arm_collision_free(q.tolist(), plane_obj, static_env_models, flex):
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate points.npy + tau_obs.npy via collision-checked joint straight-line paths (no RRT)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for points.npy and tau_obs.npy",
    )
    parser.add_argument("--num_pairs", type=int, default=100_000, help="Target number of accepted pairs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_tries_factor",
        type=int,
        default=500,
        help="Stop after max_tries_factor * num_pairs proposals if undersampled (uniform mode needs large values)",
    )
    parser.add_argument("--tau_min", type=float, default=0.01)
    parser.add_argument("--tau_max", type=float, default=2.0)
    parser.add_argument("--segment_checks", type=int, default=32, help="Collision samples along each segment")
    # Scene (defaults aligned with typical collect_data / new_setup table + drawer height)
    parser.add_argument("--table_tx", type=float, default=0.6, help="Table half-length (m) — passed as scene_info[0]")
    parser.add_argument("--table_ty", type=float, default=0.9, help="Table half-width (m)")
    parser.add_argument("--table_tz", type=float, default=0.10, help="Table thickness (m)")
    parser.add_argument("--drawer_height", type=float, default=0.40, help="Drawer/cover stack height (m)")
    parser.add_argument("--robot_x", type=float, default=0.0, help="Robot base position x (world)")
    parser.add_argument("--robot_y", type=float, default=0.0, help="Robot base position y")
    parser.add_argument("--robot_z", type=float, default=0.0, help="Robot base position z")
    parser.add_argument(
        "--norm_limit",
        type=float,
        default=0.5,
        help="With --sample_mode uniform: q in [-norm_limit*SCALE, norm_limit*SCALE] per joint",
    )
    parser.add_argument(
        "--sample_mode",
        type=str,
        choices=("gaussian", "uniform"),
        default="gaussian",
        help="gaussian: q_s,q_g ~ N(nominal, sigma^2) per joint (high accept rate); uniform: wide box (very sparse)",
    )
    parser.add_argument(
        "--nominal",
        type=float,
        nargs=6,
        default=None,
        metavar=("q1", "q2", "q3", "q4", "q5", "q6"),
        help="Joint reference for gaussian mode (radians). Default: UR5e home from collect_data.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.85,
        help="Std (rad) per joint for gaussian sampling around nominal",
    )
    parser.add_argument(
        "--q_clip",
        type=float,
        default=None,
        help="If set, clip each joint to [-q_clip, q_clip] after sampling (radians)",
    )
    parser.add_argument(
        "--allow_partial",
        action="store_true",
        help="If set, save as many pairs as collected before max_tries instead of raising",
    )
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    _saved_cwd = os.getcwd()
    try:
        # Imports resolve against sys.path, not cwd; chdir alone does not load hanwen_grasping.
        if _HANWEN_ROOT not in sys.path:
            sys.path.insert(0, _HANWEN_ROOT)
        os.chdir(_HANWEN_ROOT)
        import fcl  # noqa: F401 — ensures same env as rest of hanwen stack
        import robot_arm_configuration as RC

        scene_info = [args.table_tx, args.table_ty, args.table_tz, args.drawer_height]
        static_env_models = RC.create_static_collision_model(scene_info, None, None)

        plane_normal = np.array([0.0, 0.0, 1.0])
        col_plane = fcl.Plane(plane_normal, 0.0)
        plane_obj = fcl.CollisionObject(col_plane, fcl.Transform())

        file_path = "./assets/urdf/ur5e/meshes/collision/"
        robot_offset = np.array([args.robot_x, args.robot_y, args.robot_z], dtype=np.float64)
        rac = RC.robot_arm_configuration(file_path, robot_offset, scene_info)
    finally:
        os.chdir(_saved_cwd)

    rng = np.random.default_rng(args.seed)
    span = 2.0 * args.norm_limit * SCALE
    low = -span * 0.5 + 0.0  # [-norm_limit * SCALE, norm_limit * SCALE]
    high = span * 0.5
    nominal = np.asarray(args.nominal if args.nominal is not None else UR5E_HOME_JOINTS, dtype=np.float64).reshape(
        6
    )
    q_clip = float(args.q_clip) if args.q_clip is not None else None

    points_list: list[np.ndarray] = []
    tau_list: list[float] = []
    max_tries = max(args.num_pairs * args.max_tries_factor, args.num_pairs + 1)
    tries = 0

    if args.sample_mode == "gaussian":
        print(
            f"Sampling collision-free straight-line pairs (mode=gaussian, nominal={nominal.tolist()}, "
            f"sigma={args.sigma}, tau in [{args.tau_min}, {args.tau_max}], scene_info={scene_info})..."
        )
    else:
        print(
            f"Sampling collision-free straight-line pairs (mode=uniform, joint box [{low:.4f}, {high:.4f}] rad/dim, "
            f"tau in [{args.tau_min}, {args.tau_max}], scene_info={scene_info})..."
        )

    def draw_pair() -> tuple[np.ndarray, np.ndarray]:
        if args.sample_mode == "uniform":
            return rng.uniform(low, high, size=6), rng.uniform(low, high, size=6)
        q_s = nominal + rng.normal(0.0, args.sigma, size=6)
        q_g = nominal + rng.normal(0.0, args.sigma, size=6)
        if q_clip is not None:
            q_s = np.clip(q_s, -q_clip, q_clip)
            q_g = np.clip(q_g, -q_clip, q_clip)
        return q_s, q_g

    while len(points_list) < args.num_pairs and tries < max_tries:
        tries += 1
        q_s, q_g = draw_pair()
        tau = float(np.linalg.norm(q_g - q_s))
        if tau < args.tau_min or tau > args.tau_max:
            continue

        if not segment_collision_free(
            rac, q_s, q_g, plane_obj, static_env_models, args.segment_checks
        ):
            continue

        q_s_norm = (q_s / SCALE).astype(np.float32)
        q_g_norm = (q_g / SCALE).astype(np.float32)
        points_list.append(np.concatenate([q_s_norm, q_g_norm]))
        tau_list.append(tau)

        if len(points_list) % 10000 == 0 and len(points_list) > 0:
            print(f"  accepted {len(points_list)} / {args.num_pairs} (tries={tries})")

    if len(points_list) < args.num_pairs:
        msg = (
            f"Only collected {len(points_list)} pairs (need {args.num_pairs}) after {tries} tries. "
            "Try: --sample_mode gaussian (default), adjust --sigma/--nominal, widen --tau_max, "
            "raise --max_tries_factor, or --allow_partial to save what was found."
        )
        if args.allow_partial and len(points_list) > 0:
            print(f"Warning: {msg}\nSaving partial dataset.")
        else:
            raise RuntimeError(msg)

    if len(points_list) == 0:
        raise RuntimeError("No samples accepted; try gaussian mode, larger sigma, or looser tau bounds.")

    points = np.stack(points_list, axis=0)
    tau_obs = np.array(tau_list, dtype=np.float32)

    p_path = os.path.join(output_dir, "points.npy")
    t_path = os.path.join(output_dir, "tau_obs.npy")
    np.save(p_path, points)
    np.save(t_path, tau_obs)

    print(f"Saved {points.shape[0]} samples")
    print(f"  points:   {p_path}  shape={points.shape}")
    print(f"  tau_obs: {t_path}  shape={tau_obs.shape}")
    print(f"  tau_obs range: [{tau_obs.min():.4f}, {tau_obs.max():.4f}]")
    print("Train with: python train/train_arm_trajectory.py --data_path <output_dir>")


if __name__ == "__main__":
    main()
