#!/usr/bin/env python3
"""
Collect NTField metric-training data using the exact environment generator in
`hanwen_grasping/new_constraint_setup.py`.

This launcher calls new_constraint_setup with `--collect_ntfield_metric` so it:
  - builds the same constrained clutter scene,
  - skips RRT/OMPL planning for data collection, and
  - writes `sampled_points.npy`, `speed.npy`, `normal.npy` for
    `PI-VLA/ntrl-demo/train/train_arm.py`.

Example:
  cd PI-VLA/hanwen_grasping
  python collect_data/constraint_ntfield_data_collection.py \
      --headless \
      --metric_num_samples 100000 \
      --metric_output_dir ../ntrl-demo/datasets/arm/UR5
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import List


def _build_command(args: argparse.Namespace, passthrough: List[str], script_path: str) -> List[str]:
    if args.simple_grasp_collect:
        cmd = [
            sys.executable,
            script_path,
            "--simple_grasp_collect",
            "--env_id",
            str(args.env_id),
            "--simple_num_candidates",
            str(args.simple_num_candidates),
            "--simple_interp_steps",
            str(args.simple_interp_steps),
        ]
        if args.headless:
            cmd.append("--headless")
        cmd.extend(passthrough)
        return cmd

    metric_num_samples = args.metric_num_samples
    metric_sampler_mode = args.metric_sampler_mode
    metric_visualize_sampling = args.metric_visualize_sampling
    metric_visualize_every_accepted = args.metric_visualize_every_accepted
    metric_visualize_hold_steps = args.metric_visualize_hold_steps
    metric_qstart_only = args.metric_qstart_only
    headless = args.headless

    if args.visualize_random_grasp_run:
        # One debug run: force TRAC-IK mesh sampling and show accepted pair every run.
        metric_num_samples = 1
        metric_sampler_mode = "ik_mesh"
        metric_visualize_sampling = True
        metric_visualize_every_accepted = 1
        metric_qstart_only = True
        # Keep viewer alive long enough to inspect q_start -> q_goal transition.
        metric_visualize_hold_steps = max(metric_visualize_hold_steps, 60)
        headless = False

    cmd = [
        sys.executable,
        script_path,
        "--collect_ntfield_metric",
        "--env_id",
        str(args.env_id),
        "--metric_num_samples",
        str(metric_num_samples),
        "--metric_seed",
        str(args.metric_seed),
        "--metric_max_tries_factor",
        str(args.metric_max_tries_factor),
        "--metric_sampler_mode",
        str(metric_sampler_mode),
        "--metric_ik_pose_trials",
        str(args.metric_ik_pose_trials),
        "--metric_ik_seed_trials",
        str(args.metric_ik_seed_trials),
        "--metric_ik_surface_offset_min",
        str(args.metric_ik_surface_offset_min),
        "--metric_ik_surface_offset_max",
        str(args.metric_ik_surface_offset_max),
        "--metric_ik_tool_offset_x",
        str(args.metric_ik_tool_offset_x),
        "--metric_ik_tool_offset_y",
        str(args.metric_ik_tool_offset_y),
        "--metric_ik_tool_offset_z",
        str(args.metric_ik_tool_offset_z),
        "--metric_ik_urdf_file",
        str(args.metric_ik_urdf_file),
        "--metric_log_every_tries",
        str(args.metric_log_every_tries),
    ]

    if headless:
        cmd.append("--headless")

    if args.metric_output_dir:
        cmd.extend(["--metric_output_dir", os.path.abspath(args.metric_output_dir)])

    if metric_visualize_sampling:
        cmd.append("--metric_visualize_sampling")
        cmd.extend(
            [
                "--metric_visualize_every_accepted",
                str(metric_visualize_every_accepted),
                "--metric_visualize_hold_steps",
                str(metric_visualize_hold_steps),
            ]
        )

    if metric_qstart_only:
        cmd.append("--metric_qstart_only")
    if args.metric_save_speed_normal:
        cmd.append("--metric_save_speed_normal")

    # Forward additional scene controls supported by new_constraint_setup.py
    # (e.g. --no_walls, --compute_device_id, --graphics_device_id, etc.).
    cmd.extend(passthrough)
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Collect train_arm metric data from the same constrained environment "
            "as new_constraint_setup.py."
        )
    )
    parser.add_argument("--env_id", type=int, default=0, help="Environment ID.")
    parser.add_argument(
        "--metric_num_samples",
        type=int,
        default=100000,
        help="Number of NTField metric samples.",
    )
    parser.add_argument("--metric_seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--metric_max_tries_factor",
        type=int,
        default=2000,
        help="Max proposal tries = metric_num_samples * factor.",
    )
    parser.add_argument(
        "--metric_output_dir",
        type=str,
        default=None,
        help=(
            "Output directory for sampled_points.npy/speed.npy/normal.npy. "
            "If omitted, new_constraint_setup default is used."
        ),
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without viewer window.",
    )
    parser.add_argument(
        "--metric_sampler_mode",
        type=str,
        default="fcl_uniform",
        choices=["fcl_uniform", "ik_mesh"],
        help="q_start sampler mode: default fcl_uniform, or trac_ik mesh-based ik_mesh.",
    )
    parser.add_argument(
        "--metric_ik_pose_trials",
        type=int,
        default=80,
        help="When --metric_sampler_mode ik_mesh: mesh poses tried per sample proposal.",
    )
    parser.add_argument(
        "--metric_ik_seed_trials",
        type=int,
        default=6,
        help="When --metric_sampler_mode ik_mesh: IK restarts per pose.",
    )
    parser.add_argument(
        "--metric_ik_surface_offset_min",
        type=float,
        default=0.002,
        help="When --metric_sampler_mode ik_mesh: minimum offset (m) from mesh surface.",
    )
    parser.add_argument(
        "--metric_ik_surface_offset_max",
        type=float,
        default=0.03,
        help="When --metric_sampler_mode ik_mesh: maximum offset (m) from mesh surface.",
    )
    parser.add_argument(
        "--metric_ik_tool_offset_x",
        type=float,
        default=0.11,
        help="IK target->wrist offset x (m), default aligned with new_setup.py.",
    )
    parser.add_argument(
        "--metric_ik_tool_offset_y",
        type=float,
        default=0.0,
        help="IK target->wrist offset y (m).",
    )
    parser.add_argument(
        "--metric_ik_tool_offset_z",
        type=float,
        default=0.08,
        help="IK target->wrist offset z (m), default aligned with new_setup.py.",
    )
    parser.add_argument(
        "--metric_ik_urdf_file",
        type=str,
        default="ur5e_mimic_real_gripper_test.urdf",
        help="URDF filename under assets/urdf/ur5e for TRAC-IK in metric collection.",
    )
    parser.add_argument(
        "--metric_log_every_tries",
        type=int,
        default=2000,
        help="Print progress every N proposal tries (<=0 disables periodic logs).",
    )
    parser.add_argument(
        "--metric_qstart_only",
        action="store_true",
        help="Collect only q_start IK collision-free samples (skip q_goal sampling).",
    )
    parser.add_argument(
        "--metric_save_speed_normal",
        action="store_true",
        help="When q_goal is sampled, also save speed.npy and normal.npy.",
    )
    parser.add_argument(
        "--metric_visualize_sampling",
        action="store_true",
        help="Visualize sampled q_start/q_goal in Isaac viewer during collection (requires not --headless).",
    )
    parser.add_argument(
        "--metric_visualize_every_accepted",
        type=int,
        default=100,
        help="When visualizing: render every N accepted samples.",
    )
    parser.add_argument(
        "--metric_visualize_hold_steps",
        type=int,
        default=20,
        help="When visualizing: physics/viewer steps to hold each of q_start and q_goal.",
    )
    parser.add_argument(
        "--visualize_random_grasp_run",
        action="store_true",
        help=(
            "Debug mode for per-run visualization: Isaac Gym viewer + TRAC-IK mesh grasp "
            "sampling + collision-checked q_start/q_goal display for a single accepted sample."
        ),
    )
    parser.add_argument(
        "--simple_grasp_collect",
        action="store_true",
        help=(
            "Run simplified pipeline (no RRT): one object, sample TRAC-IK q_g candidates, "
            "pick straight-line collision-reachable q_g from current q_s, and save q_s->q_g path."
        ),
    )
    parser.add_argument(
        "--simple_num_candidates",
        type=int,
        default=100,
        help="In --simple_grasp_collect mode: number of TRAC-IK grasp candidates to test.",
    )
    parser.add_argument(
        "--simple_interp_steps",
        type=int,
        default=120,
        help="In --simple_grasp_collect mode: interpolation steps for straight-line reachability check.",
    )

    args, passthrough = parser.parse_known_args()

    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    hanwen_root = os.path.abspath(os.path.join(this_file_dir, ".."))
    new_constraint_setup = os.path.join(hanwen_root, "new_constraint_setup.py")
    if not os.path.isfile(new_constraint_setup):
        raise FileNotFoundError(
            f"Cannot find new_constraint_setup.py at: {new_constraint_setup}"
        )

    cmd = _build_command(args, passthrough, new_constraint_setup)

    print(
        "Running simple grasp-collect command:"
        if args.simple_grasp_collect
        else "Running NTField metric data collection command:"
    )
    print(" ".join(cmd))
    if args.visualize_random_grasp_run:
        print(
            "Visualization debug mode enabled: forcing single-sample ik_mesh run with viewer "
            "to inspect random grasp generation and collision-checked states."
        )
    print(
        "Tip: pass extra args after the known flags to control the same scene as "
        "new_constraint_setup.py (for example: --no_walls)."
    )
    return subprocess.call(cmd, cwd=hanwen_root)


if __name__ == "__main__":
    raise SystemExit(main())
