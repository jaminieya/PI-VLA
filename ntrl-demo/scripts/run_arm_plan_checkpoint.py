#!/usr/bin/env python3
"""
Load a trained arm metric checkpoint, run MPPI from a start/goal joint configuration,
and render the path against the obstacle point cloud (same pipeline as tests/arm_plan_stat.py).

Run from anywhere; the script cds to the ntrl-demo root.

Example:
  cd /path/to/ntrl-demo
  python scripts/run_arm_plan_checkpoint.py \\
    --checkpoint Experiments/UR5/arm_02_23_22_09/Model_Epoch_05000_ValLoss_3.579276e-03.pt \\
    --save plan.html

  # Custom joints (6 floats each, same convention as arm_plan_stat.py before BASE+scale):
  python scripts/run_arm_plan_checkpoint.py -c path/to/model.pt \\
    --start 0.2 -0.7 -1.0 1.5708 1.5708 0.0 \\
    --goal -0.2 -0.5 -0.35 0.6283 1.5708 0.0 \\
    --save out.html
"""

from __future__ import annotations

import argparse
import math
import os
import sys


def _ntrl_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    root = _ntrl_root()
    os.chdir(root)
    sys.path.insert(0, root)
    sys.path.insert(0, os.path.join(root, "tests"))

    import arm_plan_stat as aps  # noqa: E402

    p = argparse.ArgumentParser(description="Plan with trained arm checkpoint + visualize")
    p.add_argument(
        "--checkpoint",
        "-c",
        required=True,
        help="Path to Model_Epoch_*_ValLoss_*.pt (under Experiments/UR5/...)",
    )
    p.add_argument(
        "--save",
        "-s",
        default="arm_path.html",
        help="Output visualization (.html recommended)",
    )
    p.add_argument(
        "--obstacle",
        default="datasets/arm/UR5/realpc_scaled.off",
        help="Obstacle mesh/point cloud for background",
    )
    p.add_argument(
        "--start",
        type=float,
        nargs=6,
        metavar=("j0", "j1", "j2", "j3", "j4", "j5"),
        default=None,
        help="Start joint vector (6). Default: same demo as arm_plan_stat.py",
    )
    p.add_argument(
        "--goal",
        type=float,
        nargs=6,
        metavar=("j0", "j1", "j2", "j3", "j4", "j5"),
        default=None,
        help="Goal joint vector (6). Default: same demo as arm_plan_stat.py",
    )
    p.add_argument(
        "--no-base",
        action="store_true",
        help="Do not add the fixed BASE offset used in arm_plan_stat.py",
    )
    p.add_argument("--mppi-runs", type=int, default=5, help="Number of MPPI outer repeats")
    p.add_argument("--seed", type=int, default=None, help="Random seed for MPPI noise")

    args = p.parse_args()
    ckpt = args.checkpoint
    if not os.path.isabs(ckpt):
        ckpt = os.path.join(root, ckpt)
    if os.path.isdir(ckpt):
        print(
            f"Error: --checkpoint must be a .pt file, not a directory:\n  {ckpt}\n"
            f"Example: .../Model_Epoch_04300_ValLoss_*.pt inside that folder",
            file=sys.stderr,
        )
        sys.exit(1)
    if not os.path.isfile(ckpt):
        print(f"Error: checkpoint not found: {ckpt}", file=sys.stderr)
        sys.exit(1)

    if args.start is None:
        start6 = [0.2, -0.7, -1.0, 0.5 * math.pi, 0.5 * math.pi, 0.0]
    else:
        start6 = list(args.start)
    if args.goal is None:
        goal6 = [-0.2, -0.5, -0.35, 0.2 * math.pi, 0.5 * math.pi, 0.0]
    else:
        goal6 = list(args.goal)

    obs = args.obstacle
    if not os.path.isabs(obs):
        obs = os.path.join(root, obs)

    aps.run_plan_and_viz(
        ckpt,
        start6,
        goal6,
        save_path=args.save,
        obstacle_mesh=obs,
        apply_base=not args.no_base,
        mppi_runs=args.mppi_runs,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
