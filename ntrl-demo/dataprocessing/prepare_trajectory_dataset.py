#!/usr/bin/env python3
"""
Prepare trajectory dataset for NTField training.

Loads RRT-generated trajectories from HDF5 files (produced by
hanwen_grasping/new_setup_dataset_collect.py), samples (q_s, q_g) pairs
with tau_obs (path length), and saves to points.npy and tau_obs.npy.

Usage:
    python dataprocessing/prepare_trajectory_dataset.py \\
        --data_dir ../collected_data \\
        --output_dir ./datasets/arm/UR5_trajectory \\
        --num_pairs 100000

Or from PI-VLA root:
    python ntrl-demo/dataprocessing/prepare_trajectory_dataset.py \\
        --data_dir collected_data \\
        --output_dir ntrl-demo/datasets/arm/UR5_trajectory \\
        --num_pairs 100000
"""

import argparse
import os
import sys

import numpy as np

# Allow importing from dataprocessing when run from ntrl-demo root or dataprocessing/
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
from trajectory_sampler import load_trajectories_from_h5, sample_pairs_from_trajectories


def main():
    parser = argparse.ArgumentParser(
        description="Prepare trajectory dataset for NTField training"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to directory containing *.h5 files (e.g. collected_data/)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for points.npy and tau_obs.npy",
    )
    parser.add_argument(
        "--num_pairs",
        type=int,
        default=100000,
        help="Number of (q_s, q_g) pairs to sample (default: 100000)",
    )
    parser.add_argument(
        "--tau_min",
        type=float,
        default=0.01,
        help="Minimum tau_obs (path length) to filter trivial pairs (default: 0.01)",
    )
    parser.add_argument(
        "--tau_max",
        type=float,
        default=2.0,
        help="Maximum tau_obs to filter very long pairs (default: 2.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    print(f"Loading trajectories from {data_dir}...")
    trajectories = load_trajectories_from_h5(data_dir)
    print(f"Loaded {len(trajectories)} trajectories")

    if not trajectories:
        raise ValueError("No valid trajectories found. Check HDF5 files.")

    total_waypoints = sum(len(t) for t in trajectories)
    print(f"Total waypoints: {total_waypoints}")

    rng = np.random.default_rng(args.seed)
    print(f"Sampling {args.num_pairs} pairs (tau in [{args.tau_min}, {args.tau_max}])...")
    points, tau_obs = sample_pairs_from_trajectories(
        trajectories,
        num_pairs=args.num_pairs,
        tau_min=args.tau_min,
        tau_max=args.tau_max,
        rng=rng,
    )

    os.makedirs(output_dir, exist_ok=True)
    points_path = os.path.join(output_dir, "points.npy")
    tau_path = os.path.join(output_dir, "tau_obs.npy")

    np.save(points_path, points)
    np.save(tau_path, tau_obs)

    print(f"Saved {points.shape[0]} samples:")
    print(f"  points:   {points_path}  shape={points.shape}")
    print(f"  tau_obs: {tau_path}  shape={tau_obs.shape}")
    print(f"  tau_obs range: [{tau_obs.min():.4f}, {tau_obs.max():.4f}]")


if __name__ == "__main__":
    main()
