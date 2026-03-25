#!/usr/bin/env python3
"""
Quick test of the trajectory data pipeline with dummy HDF5 data.

Creates a temporary dummy .h5 file and runs the pipeline to verify
load_trajectories_from_h5 and sample_pairs_from_trajectories work correctly.
"""

import os
import sys
import tempfile
import numpy as np
import h5py

# Add dataprocessing to path
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from trajectory_sampler import (
    load_trajectories_from_h5,
    sample_pairs_from_trajectories,
    arc_length,
)


def create_dummy_h5(path, waypoints_per_traj=50):
    """Create dummy HDF5 matching new_setup_dataset_collect.py format."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # Linear interpolation with ~0.05 step size so path length is reasonable
    t = np.linspace(0, 1, waypoints_per_traj)
    joint_configs = np.column_stack([
        np.sin(t * np.pi) * 0.5,
        np.cos(t * np.pi) * 0.3,
        t * 0.4,
        -t * 0.2,
        np.sin(2 * t * np.pi) * 0.3,
        t * 0.5,
    ]).astype(np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("joint_configs", data=joint_configs)
        f.create_dataset("final_joint_config", data=joint_configs[-1])
        f.create_dataset("images", data=np.zeros((waypoints_per_traj, 720, 1280, 3), dtype=np.uint8))
    return path


def test_arc_length():
    # path: (0,0,...) -> (1,0,...) -> (1,1,...); segment lengths 1 and 1
    path = np.array([[0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0]])
    assert np.isclose(arc_length(path, 0, 2), 2.0)
    print("arc_length: OK")


def test_load_and_sample():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create 2 dummy HDF5 files
        create_dummy_h5(os.path.join(tmpdir, "demo_001.h5"), waypoints_per_traj=30)
        create_dummy_h5(os.path.join(tmpdir, "demo_002.h5"), waypoints_per_traj=40)

        trajectories = load_trajectories_from_h5(tmpdir)
        assert len(trajectories) == 2
        assert trajectories[0].shape == (30, 6)
        assert trajectories[1].shape == (40, 6)
        print("load_trajectories_from_h5: OK")

        points, tau_obs = sample_pairs_from_trajectories(
            trajectories, num_pairs=100, tau_min=0.001, tau_max=5.0
        )
        assert points.shape == (100, 12)
        assert tau_obs.shape == (100,)
        assert np.all(tau_obs >= 0.001)
        assert np.all(tau_obs <= 5.0)
        print("sample_pairs_from_trajectories: OK")
        print(f"  points shape: {points.shape}, tau_obs range: [{tau_obs.min():.4f}, {tau_obs.max():.4f}]")


if __name__ == "__main__":
    test_arc_length()
    test_load_and_sample()
    print("\nAll tests passed.")
