"""
Trajectory sampler for NTField training.

Loads RRT-generated trajectories from HDF5 files (produced by
hanwen_grasping/new_setup_dataset_collect.py) and samples (q_s, q_g) pairs
with observed travel time tau_obs (path length).
"""

import os
import glob
import numpy as np
import h5py


# Scale to match ntrl-demo NTField input space (same as speed_sampling_arm_normal)
SCALE = np.pi / 0.5


def load_trajectories_from_h5(data_dir):
    """
    Load trajectories from HDF5 files in the given directory.

    Expected HDF5 format (from new_setup_dataset_collect.py):
        - joint_configs: (N, 6) array of 6-DoF joint configs along trajectory (radians)
        - final_joint_config: (6,) goal config (optional, for validation)

    Args:
        data_dir: Path to directory containing *.h5 files (e.g. collected_data/)

    Returns:
        List of trajectories, each is (T, 6) numpy array in radians.
    """
    pattern = os.path.join(data_dir, "*.h5")
    h5_files = sorted(glob.glob(pattern))

    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in {data_dir}")

    trajectories = []
    for path in h5_files:
        try:
            with h5py.File(path, "r") as f:
                if "joint_configs" not in f:
                    print(f"Skip {path}: no 'joint_configs' dataset")
                    continue
                joint_configs = np.array(f["joint_configs"][:], dtype=np.float64)
                if joint_configs.ndim != 2 or joint_configs.shape[1] != 6:
                    print(f"Skip {path}: joint_configs shape {joint_configs.shape}")
                    continue
                if len(joint_configs) < 2:
                    print(f"Skip {path}: trajectory too short ({len(joint_configs)} waypoints)")
                    continue
                trajectories.append(joint_configs)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue

    return trajectories


def arc_length(path, start_idx, end_idx):
    """
    Compute arc length (path length) between waypoints start_idx and end_idx.

    tau_obs = sum(||q[k+1] - q[k]||) for k in [start_idx, end_idx-1]
    """
    segment = path[start_idx : end_idx + 1]
    if len(segment) < 2:
        return 0.0
    diffs = np.diff(segment, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def sample_pairs_from_trajectories(
    trajectories,
    num_pairs,
    scale=SCALE,
    tau_min=0.01,
    tau_max=2.0,
    rng=None,
):
    """
    Sample (q_s, q_g) pairs from trajectories and compute tau_obs (path length).

    Args:
        trajectories: List of (T, 6) arrays (joint configs in radians)
        num_pairs: Target number of pairs to sample
        scale: Normalization scale (config / scale) to match NTField input space
        tau_min: Minimum tau_obs (filter trivial pairs)
        tau_max: Maximum tau_obs (filter very long pairs)
        rng: Optional numpy RandomState or Generator for reproducibility

    Returns:
        points: (N, 12) array = [q_s_norm, q_g_norm] concatenated
        tau_obs: (N,) array of path lengths
    """
    if rng is None:
        rng = np.random.default_rng()

    points_list = []
    tau_list = []

    # Build list of (traj_idx, i, j) candidates for sampling
    candidates = []
    for traj_idx, path in enumerate(trajectories):
        T = len(path)
        for i in range(T - 1):
            for j in range(i + 1, T):
                tau = arc_length(path, i, j)
                if tau_min <= tau <= tau_max:
                    candidates.append((traj_idx, i, j, tau))

    if not candidates:
        raise ValueError(
            f"No valid pairs found. Try relaxing tau_min={tau_min}, tau_max={tau_max} "
            "or ensure trajectories have enough waypoints."
        )

    # Sample pairs (with replacement if num_pairs > n_candidates)
    n_candidates = len(candidates)
    replace = num_pairs > n_candidates
    indices = rng.choice(n_candidates, size=num_pairs, replace=replace)

    for idx in indices:
        traj_idx, i, j, tau = candidates[idx]
        path = trajectories[traj_idx]
        q_s = path[i]
        q_g = path[j]

        # Normalize to match NTField training space
        q_s_norm = q_s / scale
        q_g_norm = q_g / scale

        points_list.append(np.concatenate([q_s_norm, q_g_norm]))
        tau_list.append(tau)

    points = np.array(points_list, dtype=np.float32)
    tau_obs = np.array(tau_list, dtype=np.float32)

    return points, tau_obs
