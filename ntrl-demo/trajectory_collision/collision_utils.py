"""
Collision utilities for NTField training and planning with collision avoidance.

Provides obstacle KD-tree construction and arm obstacle distance computation.
"""

import os
import numpy as np
import torch
import igl
import torch_kdtree
import pytorch_kinematics as pk


def build_obstacle_kdtree(mesh_path, device="cuda", num_samples=50000):
    """
    Load obstacle mesh and build KD-tree for nearest-neighbor queries.

    Args:
        mesh_path: Path to .off or .obj mesh file.
        device: Device for tensors ('cuda' or 'cpu').
        num_samples: Number of points to sample on mesh surface (default 50000).
                     Ignored when mesh has no faces (point cloud).

    Returns:
        (kdtree, v_obs): KD-tree and obstacle point cloud on device.
    """
    v, f = igl.read_triangle_mesh(mesh_path)
    if f.size > 0:
        bary, FI, _ = igl.random_points_on_mesh(num_samples, v, f)
        face_verts = v[f[FI], :]
        v_obs = np.sum(bary[..., np.newaxis] * face_verts, axis=1)
    else:
        v_obs = np.asarray(v, dtype=np.float32)
    v_obs = torch.tensor(v_obs, dtype=torch.float32, device=device)
    kdtree = torch_kdtree.build_kd_tree(v_obs)
    return kdtree, v_obs


def build_arm_chain(base_path=None):
    """
    Build UR5 kinematic chain and sphere mesh list for collision checking.

    Args:
        base_path: Base path to ntrl-demo (contains datasets/arm/UR5).
                   If None, uses ntrl-demo root (parent of trajectory_collision).

    Returns:
        (chain, mesh_list): pytorch_kinematics chain and list of sphere tensors.
    """
    if base_path is None:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(base_path, "datasets", "arm", "UR5")
    end_effect = "wrist_3_link"
    d = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    chain = pk.build_serial_chain_from_urdf(
        open(os.path.join(out_path, "ur5e.urdf")).read(), end_effect
    )
    chain = chain.to(dtype=dtype, device=d)
    th_batch = torch.rand(1, 6, device=d)
    tg_batch = chain.forward_kinematics(th_batch, end_only=False)
    mesh_list = []
    iter_count = 0
    for tg in tg_batch:
        if iter_count > 1 and iter_count < 8:
            ball_path = os.path.join(out_path, "meshes", "sphere", "sphere", f"{tg}.npy")
            ball_list = np.load(ball_path)
            mesh_list.append(torch.tensor(ball_list, dtype=torch.float32, device=d))
        iter_count += 1
    return chain, mesh_list


def arm_obstacle_distance_batch(th_batch, chain, mesh_list, kdtree, v_obs, device="cuda"):
    """
    Compute minimum obstacle distance and its gradient w.r.t. joint configs.

    Args:
        th_batch: (N, 6) joint configs in radians.
        chain: UR5 kinematic chain from build_arm_chain.
        mesh_list: Sphere mesh list from build_arm_chain.
        kdtree: KD-tree from build_obstacle_kdtree.
        v_obs: Obstacle point cloud from build_obstacle_kdtree.
        device: Device for output tensors.

    Returns:
        (distance, normal): distance (N,), normal (N, 6) gradient in config space.
    """
    from . import collision_arm

    th_batch = th_batch.to(device)
    distance, normal = collision_arm.arm_obstacle_distance(
        th_batch, chain, mesh_list, kdtree, v_obs
    )
    return distance, normal
