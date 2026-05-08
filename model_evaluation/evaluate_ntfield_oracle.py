#!/usr/bin/env python3
"""
NTField oracle evaluation — uses ground-truth z_goal from the test dataset.

Measures the ceiling performance: how well does NTField plan when given
the perfect teacher latent, with no student prediction error at all.

Results:
  <output_dir>/
    results.jsonl   — one line per record (finger_midpoint→target XY distance + Z diff)
    summary.json    — aggregate success rate + planner stats

Example:
  python evaluate_ntfield_oracle.py \\
    --test_dataset  /path/to/test_run_pt_shards \\
    --ntfield_checkpoint /path/to/teacher_model.pt \\
    --output_dir    /path/to/output/eval_ntfield_oracle
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# ── Path setup ────────────────────────────────────────────────────────────────
_PI_VLA_ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HANWEN_GRASPING_ROOT = os.path.join(_PI_VLA_ROOT, "hanwen_grasping")
_COLLECT_DATA_DIR    = os.path.join(HANWEN_GRASPING_ROOT, "collect_data")
_UTIL_DIR            = os.path.join(_COLLECT_DATA_DIR, "util")
_GRASP_UTIL_DIR      = os.path.join(_COLLECT_DATA_DIR, "grasp_util")
_NTRL_DEMO           = os.path.join(_PI_VLA_ROOT, "ntrl-demo")
_STUDENT_DIR         = os.path.join(_PI_VLA_ROOT, "student_model_training")

for _p in (HANWEN_GRASPING_ROOT, _UTIL_DIR, _GRASP_UTIL_DIR,
           _PI_VLA_ROOT, _NTRL_DEMO, _STUDENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# Indices follow hanwen_grasping/assets/urdf/ycb/object_urdf_grasp.txt
OBJECT_HEIGHTS_M: Dict[int, float] = {
    1: 0.176887,   # 004_sugar_box
    3: 0.192159,   # 006_mustard_bottle
    5: 0.037270,   # 011_banana
}
OBJECT_HEIGHT_DEFAULT_M = 0.10


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_test_shards(dataset_dir: str) -> List[Dict]:
    import torch

    dataset_dir = Path(dataset_dir)
    manifest_path = dataset_dir / "manifest.pt"

    if manifest_path.exists():
        manifest    = torch.load(manifest_path, map_location="cpu")
        shard_paths = manifest.get("shards", [])
        print(f"Manifest: {manifest.get('num_samples')} samples, "
              f"{manifest.get('num_shards')} shards, "
              f"z_dim={manifest.get('z_dim')}")
    else:
        shard_paths = sorted(str(p) for p in dataset_dir.glob("test_shard_*.pt"))
        print(f"No manifest, discovered {len(shard_paths)} shards.")

    if not shard_paths:
        raise FileNotFoundError(f"No shards found under {dataset_dir}")

    records = []
    for sp in shard_paths:
        records.extend(torch.load(sp, map_location="cpu"))
    print(f"Loaded {len(records)} records.")
    return records


# ── NTField planner ───────────────────────────────────────────────────────────

def ntfield_plan_latent(
    teacher_network,
    q_start:    np.ndarray,
    z_goal:     np.ndarray,
    step_size:  float = 0.02,
    max_steps:  int   = 200,
    tol:        float = 0.01,
    device:     str   = "cuda",
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    import torch

    from planning.gradient_planner_trajectory import SCALE as NTFIELD_SCALE

    def _dist(qn, zg):
        with torch.no_grad():
            d, _, _ = teacher_network.out_with_goal_latent(qn.detach(), zg)
            return float(d.item())

    def _step(q_t, zg, stp):
        q_t  = q_t.detach().requires_grad_(True)
        dist, _, coords_out = teacher_network.out_with_goal_latent(q_t, zg)
        d_pre = float(dist.item())
        if d_pre < tol:
            return q_t.detach(), d_pre, True
        grad_start = torch.autograd.grad(dist, coords_out)[0][:, :6]
        with torch.no_grad():
            q_next = q_t - stp * grad_start
        d_post = _dist(q_next, zg)
        return q_next.detach(), d_post, d_post < tol

    q_start    = np.asarray(q_start, dtype=np.float32).reshape(-1)
    q_norm     = q_start / NTFIELD_SCALE
    q_t        = torch.tensor(q_norm, dtype=torch.float32, device=device).unsqueeze(0)
    z_goal_t   = torch.tensor(
        np.asarray(z_goal, dtype=np.float32).reshape(1, -1), device=device
    )

    path       = [q_norm.copy()]
    final_dist = None
    converged  = False
    stopped    = "max_steps"

    for _ in range(max_steps):
        q_t, final_dist, converged = _step(q_t, z_goal_t, step_size)
        path.append(q_t.detach().cpu().numpy()[0].copy())
        if converged:
            stopped = "latent_tol"
            break

    return (
        [p * NTFIELD_SCALE for p in path],
        {"final_latent_dist": final_dist, "stopped": stopped, "path_len": len(path)},
    )


# ── Isaac Gym scene helpers ───────────────────────────────────────────────────

def _build_env(gym, sim, ur5e_asset, table_asset, table_pose, ur5e_pose,
               object_assets, object_collision_files, object_offset, asset_root,
               all_object_ids, all_object_locations, use_viewer):
    from isaacgym import gymapi
    import fcl
    from obj_reader import obj_reader

    spacing   = 2
    env       = gym.create_env(sim,
                               gymapi.Vec3(-spacing, -spacing, 0),
                               gymapi.Vec3( spacing,  spacing, 0), 1)
    ur_handle = gym.create_actor(env, ur5e_asset, ur5e_pose, "ur5e", 0, 32767)
    gym.create_actor(env, table_asset, table_pose, "table", 0, 1)

    spj = gym.find_actor_dof_handle(env, ur_handle, "shoulder_pan_joint")
    slj = gym.find_actor_dof_handle(env, ur_handle, "shoulder_lift_joint")
    ej  = gym.find_actor_dof_handle(env, ur_handle, "elbow_joint")
    wj1 = gym.find_actor_dof_handle(env, ur_handle, "wrist_1_joint")
    wj2 = gym.find_actor_dof_handle(env, ur_handle, "wrist_2_joint")
    wj3 = gym.find_actor_dof_handle(env, ur_handle, "wrist_3_joint")

    obj_handles  = []
    obj_col_lib  = []
    obj_status   = []

    for k, (obj_id, xyz) in enumerate(zip(all_object_ids, all_object_locations)):
        col_mesh = obj_reader(asset_root + object_collision_files[obj_id])
        col_mesh.set_scale(1.0)
        col_mesh.add_offset(object_offset[obj_id])
        verts, tris = col_mesh.get_bounding_box_mesh()
        m = fcl.BVHModel()
        m.beginModel(len(verts), len(tris))
        m.addSubModel(verts, tris)
        m.endModel()

        pose   = gymapi.Transform()
        pose.p = gymapi.Vec3(float(xyz[0]), float(xyz[1]), float(xyz[2]))
        handle = gym.create_actor(env, object_assets[obj_id], pose,
                                  f"obj_{k}", 0, 2 ** (k + 1), k + 1)
        gym.set_actor_scale(env, handle, 1.0)
        obj_handles.append(handle)
        obj_col_lib.append(m)
        obj_status.append([col_mesh.get_center(), col_mesh.get_bounding_box()])

    viewer = None
    if use_viewer:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())

    return env, ur_handle, spj, slj, ej, wj1, wj2, wj3, \
           obj_handles, obj_col_lib, obj_status, viewer


def _warmup(gym, sim, env, spj, slj, ej, wj1, wj2, wj3, viewer,
            obj_handles, obj_status, obj_col_lib, n_steps=2000):
    from scipy.spatial.transform import Rotation as R
    from isaacgym import gymapi
    import fcl

    real_pos      = False
    _HOME         = [0.7, -2.0, 2.5, -0.3, 0.7, 0.0]
    object_mesh   = []          # populated at t==999
    flex_col_models = []

    for t in range(n_steps):
        if not real_pos:
            gym.set_dof_target_position(env, spj, 0)
            gym.set_dof_target_position(env, slj, -math.pi / 2)
            gym.set_dof_target_position(env, ej,  0)
            gym.set_dof_target_position(env, wj1, -math.pi / 2)
            gym.set_dof_target_position(env, wj2, 0)
            gym.set_dof_target_position(env, wj3, 0)
            real_pos = True

        if t == 999:
            for ii, element in enumerate(obj_handles):
                states      = gym.get_actor_rigid_body_states(env, element, 1)
                rotation    = np.array(np.array(states[0][0][1]).item())
                translation = np.array(np.array(states[0][0][0]).item())
                obj_status[ii][0] += translation
                r1 = R.from_quat(rotation)
                tf = fcl.Transform(r1.as_matrix(), translation)
                flex_col_models.append([fcl.CollisionObject(obj_col_lib[ii], tf), 0])
                object_mesh.append({
                    "center":      obj_status[ii][0].copy(),      # (3,) world XYZ
                    "translation": translation.copy(),
                })

        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        if viewer:
            gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    for _ in range(30):
        for dof, val in zip([spj, slj, ej, wj1, wj2, wj3], _HOME):
            gym.set_dof_target_position(env, dof, val)
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        if viewer:
            gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    return object_mesh, flex_col_models


def _execute_path_and_get_ee(
    gym, sim, env, ur_handle,
    spj, slj, ej, wj1, wj2, wj3,
    viewer,
    path_raw: List[np.ndarray],
    settle_steps: int = 60,
) -> Tuple[np.ndarray, str]:
    """
    Step the arm through path_raw joint-by-joint, settle, then return
    world-space XYZ of the wrist-proxy end-effector.
    Matches the MDN pipeline's wrist-proxy candidate list:
      wrist_3_link → tool0 → ee_link → robotiq_arg2f_base_link → last rigid body.
    Finger midpoint is computed separately in the main loop.
    Returns (ee_xyz, ee_link_name).
    """
    from isaacgym import gymapi

    dofs = [spj, slj, ej, wj1, wj2, wj3]

    for q in path_raw:
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        for dof, val in zip(dofs, q[:6]):
            gym.set_dof_target_position(env, dof, float(val))
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        if viewer:
            gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    for _ in range(settle_steps):
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        if viewer:
            gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    rb_ur = gym.get_actor_rigid_body_dict(env, ur_handle)
    states = gym.get_actor_rigid_body_states(env, ur_handle, gymapi.STATE_POS)

    # Wrist-proxy: same candidate order as MDN pipeline
    for cand in ("wrist_3_link", "tool0", "ee_link", "robotiq_arg2f_base_link"):
        if cand in rb_ur:
            T_ee = gymapi.Transform.from_buffer(states["pose"][int(rb_ur[cand])])
            return np.array([T_ee.p.x, T_ee.p.y, T_ee.p.z], dtype=np.float64), cand

    # Fallback: last rigid body
    num_bodies = gym.get_actor_rigid_body_count(env, ur_handle)
    ee_pos = np.array(np.array(states[num_bodies - 1][0][0]).item(), dtype=np.float64)
    return ee_pos, "wrist_fallback"

def _ee_to_target_dist(ee_pos: np.ndarray, object_mesh_entry: dict) -> float:
    """Euclidean distance from EE to the settled object center (world XYZ)."""
    target_xyz = np.asarray(object_mesh_entry["center"], dtype=np.float64)
    return float(np.linalg.norm(ee_pos - target_xyz))


def _get_rigid_body_world_pos(gym, env, actor_handle, body_name: str) -> Optional[np.ndarray]:
    """Return world XYZ for actor rigid body by name, or None if missing."""
    from isaacgym import gymapi

    rb_dict = gym.get_actor_rigid_body_dict(env, actor_handle)
    rb_idx = rb_dict.get(body_name, None)
    if rb_idx is None:
        return None
    rb_states = gym.get_actor_rigid_body_states(env, actor_handle, gymapi.STATE_POS)
    pose = rb_states["pose"][int(rb_idx)]
    p = pose["p"]
    return np.array([p["x"], p["y"], p["z"]], dtype=np.float64)


def _between_fingers_xy_z(
    left: np.ndarray,
    right: np.ndarray,
    obj_root: np.ndarray,
    *,
    axis_margin_m: float,
    lateral_thresh_m: float,
    z_thresh_m: float,
    obj_half_height_m: float,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Stage 1: XY axis check (finger segment in XY, object root XY vs axis).
    Stage 2: |finger_mid_z - grasp_center_z| <= z_thresh,
            grasp_center_z = root_z + half_height (standard height offset from root).

    Overall success iff both stages pass.
    """
    l = np.asarray(left, dtype=np.float64).reshape(3)
    r = np.asarray(right, dtype=np.float64).reshape(3)
    root = np.asarray(obj_root, dtype=np.float64).reshape(3)
    finger_mid = 0.5 * (l + r)

    half_h = float(obj_half_height_m)
    grasp_center_z = root[2] + half_h

    left_xy = l[:2]
    right_xy = r[:2]
    obj_xy = root[:2]
    seg_xy = right_xy - left_xy
    gap_xy = float(np.linalg.norm(seg_xy))

    meta: Dict[str, Any] = {
        "xy_axis_success": False,
        "z_height_success": False,
        "finger_gap_m": gap_xy,
        "finger_axis_proj_m": float("nan"),
        "finger_axis_proj_norm": float("nan"),
        "finger_lateral_dist_xy_m": float("nan"),
        "finger_z_diff_m": float("nan"),
        "object_root_pos": root.tolist(),
        "object_grasp_target_z": float(grasp_center_z),
        "finger_midpoint_z": float(finger_mid[2]),
    }

    if gap_xy < 1e-9:
        return False, "degenerate_finger_gap", meta

    axis_xy = seg_xy / gap_xy
    rel_xy = obj_xy - left_xy
    axis_proj = float(np.dot(rel_xy, axis_xy))
    axis_proj_norm = axis_proj / gap_xy
    lateral_vec_xy = rel_xy - axis_proj * axis_xy
    lateral_dist_xy = float(np.linalg.norm(lateral_vec_xy))

    meta["finger_axis_proj_m"] = axis_proj
    meta["finger_axis_proj_norm"] = axis_proj_norm
    meta["finger_lateral_dist_xy_m"] = lateral_dist_xy

    inside_axis = (-axis_margin_m <= axis_proj <= gap_xy + axis_margin_m)
    inside_lateral = lateral_dist_xy <= lateral_thresh_m
    xy_success = bool(inside_axis and inside_lateral)
    meta["xy_axis_success"] = xy_success

    if not xy_success:
        reason = (
            f"xy_fail: proj={axis_proj:.4f} "
            f"(need 0–{gap_xy:.4f}m ±{axis_margin_m}m), "
            f"lateral={lateral_dist_xy:.4f}m (thresh={lateral_thresh_m}m)"
        )
        return False, reason, meta

    z_diff = abs(float(finger_mid[2]) - float(grasp_center_z))
    meta["finger_z_diff_m"] = z_diff
    z_success = z_diff <= z_thresh_m
    meta["z_height_success"] = z_success

    if not z_success:
        reason = (
            f"z_fail: finger_z={finger_mid[2]:.4f}m "
            f"obj_grasp_z={grasp_center_z:.4f}m "
            f"diff={z_diff:.4f}m (thresh={z_thresh_m}m)"
        )
        return False, reason, meta

    return True, "ok", meta

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    from isaacgym import gymapi, gymutil
    import torch
    from trajectory_evaluation.ntfield.eval_trajectory_ntfield import (
        load_network_and_function,
    )
    from trajectory_evaluation.comparison.run_rrt_ntfield_benchmark import (
        TABLE_DIMS_X, TABLE_DIMS_Y, TABLE_DIMS_Z, sim_dt,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test_dataset",       required=True)
    parser.add_argument("--ntfield_checkpoint", required=True)
    parser.add_argument("--output_dir",         required=True)
    parser.add_argument("--ntfield_step_size",  type=float, default=0.02)
    parser.add_argument("--ntfield_max_steps",  type=int,   default=200)
    parser.add_argument("--ntfield_tol",        type=float, default=0.01)
    parser.add_argument("--ee_success_thresh",  type=float, default=0.08,
                        help="EE-to-target distance threshold for success (metres). Default 0.08m.")
    parser.add_argument(
        "--finger_mid_xy_success_thresh_m",
        type=float,
        default=0.08,
        help="Finger-midpoint→target XY distance threshold for success (metres). Default 0.08m.",
    )
    parser.add_argument(
        "--finger_mid_z_success_thresh_m",
        type=float,
        default=0.05,
        help="Abs(finger-midpoint Z diff) threshold for success (metres). Default 0.05m.",
    )
    parser.add_argument("--between_axis_margin_m", type=float, default=0.005,
                        help="Allowed projection margin beyond finger segment ends (m).")
    parser.add_argument("--between_lateral_thresh_m", type=float, default=0.02,
                        help="Max perpendicular distance from finger axis for between-fingers success (m).")
    parser.add_argument(
        "--between_z_thresh_m",
        type=float,
        default=0.05,
        help=(
            "Max |finger_midpoint_z - object_grasp_center_z| for Z grasp success (m). "
            "Default 0.05m."
        ),
    )
    parser.add_argument("--ntfield_device", default="cuda:0")
    parser.add_argument("--max_records",        type=int,   default=-1,
                        help="Evaluate only the first N records (default: all).")
    parser.add_argument("--physx_cpu",          action="store_true")
    parser.add_argument("--use_viewer",         action="store_true")
    parser.add_argument("--no_isaac_hard_exit", action="store_true")
    args, argv_remainder = parser.parse_known_args()

    # ── Output ────────────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    summary_path = out_dir / "summary.json"

    # ── Isaac Gym ─────────────────────────────────────────────────────────────
    argv_gym = list(argv_remainder)
    if not args.use_viewer and "--headless" not in argv_gym:
        argv_gym.append("--headless")
    sys.argv = [sys.argv[0]] + argv_gym

    gym      = gymapi.acquire_gym()
    gym_args = gymutil.parse_arguments(description="ntfield_oracle",
                                       headless=True, custom_parameters=[])
    gym_args.headless = not args.use_viewer
    if args.physx_cpu:
        gym_args.use_gpu = False

    table_dims  = gymapi.Vec3(TABLE_DIMS_X, TABLE_DIMS_Y, TABLE_DIMS_Z)
    sim_params  = gymapi.SimParams()
    sim_params.substeps  = 2
    sim_params.dt        = sim_dt
    sim_params.up_axis   = gymapi.UP_AXIS_Z
    sim_params.gravity   = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.physx.solver_type             = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads             = gym_args.num_threads
    sim_params.physx.use_gpu                 = gym_args.use_gpu
    sim_params.use_gpu_pipeline              = False

    sim = gym.create_sim(gym_args.compute_device_id, gym_args.graphics_device_id,
                         gym_args.physics_engine, sim_params)
    if sim is None:
        raise SystemExit("Failed to create sim")

    plane_params        = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)

    # ── NTField — load after sim ──────────────────────────────────────────────
    dev_nt = torch.device(
        "cpu" if args.ntfield_device == "cpu" or not torch.cuda.is_available()
        else args.ntfield_device
    )
    nt_net, _ = load_network_and_function(
        os.path.abspath(args.ntfield_checkpoint), None, dev_nt, dim=6
    )
    print(f"NTField loaded on {dev_nt}")

    # ── Load assets once ──────────────────────────────────────────────────────
    _cwd_prev = os.getcwd()
    os.chdir(HANWEN_GRASPING_ROOT)
    asset_root = "./assets/"

    ao = gymapi.AssetOptions()
    ao.fix_base_link          = True
    ao.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)
    ao.mesh_normal_mode       = gymapi.COMPUTE_PER_VERTEX
    ao.use_mesh_materials     = True

    ur5e_asset  = gym.load_asset(sim, asset_root,
                                 "urdf/ur5e/ur5e_mimic_real_gripper_test.urdf", ao)
    table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, ao)

    pfx = "urdf/ycb/"
    with open(asset_root + "urdf/ycb/object_urdf_grasp.txt") as f:
        object_asset_files = [pfx + l.strip() for l in f if l.strip()]
    with open(asset_root + "urdf/ycb/object_collision_grasp.txt") as f:
        object_collision_files = [pfx + l.strip() for l in f if l.strip()]
    with open(asset_root + "urdf/ycb/object_offset_grasp.txt") as f:
        object_offset = [[float(x) for x in l.strip().split()] for l in f if l.strip()]

    ao.fix_base_link = False
    object_assets    = [gym.load_asset(sim, asset_root, ob, ao)
                        for ob in object_asset_files]

    ur5e_pose    = gymapi.Transform()
    ur5e_pose.p  = gymapi.Vec3(0, 0, 0)
    ur5e_pose.r  = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0.5 * math.pi)
    table_pose   = gymapi.Transform()
    table_pose.p = gymapi.Vec3(table_dims.x * 0.5 + 0.3, 0.0, table_dims.z * 0.5)

    gym.set_light_parameters(sim, 0, gymapi.Vec3(0.3,0.3,0.3),
                             gymapi.Vec3(1,1,1), gymapi.Vec3(-1,0,0))
    gym.set_light_parameters(sim, 1, gymapi.Vec3(0.3,0.3,0.3),
                             gymapi.Vec3(1,1,1), gymapi.Vec3(1,0,0))

    # ── Load records ──────────────────────────────────────────────────────────
    records = load_test_shards(args.test_dataset)
    if args.max_records > 0:
        records = records[:args.max_records]
    print(f"Evaluating {len(records)} records with ground-truth z_goal.")

    # ── Evaluation loop ───────────────────────────────────────────────────────
    n_mid_success     = 0
    n_ee_success      = 0
    final_dist_list   = []
    path_len_list     = []
    ee_dist_list      = []     # physical EE-to-target distances
    finger_mid_to_target_xy_list: List[float] = []
    finger_mid_to_target_z_diff_list: List[float] = []
    stop_counts: Dict[str, int] = {}
    all_results: List[Dict]     = []

    with open(results_path, "w") as results_f:

        for rec_idx, record in enumerate(tqdm(records, desc="NTField oracle")):
            all_object_ids       = record["all_object_ids"]
            all_object_locations = record["all_object_locations"]
            z_goal_true          = record["z_goal"].float().numpy()   # teacher latent
            source_file          = record.get("source_file", "")
            seed                 = record.get("seed", -1)
            object_name          = record.get("object_name", "object")
            target_obj_idx       = record.get("target_obj_idx", 0)

            result: Dict[str, Any] = {
                "record_idx":        rec_idx,
                "source_file":       source_file,
                "seed":              seed,
                "object_name":       object_name,
                "target_obj_idx":    target_obj_idx,
                "success":           False,
                "ee_success":        False,
                "final_latent_dist": None,
                "path_len":          None,
                "ee_pos":            None,
                "ee_link":           None,
                "ee_dist_m":         None,
                "ee_success_thresh": args.ee_success_thresh,
                "finger_mid_xy_success_thresh_m": args.finger_mid_xy_success_thresh_m,
                "finger_mid_z_success_thresh_m": args.finger_mid_z_success_thresh_m,
                "finger_midpoint_to_target_xy_distance_m": None,
                "finger_midpoint_to_target_z_diff_m": None,
                "finger_mid_success": False,
                "between_fingers_success": False,
                "between_fingers_reason": None,
                "xy_axis_success": None,
                "z_height_success": None,
                "left_finger_pos": None,
                "right_finger_pos": None,
                "object_root_pos": None,
                "finger_gap_m": None,
                "finger_axis_proj_m": None,
                "finger_axis_proj_norm": None,
                "finger_lateral_dist_xy_m": None,
                "between_axis_margin_m": args.between_axis_margin_m,
                "between_lateral_thresh_m": args.between_lateral_thresh_m,
                "between_z_thresh_m": args.between_z_thresh_m,
            }
            stop_reason = "error"

            try:
                env, ur_handle, spj, slj, ej, wj1, wj2, wj3, \
                obj_handles, obj_col_lib, obj_status, viewer = _build_env(
                    gym, sim, ur5e_asset, table_asset, table_pose, ur5e_pose,
                    object_assets, object_collision_files, object_offset, asset_root,
                    all_object_ids, all_object_locations, args.use_viewer,
                )

                # _warmup now returns settled mesh state
                object_mesh, _ = _warmup(
                    gym, sim, env, spj, slj, ej, wj1, wj2, wj3, viewer,
                    obj_handles, obj_status, obj_col_lib,
                )

                dof_state    = gym.get_actor_dof_states(env, ur_handle, gymapi.STATE_POS)
                q_start_live = np.array(dof_state["pos"][:6], dtype=np.float64)

                path_raw, meta = ntfield_plan_latent(
                    nt_net,
                    q_start_live,
                    z_goal_true,
                    step_size = args.ntfield_step_size,
                    max_steps = args.ntfield_max_steps,
                    tol       = args.ntfield_tol,
                    device    = str(dev_nt),
                )
                stop_reason = str(meta.get("stopped", "unknown"))

                # ── Execute path + measure EE distance ────────────────────
                ee_pos, ee_link = _execute_path_and_get_ee(
                    gym, sim, env, ur_handle,
                    spj, slj, ej, wj1, wj2, wj3, viewer,
                    path_raw,
                )

                # target object index into both the warmed-up mesh list and
                # the live Isaac Gym rigid-body actors.
                mesh_idx = min(target_obj_idx, len(object_mesh) - 1)

                # Use *live* simulated root position as the target reference
                # (more accurate than mesh["center"], which is tied to warmup-time).
                st_obj_tgt = gym.get_actor_rigid_body_states(
                    env, obj_handles[mesh_idx], gymapi.STATE_POS
                )
                T_root = gymapi.Transform.from_buffer(st_obj_tgt["pose"][0])
                obj_root_xyz = np.array(
                    [T_root.p.x, T_root.p.y, T_root.p.z],
                    dtype=np.float64,
                )
                object_center = obj_root_xyz

                # Physical EE distance (wrist proxy) to the simulated object root.
                ee_dist = float(np.linalg.norm(np.asarray(ee_pos, dtype=np.float64) - object_center))

                left_finger_pos = _get_rigid_body_world_pos(
                    gym, env, ur_handle, "left_inner_finger"
                )
                right_finger_pos = _get_rigid_body_world_pos(
                    gym, env, ur_handle, "right_inner_finger"
                )
                finger_midpoint_to_target_xy_distance_m = None
                finger_midpoint_to_target_z_diff_m = None
                if left_finger_pos is not None and right_finger_pos is not None:
                    finger_mid = 0.5 * (left_finger_pos + right_finger_pos)
                    dxy = finger_mid[:2] - object_center[:2]
                    finger_midpoint_to_target_xy_distance_m = float(
                        np.hypot(float(dxy[0]), float(dxy[1]))
                    )
                    finger_midpoint_to_target_z_diff_m = float(
                        finger_mid[2] - object_center[2]
                    )
                between_success = False
                between_reason: Optional[str] = None
                between_meta: Dict[str, Any] = {}
                xy_ok: Optional[bool] = None
                z_ok: Optional[bool] = None

                oid_ix = min(target_obj_idx, len(all_object_ids) - 1)
                asset_id = int(all_object_ids[oid_ix])
                half_height_m = float(
                    OBJECT_HEIGHTS_M.get(asset_id, OBJECT_HEIGHT_DEFAULT_M) * 0.5
                )

                if left_finger_pos is None or right_finger_pos is None:
                    between_reason = "finger_links_missing"
                    xy_ok, z_ok = None, None
                else:
                    between_success, between_reason, between_meta = _between_fingers_xy_z(
                        left_finger_pos,
                        right_finger_pos,
                        obj_root_xyz,
                        axis_margin_m=args.between_axis_margin_m,
                        lateral_thresh_m=args.between_lateral_thresh_m,
                        z_thresh_m=args.between_z_thresh_m,
                        obj_half_height_m=half_height_m,
                    )
                    xy_ok = bool(between_meta.get("xy_axis_success", False))
                    z_ok = bool(between_meta.get("z_height_success", False))

                # Success definitions:
                # - EE success: wrist-proxy EE reached target threshold (legacy / diagnostic)
                # - Mid success: finger midpoint XY within thresh AND |Z diff| within thresh
                ee_success = ee_dist <= args.ee_success_thresh
                mid_success = (
                    finger_midpoint_to_target_xy_distance_m is not None
                    and finger_midpoint_to_target_z_diff_m is not None
                    and float(finger_midpoint_to_target_xy_distance_m)
                    <= float(args.finger_mid_xy_success_thresh_m)
                    and abs(float(finger_midpoint_to_target_z_diff_m))
                    <= float(args.finger_mid_z_success_thresh_m)
                )

                result.update({
                    "ee_success":        ee_success,
                    "success":           mid_success,         # primary metric
                    "final_latent_dist": meta["final_latent_dist"],
                    "path_len":          meta["path_len"],
                    "ee_pos":            ee_pos.tolist(),
                    "ee_link":           ee_link,
                    "ee_dist_m":         ee_dist,
                    "ee_success_thresh": args.ee_success_thresh,
                    "finger_midpoint_to_target_xy_distance_m": finger_midpoint_to_target_xy_distance_m,
                    "finger_midpoint_to_target_z_diff_m": finger_midpoint_to_target_z_diff_m,
                    "finger_mid_success": mid_success,
                    "between_fingers_success": between_success,
                    "between_fingers_reason": between_reason,
                    "xy_axis_success": xy_ok,
                    "z_height_success": z_ok,
                    "object_asset_id_for_height": asset_id,
                    "object_half_height_m": half_height_m,
                    "left_finger_pos": None if left_finger_pos is None else left_finger_pos.tolist(),
                    "right_finger_pos": None if right_finger_pos is None else right_finger_pos.tolist(),
                    "object_root_pos": obj_root_xyz.tolist(),
                    "finger_gap_m": between_meta.get("finger_gap_m"),
                    "finger_axis_proj_m": between_meta.get("finger_axis_proj_m"),
                    "finger_axis_proj_norm": between_meta.get("finger_axis_proj_norm"),
                    "finger_lateral_dist_xy_m": between_meta.get("finger_lateral_dist_xy_m"),
                    "finger_z_diff_m": between_meta.get("finger_z_diff_m"),
                    "object_grasp_target_z": between_meta.get("object_grasp_target_z"),
                    "finger_midpoint_z": between_meta.get("finger_midpoint_z"),
                    "between_axis_margin_m": args.between_axis_margin_m,
                    "between_lateral_thresh_m": args.between_lateral_thresh_m,
                    "between_z_thresh_m": args.between_z_thresh_m,
                })
                if ee_success:
                    n_ee_success += 1
                if mid_success:
                    n_mid_success += 1
                if meta["final_latent_dist"] is not None:
                    final_dist_list.append(meta["final_latent_dist"])
                path_len_list.append(meta["path_len"])
                ee_dist_list.append(ee_dist)
                if finger_midpoint_to_target_xy_distance_m is not None:
                    finger_mid_to_target_xy_list.append(
                        finger_midpoint_to_target_xy_distance_m
                    )
                if finger_midpoint_to_target_z_diff_m is not None:
                    finger_mid_to_target_z_diff_list.append(
                        finger_midpoint_to_target_z_diff_m
                    )

            except Exception as e:
                result["error"] = str(e)
                tqdm.write(f"  [ERROR] record {rec_idx}: {e}")

            stop_counts[stop_reason] = stop_counts.get(stop_reason, 0) + 1

            all_results.append(result)
            results_f.write(json.dumps(result) + "\n")
            results_f.flush()

            tqdm.write(
                f"  [{rec_idx:04d}] {object_name:20s} | "
                f"success={result['success']} | "
                f"ee_success={result['ee_success']} | "
                f"grasp={result['between_fingers_success']} "
                f"(xy={result.get('xy_axis_success')} z={result.get('z_height_success')}) | "
                f"stop={stop_reason} | "
                f"ee_dist={result['ee_dist_m'] if result['ee_dist_m'] is not None else float('nan'):.4f}m | "
                f"fm_xy={result['finger_midpoint_to_target_xy_distance_m'] if result['finger_midpoint_to_target_xy_distance_m'] is not None else float('nan'):.4f}m | "
                f"fm_dz={result['finger_midpoint_to_target_z_diff_m'] if result['finger_midpoint_to_target_z_diff_m'] is not None else float('nan'):+.4f}m | "
                f"latent_dist={result['final_latent_dist']}"
            )

    # ── Summary ───────────────────────────────────────────────────────────────
    n_total     = len(all_results)
    n_errors    = sum(1 for r in all_results if "error" in r)
    n_evaluated = n_total - n_errors

    summary = {
        "mode":                    "ntfield_oracle",
        "ntfield_checkpoint":      args.ntfield_checkpoint,
        "test_dataset":            args.test_dataset,
        "ntfield_step_size":       args.ntfield_step_size,
        "ntfield_max_steps":       args.ntfield_max_steps,
        "ntfield_tol":             args.ntfield_tol,
        "ee_success_thresh_m":     args.ee_success_thresh,
        "finger_mid_xy_success_thresh_m": args.finger_mid_xy_success_thresh_m,
        "finger_mid_z_success_thresh_m": args.finger_mid_z_success_thresh_m,
        "between_axis_margin_m":   args.between_axis_margin_m,
        "between_lateral_thresh_m":args.between_lateral_thresh_m,
        "between_z_thresh_m":      args.between_z_thresh_m,
        "n_total":                 n_total,
        "n_evaluated":             n_evaluated,
        "n_errors":                n_errors,
        # Primary metric — finger midpoint XY + Z thresholds
        "n_success":               n_mid_success,
        "success_rate":            n_mid_success / max(n_evaluated, 1),
        # Diagnostic metric — physical EE distance (wrist proxy)
        "n_ee_success":            n_ee_success,
        "ee_success_rate":         n_ee_success / max(n_evaluated, 1),
        "ee_dist_mean_m":          float(np.mean(ee_dist_list))   if ee_dist_list else None,
        "ee_dist_median_m":        float(np.median(ee_dist_list)) if ee_dist_list else None,
        "ee_dist_std_m":           float(np.std(ee_dist_list))    if ee_dist_list else None,
        "finger_midpoint_to_target_xy_distance_mean_m": (
            float(np.mean(finger_mid_to_target_xy_list))
            if finger_mid_to_target_xy_list else None
        ),
        "finger_midpoint_to_target_xy_distance_median_m": (
            float(np.median(finger_mid_to_target_xy_list))
            if finger_mid_to_target_xy_list else None
        ),
        "finger_midpoint_to_target_xy_distance_std_m": (
            float(np.std(finger_mid_to_target_xy_list))
            if finger_mid_to_target_xy_list else None
        ),
        "finger_midpoint_to_target_z_diff_mean_m": (
            float(np.mean(finger_mid_to_target_z_diff_list))
            if finger_mid_to_target_z_diff_list else None
        ),
        "finger_midpoint_to_target_z_diff_median_m": (
            float(np.median(finger_mid_to_target_z_diff_list))
            if finger_mid_to_target_z_diff_list else None
        ),
        "finger_midpoint_to_target_z_diff_std_m": (
            float(np.std(finger_mid_to_target_z_diff_list))
            if finger_mid_to_target_z_diff_list else None
        ),
        "n_between_fingers_success": sum(1 for r in all_results if r.get("between_fingers_success")),
        "between_fingers_success_rate":
            sum(1 for r in all_results if r.get("between_fingers_success")) / max(n_evaluated, 1),
        "final_latent_dist_mean":  float(np.mean(final_dist_list))   if final_dist_list else None,
        "final_latent_dist_median":float(np.median(final_dist_list)) if final_dist_list else None,
        "final_latent_dist_std":   float(np.std(final_dist_list))    if final_dist_list else None,
        "path_len_mean":           float(np.mean(path_len_list))     if path_len_list   else None,
        "stop_reason_counts":      stop_counts,
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"NTField oracle results  ({args.ntfield_checkpoint})")
    print(f"Records evaluated  : {n_evaluated}  (errors: {n_errors})")
    print(
        f"Success (fm_xy<={args.finger_mid_xy_success_thresh_m}m AND "
        f"|fm_dz|<={args.finger_mid_z_success_thresh_m}m) : "
        f"{n_mid_success} / {n_evaluated}  ({100.0 * summary['success_rate']:.1f}%)"
    )
    print(
        f"EE success (<{args.ee_success_thresh}m) : {n_ee_success} / {n_evaluated}  "
        f"({100.0 * summary['ee_success_rate']:.1f}%)"
    )
    if ee_dist_list:
        print(f"EE dist (m)        : mean={summary['ee_dist_mean_m']:.4f}  "
              f"median={summary['ee_dist_median_m']:.4f}  "
              f"std={summary['ee_dist_std_m']:.4f}")
    if finger_mid_to_target_xy_list:
        print(
            "Finger mid → target XY (m): "
            f"mean={summary['finger_midpoint_to_target_xy_distance_mean_m']:.4f}  "
            f"median={summary['finger_midpoint_to_target_xy_distance_median_m']:.4f}  "
            f"std={summary['finger_midpoint_to_target_xy_distance_std_m']:.4f}"
        )
    if finger_mid_to_target_z_diff_list:
        print(
            "Finger mid Z − target_center Z (m): "
            f"mean={summary['finger_midpoint_to_target_z_diff_mean_m']:+.4f}  "
            f"median={summary['finger_midpoint_to_target_z_diff_median_m']:+.4f}  "
            f"std={summary['finger_midpoint_to_target_z_diff_std_m']:.4f}"
        )
    print(f"Between fingers     : {summary['n_between_fingers_success']} / {n_evaluated}  "
          f"({100.0 * summary['between_fingers_success_rate']:.1f}%)")
    if final_dist_list:
        print(f"Final latent dist  : mean={summary['final_latent_dist_mean']:.4f}  "
              f"median={summary['final_latent_dist_median']:.4f}")
    print(f"Stop reasons       : {stop_counts}")
    print(f"Results            : {results_path}")
    print(f"Summary            : {summary_path}")
    print("=" * 60)

    os.chdir(_cwd_prev)
    if not args.no_isaac_hard_exit:
        os._exit(0)


if __name__ == "__main__":
    main()