from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R
from isaacgym import gymapi
from isaacgym import gymutil
import fcl

_PI_VLA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HANWEN_GRASPING_ROOT = os.path.join(_PI_VLA_ROOT, "hanwen_grasping")
file_dir = os.path.join(HANWEN_GRASPING_ROOT, "collect_data")
util_dir = os.path.join(file_dir, "util")
grasp_util_dir = os.path.join(file_dir, "grasp_util")
sys.path.insert(0, HANWEN_GRASPING_ROOT)
sys.path.append(util_dir)
sys.path.append(grasp_util_dir)
sys.path.insert(0, _PI_VLA_ROOT)
sys.path.insert(0, os.path.join(_PI_VLA_ROOT, "ntrl-demo"))

import torch
import robot_arm_configuration as RC
from stl_reader import stl_reader
from obj_reader import obj_reader
from trajectory_evaluation.ntfield.eval_trajectory_ntfield import _ModelShim, load_network_and_function
from planning.gradient_planner_trajectory import SCALE as NTFIELD_SCALE
from planning.gradient_planner_trajectory import plan as ntfield_plan

TABLE_DIMS_X = 0.8
TABLE_DIMS_Y = 1.0
TABLE_DIMS_Z = 0.10
DRAWER_HEIGHT = 0.40
NUM_OF_OBJECTS = 3
# Three different default YCB objects (matches legacy new_setup pattern [1, 3, 5]).
# The first entry is the designated target object at (--object_x, --object_y, --object_z).
TARGET_OBJ_INDEX = [1, 3, 5]
ADD_COVER = False
MAX_RANDOM_PLACEMENT_ATTEMPTS = 500
XY_MIN_SEPARATION = 0.16

sim_dt = 1.0 / 60.0
SETTLE_STEPS = 15
FINAL_HOLD_STEPS = 80
RAD_PER_SIM_STEP_HEURISTIC = 0.018
EE_COLLISION_PROXY_RADIUS_M = 0.09
EE_PROXY_MAX_RADIUS_M_DEFAULT = 0.025


def _resolve_pi_vla_checkpoint(path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(_PI_VLA_ROOT, path))


def _path_as_6_list(path):
    return [np.asarray(p, dtype=np.float64).reshape(-1)[:6].tolist() for p in path]


def _resample_path_fixed_waypoints(path_6, target_count):
    if path_6 is None:
        return None
    if target_count is None or int(target_count) <= 0:
        return path_6
    target_count = int(target_count)
    if len(path_6) == 0:
        return path_6
    if len(path_6) == 1:
        return [path_6[0] for _ in range(target_count)]
    if target_count == 1:
        return [path_6[0]]
    arr = np.asarray(path_6, dtype=np.float64).reshape(-1, 6)
    seg = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg)))
    total = float(s[-1])
    if total <= 1e-12:
        return [arr[0].tolist() for _ in range(target_count)]
    s_new = np.linspace(0.0, total, target_count)
    out = np.empty((target_count, 6), dtype=np.float64)
    for j in range(6):
        out[:, j] = np.interp(s_new, s, arr[:, j])
    return out.tolist()


def get_swept_volume_size(main_swept):
    min_x, min_y, min_z = sys.maxsize, sys.maxsize, sys.maxsize
    max_x, max_y, max_z = -sys.maxsize, -sys.maxsize, -sys.maxsize
    for tx, ty, tz in main_swept:
        min_x = min(min_x, tx)
        min_y = min(min_y, ty)
        min_z = min(min_z, tz)
        max_x = max(max_x, tx)
        max_y = max(max_y, ty)
        max_z = max(max_z, tz)
    return max_y - min_y


def joint_metrics(path_6, q_start, q_goal):
    arr = np.array(path_6, dtype=np.float64).reshape(-1, 6)
    out = {}
    q0 = np.asarray(q_start, dtype=np.float64).reshape(6)
    qg = np.asarray(q_goal, dtype=np.float64).reshape(6)
    out["joint_net_abs_delta_rad"] = np.abs(qg - q0).tolist()
    out["joint_net_abs_delta_l1_rad"] = float(np.sum(np.abs(qg - q0)))
    out["joint_net_abs_delta_l2_rad"] = float(np.linalg.norm(qg - q0))
    if len(arr) < 2:
        out["joint_cumulative_abs_delta_per_joint_rad"] = [0.0] * 6
        out["path_segment_l1_sum_rad"] = 0.0
        out["path_segment_l2_sum_rad"] = 0.0
        return out
    d = np.abs(np.diff(arr, axis=0))
    out["joint_cumulative_abs_delta_per_joint_rad"] = np.sum(d, axis=0).tolist()
    out["path_segment_l1_sum_rad"] = float(np.sum(d))
    segl2 = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    out["path_segment_l2_sum_rad"] = float(np.sum(segl2))
    return out


def _settle_steps_at_waypoint(path_local, waypoint_idx):
    if not path_local or waypoint_idx <= 0:
        return SETTLE_STEPS
    q0 = np.asarray(path_local[waypoint_idx - 1], dtype=np.float64)
    q1 = np.asarray(path_local[waypoint_idx], dtype=np.float64)
    dq = float(np.max(np.abs(q1 - q0)))
    return max(SETTLE_STEPS, min(600, int(math.ceil(dq / RAD_PER_SIM_STEP_HEURISTIC))))


def _append_cam_rgb(gym, sim, env, main_cam_handle, camera_props, record_rgb):
    if record_rgb is None or main_cam_handle is None:
        return
    gym.render_all_camera_sensors(sim)
    raw_main = gym.get_camera_image(sim, env, main_cam_handle, gymapi.IMAGE_COLOR)
    rgba_main = raw_main.reshape(camera_props.height, camera_props.width, 4)
    record_rgb.append(rgba_main[..., :3].copy())


def _save_mp4_rgb(frames, out_mp4, fps=60.0):
    if not frames:
        print(f"No frames; skip {out_mp4}")
        return
    os.makedirs(os.path.dirname(os.path.abspath(out_mp4)), exist_ok=True)
    try:
        import imageio

        imageio.mimsave(out_mp4, frames, fps=fps)
        print(f"Saved {len(frames)} frames -> {out_mp4}")
    except Exception as e1:
        try:
            import cv2

            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            wri = cv2.VideoWriter(out_mp4, fourcc, fps, (w, h))
            for fr in frames:
                wri.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
            wri.release()
            print(f"Saved {len(frames)} frames (OpenCV) -> {out_mp4}")
        except Exception as e2:
            print(f"Could not save {out_mp4}: {e1}; {e2}")


def _compute_ee_proxy_from_gripper(
    rac,
    dof_result,
    center_mode="between_fingertips",
    max_radius_m=EE_PROXY_MAX_RADIUS_M_DEFAULT,
):
    """Build EE sphere proxy from gripper mesh (matches comparison/run_rrt_ntfield_benchmark.py)."""
    pose_array = rac.calculate_transform_from_angles(dof_result)
    ee_center = np.asarray(pose_array[8][0], dtype=np.float64).reshape(3)
    grip_verts_local = np.asarray(rac.collision_models_["gripper"][0], dtype=np.float64).reshape(-1, 3)
    grip_rot = R.from_quat(np.asarray(pose_array[8][1], dtype=np.float64)).as_matrix()
    grip_verts_world = (grip_verts_local @ grip_rot.T) + ee_center.reshape(1, 3)

    center_world = ee_center
    center_local = np.zeros(3, dtype=np.float64)
    if center_mode == "between_fingertips":
        left_mask = grip_verts_local[:, 0] >= 0.0
        right_mask = grip_verts_local[:, 0] < 0.0
        if np.any(left_mask) and np.any(right_mask):
            left_local = grip_verts_local[left_mask]
            right_local = grip_verts_local[right_mask]
            left_world = grip_verts_world[left_mask]
            right_world = grip_verts_world[right_mask]
            left_tip_idx = int(np.argmax(left_local[:, 2]))
            right_tip_idx = int(np.argmax(right_local[:, 2]))
            center_world = 0.5 * (left_world[left_tip_idx] + right_world[right_tip_idx])
            center_local = 0.5 * (left_local[left_tip_idx] + right_local[right_tip_idx])

    derived_radius = float(np.max(np.linalg.norm(grip_verts_local - center_local.reshape(1, 3), axis=1)))
    if max_radius_m is not None and float(max_radius_m) > 0.0:
        derived_radius = float(min(derived_radius, float(max_radius_m)))
    return center_world, derived_radius


def _evaluate_ee_proxy_against_scene(
    rac,
    dof_result,
    flex_collision_models,
    target_idx,
    spawned_object_asset_indices,
    center_mode="between_fingertips",
    max_radius_m=EE_PROXY_MAX_RADIUS_M_DEFAULT,
):
    """
    Single FCL pair query of EE sphere proxy against:
    - the designated target object's settled BVH mesh (must collide for grasp to count),
    - every other (obstacle) object's settled BVH mesh (must NOT collide).

    Returns a dict with: target/obstacles/no_obstacle_collision/success/ee_center_world_m/radius_m.
    """
    if not flex_collision_models or target_idx < 0 or target_idx >= len(flex_collision_models):
        return {
            "evaluated": False,
            "target": None,
            "obstacles": [],
            "no_obstacle_collision": False,
            "success": False,
            "ee_center_world_m": None,
            "radius_m": None,
        }
    ee_center, derived_radius = _compute_ee_proxy_from_gripper(
        rac, dof_result, center_mode=center_mode, max_radius_m=max_radius_m
    )
    ee_proxy_obj = fcl.CollisionObject(fcl.Sphere(derived_radius), fcl.Transform(ee_center))

    def _pair(co):
        req = fcl.CollisionRequest(num_max_contacts=20, enable_contact=True)
        res = fcl.CollisionResult()
        n = int(fcl.collide(ee_proxy_obj, co, req, res))
        return bool(n > 0), int(n)

    target_co = flex_collision_models[target_idx][0]
    target_hit, target_n = _pair(target_co)
    target_block = {
        "object_index": int(target_idx),
        "asset_index": int(spawned_object_asset_indices[target_idx]) if spawned_object_asset_indices is not None else None,
        "collision": bool(target_hit),
        "num_contacts": int(target_n),
    }

    obstacles = []
    no_obstacle_collision = True
    for k, item in enumerate(flex_collision_models):
        if k == target_idx:
            continue
        hit_k, n_k = _pair(item[0])
        if hit_k:
            no_obstacle_collision = False
        obstacles.append({
            "object_index": int(k),
            "asset_index": int(spawned_object_asset_indices[k]) if spawned_object_asset_indices is not None else None,
            "collision": bool(hit_k),
            "num_contacts": int(n_k),
        })

    success = bool(target_hit and no_obstacle_collision)
    return {
        "evaluated": True,
        "target": target_block,
        "obstacles": obstacles,
        "no_obstacle_collision": bool(no_obstacle_collision),
        "success": success,
        "ee_center_world_m": ee_center.tolist(),
        "radius_m": float(derived_radius),
    }


def planner_terminal_ee_check(
    rac,
    planner_path_6,
    flex_collision_models,
    target_idx,
    spawned_object_asset_indices,
    center_mode,
    max_radius_m,
):
    """Evaluate the EE proxy / scene pair check at the planner's terminal waypoint."""
    if planner_path_6 is None or len(planner_path_6) == 0:
        return {
            "evaluated": False,
            "target": None,
            "obstacles": [],
            "no_obstacle_collision": False,
            "success": False,
            "ee_center_world_m": None,
            "radius_m": None,
        }
    q_last = np.asarray(planner_path_6[-1], dtype=np.float64).reshape(6)
    return _evaluate_ee_proxy_against_scene(
        rac,
        q_last,
        flex_collision_models,
        target_idx,
        spawned_object_asset_indices,
        center_mode=center_mode,
        max_radius_m=max_radius_m,
    )


def save_final_geometric_debug_image_multi(
    out_path,
    rac,
    dof_result,
    object_meshes,
    target_idx,
    ee_center_world,
    ee_radius_m=None,
):
    """3D debug PNG: link chain + gripper mesh + EE sphere + target (orange) and obstacle (gray) meshes."""
    pose_array = rac.calculate_transform_from_angles(dof_result)
    link_pts = np.asarray([p[0] for p in pose_array[:9]], dtype=np.float64).reshape(-1, 3)
    ee_center = np.asarray(ee_center_world, dtype=np.float64).reshape(3)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    obj_pts_for_bounds = []
    for k, mesh in enumerate(object_meshes):
        if mesh is None:
            continue
        verts = np.asarray(mesh[0], dtype=np.float64)
        faces = np.asarray(mesh[1], dtype=np.int32)
        if verts.size == 0 or faces.size == 0:
            continue
        is_target = (k == target_idx)
        face_color = "tab:orange" if is_target else "lightgray"
        edge_color = "k" if is_target else "dimgray"
        alpha = 0.32 if is_target else 0.18
        tri_polys = [verts[f] for f in faces]
        coll = Poly3DCollection(tri_polys, alpha=alpha, facecolor=face_color, edgecolor=edge_color, linewidths=0.2)
        ax.add_collection3d(coll)
        obj_pts_for_bounds.append(verts)

    ax.plot(link_pts[:, 0], link_pts[:, 1], link_pts[:, 2], color="tab:blue", linewidth=2.0, marker="o", markersize=3)

    grip_verts_local, grip_faces = rac.collision_models_["gripper"]
    grip_verts_local = np.asarray(grip_verts_local, dtype=np.float64)
    grip_faces = np.asarray(grip_faces, dtype=np.int32)
    grip_rot = R.from_quat(np.asarray(pose_array[8][1], dtype=np.float64)).as_matrix()
    grip_tran = np.asarray(pose_array[8][0], dtype=np.float64).reshape(3)
    grip_verts_world = (grip_verts_local @ grip_rot.T) + grip_tran.reshape(1, 3)
    grip_polys = [grip_verts_world[f] for f in grip_faces]
    grip_coll = Poly3DCollection(grip_polys, alpha=0.35, facecolor="tab:red", edgecolor="k", linewidths=0.15)
    ax.add_collection3d(grip_coll)
    ax.scatter([ee_center[0]], [ee_center[1]], [ee_center[2]], color="red", s=24, label="EE proxy center")
    if ee_radius_m is not None:
        u = np.linspace(0.0, 2.0 * np.pi, 40)
        v = np.linspace(0.0, np.pi, 24)
        xs = ee_center[0] + ee_radius_m * np.outer(np.cos(u), np.sin(v))
        ys = ee_center[1] + ee_radius_m * np.outer(np.sin(u), np.sin(v))
        zs = ee_center[2] + ee_radius_m * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(xs, ys, zs, color="magenta", alpha=0.16, linewidth=0, antialiased=True)

    pts_for_bounds = [link_pts, grip_verts_world, ee_center.reshape(1, 3)] + obj_pts_for_bounds
    all_pts = np.vstack(pts_for_bounds)
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    center = 0.5 * (mins + maxs)
    half = 0.5 * np.max(maxs - mins) + 1e-6
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Final geometric debug (multi): target=orange, obstacles=gray, EE sphere=magenta")
    ax.legend(loc="upper right")
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def execute_path_and_time(
    gym,
    sim,
    env,
    ur_handle,
    spj,
    slj,
    ej,
    wj1,
    wj2,
    wj3,
    viewer,
    path_local,
    label,
    main_cam_handle=None,
    camera_props=None,
    record_rgb=None,
    planner_playback="direct",
):
    if not path_local:
        return {"label": label, "success": False, "execution_wall_s": None, "execution_sim_s": None, "physics_steps": 0}
    t0 = time.perf_counter()
    n_sub = 0
    for path_id in range(len(path_local)):
        dof_result = path_local[path_id]
        n_hold = _settle_steps_at_waypoint(path_local, path_id) if planner_playback == "settle" else 1
        for _ in range(n_hold):
            gym.set_dof_target_position(env, spj, float(dof_result[0]))
            gym.set_dof_target_position(env, slj, float(dof_result[1]))
            gym.set_dof_target_position(env, ej, float(dof_result[2]))
            gym.set_dof_target_position(env, wj1, float(dof_result[3]))
            gym.set_dof_target_position(env, wj2, float(dof_result[4]))
            gym.set_dof_target_position(env, wj3, float(dof_result[5]))
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.step_graphics(sim)
            if viewer is not None:
                gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)
            n_sub += 1
            _append_cam_rgb(gym, sim, env, main_cam_handle, camera_props, record_rgb)
    dof_last = path_local[-1]
    for _ in range(FINAL_HOLD_STEPS):
        gym.set_dof_target_position(env, spj, float(dof_last[0]))
        gym.set_dof_target_position(env, slj, float(dof_last[1]))
        gym.set_dof_target_position(env, ej, float(dof_last[2]))
        gym.set_dof_target_position(env, wj1, float(dof_last[3]))
        gym.set_dof_target_position(env, wj2, float(dof_last[4]))
        gym.set_dof_target_position(env, wj3, float(dof_last[5]))
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        if viewer is not None:
            gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)
        n_sub += 1
        _append_cam_rgb(gym, sim, env, main_cam_handle, camera_props, record_rgb)
    t1 = time.perf_counter()
    return {
        "label": label,
        "success": True,
        "execution_wall_s": float(t1 - t0),
        "execution_sim_s": float(n_sub * sim_dt),
        "physics_steps": int(n_sub),
    }


def reset_arm_to_q(gym, sim, env, ur_handle, spj, slj, ej, wj1, wj2, wj3, viewer, q, n_steps=200):
    for _ in range(n_steps):
        gym.set_dof_target_position(env, spj, float(q[0]))
        gym.set_dof_target_position(env, slj, float(q[1]))
        gym.set_dof_target_position(env, ej, float(q[2]))
        gym.set_dof_target_position(env, wj1, float(q[3]))
        gym.set_dof_target_position(env, wj2, float(q[4]))
        gym.set_dof_target_position(env, wj3, float(q[5]))
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        if viewer is not None:
            gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)


def _make_object_collision_model(asset_root, object_collision_files, object_offset, file_idx, scaling):
    file_path = object_collision_files[file_idx]
    collision_mesh = obj_reader(asset_root + file_path)
    collision_mesh.set_scale(scaling)
    collision_mesh.add_offset(object_offset[file_idx])
    verts, tris = collision_mesh.get_bounding_box_mesh()
    temp_center = collision_mesh.get_center()
    temp_bounding_box = collision_mesh.get_bounding_box()
    m = fcl.BVHModel()
    m.beginModel(len(verts), len(tris))
    m.addSubModel(verts, tris)
    m.endModel()
    return collision_mesh, temp_center, temp_bounding_box, m


def _sample_random_xy(table_dims):
    tx = np.random.uniform(0.35, table_dims.x + 0.2)
    ty = np.random.uniform(-table_dims.y / 2 + 0.1, table_dims.y / 2 - 0.2)
    return float(tx), float(ty)


def main():
    parser = argparse.ArgumentParser(description="RRTConnect vs NTField benchmark with 3 objects (1 fixed + 2 random).")
    parser.add_argument("--object_x", type=float, required=True, help="Designated target object x (world m)")
    parser.add_argument("--object_y", type=float, required=True, help="Designated target object y (world m)")
    parser.add_argument("--object_z", type=float, required=True, help="Designated target object z (world m)")
    parser.add_argument("--ntfield_checkpoint", type=str, required=True, help="Trajectory NTField Model_Epoch_*.pt")
    parser.add_argument("--ntfield_experiment_dir", type=str, default=None)
    parser.add_argument("--use_viewer", action="store_true")
    parser.add_argument("--ntfield_device", type=str, default="cuda:0")
    parser.add_argument("--ntfield_step_size", type=float, default=0.02)
    parser.add_argument("--ntfield_max_steps", type=int, default=200)
    parser.add_argument("--ntfield_tol", type=float, default=0.01)
    parser.add_argument("--ntfield_goal_eps_rad", type=float, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--record_dir", type=str, default=None)
    parser.add_argument("--video_fps", type=float, default=60.0)
    parser.add_argument("--no_video", action="store_true")
    parser.add_argument("--planner_playback", type=str, choices=("direct", "settle"), default="direct")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--ntfield_waypoint_mode", type=str, choices=("full", "two_point"), default="full")
    parser.add_argument("--ntfield_fixed_waypoints", type=int, default=0)
    parser.add_argument(
        "--save_final_geometric_debug",
        action="store_true",
        help="Save a PNG showing final robot configuration, EE collision proxy, and all object meshes.",
    )
    parser.add_argument(
        "--final_geometric_debug_path",
        type=str,
        default=None,
        help="Optional output path for final geometric debug PNG (defaults to <record_dir>/final_geometric_debug.png).",
    )
    parser.add_argument(
        "--require_ee_object_collision",
        action="store_true",
        help="If set, mark a planner's success=False unless EE proxy collides with target AND not with obstacles.",
    )
    parser.add_argument(
        "--ee_proxy_radius_m",
        type=float,
        default=EE_COLLISION_PROXY_RADIUS_M,
        help="Legacy arg (ignored): use --ee_proxy_max_radius_m instead.",
    )
    parser.add_argument(
        "--ee_proxy_center_mode",
        type=str,
        choices=("ee_origin", "between_fingertips"),
        default="between_fingertips",
        help="Center EE proxy at EE origin or midpoint between gripper fingertip proxies.",
    )
    parser.add_argument(
        "--ee_proxy_max_radius_m",
        type=float,
        default=EE_PROXY_MAX_RADIUS_M_DEFAULT,
        help="Cap EE proxy sphere radius in meters after auto-derivation.",
    )
    args, argv_remainder = parser.parse_known_args()
    argv_gym = list(argv_remainder)
    if not args.use_viewer and "--headless" not in argv_gym:
        argv_gym.append("--headless")
    sys.argv = [sys.argv[0]] + argv_gym
    if args.seed is not None:
        np.random.seed(args.seed)

    ckpt_abs = _resolve_pi_vla_checkpoint(args.ntfield_checkpoint)
    if not os.path.isfile(ckpt_abs):
        print(f"Checkpoint not found: {ckpt_abs}")
        sys.exit(1)
    if args.ntfield_device != "cpu" and not torch.cuda.is_available():
        print("Warning: CUDA unavailable; NTField using cpu")
    dev = torch.device("cpu" if args.ntfield_device == "cpu" or not torch.cuda.is_available() else args.ntfield_device)
    _, ntfield_fn = load_network_and_function(ckpt_abs, args.ntfield_experiment_dir, dev, dim=6)
    nt_model = _ModelShim(ntfield_fn)
    ntfield_device_str = str(dev) if dev.type == "cuda" else "cpu"
    goal_eps = float(args.ntfield_goal_eps_rad) if args.ntfield_goal_eps_rad is not None else float(args.ntfield_tol * NTFIELD_SCALE)

    _invoke_cwd = os.getcwd()
    os.chdir(HANWEN_GRASPING_ROOT)

    gym = gymapi.acquire_gym()
    gym_args = gymutil.parse_arguments(description="benchmark", headless=True, custom_parameters=[])
    gym_args.headless = not args.use_viewer

    table_dims = gymapi.Vec3(TABLE_DIMS_X, TABLE_DIMS_Y, TABLE_DIMS_Z)
    drawer_height = DRAWER_HEIGHT

    sim_params = gymapi.SimParams()
    sim_params.substeps = 2
    sim_params.dt = sim_dt
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = gym_args.num_threads
    sim_params.physx.use_gpu = gym_args.use_gpu
    sim_params.use_gpu_pipeline = False

    sim = gym.create_sim(gym_args.compute_device_id, gym_args.graphics_device_id, gym_args.physics_engine, sim_params)
    if sim is None:
        print("Failed to create sim")
        sys.exit(1)

    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)

    asset_root = "./assets/"
    ur5e_asset_file = "urdf/ur5e/ur5e_mimic_real_gripper_test.urdf"
    ur5e_collision_parts = [
        "urdf/ur5e/meshes/collision/base.stl",
        "urdf/ur5e/meshes/collision/shoulder.stl",
        "urdf/ur5e/meshes/collision/upperarm.stl",
        "urdf/ur5e/meshes/collision/forearm.stl",
        "urdf/ur5e/meshes/collision/wrist1.stl",
        "urdf/ur5e/meshes/collision/wrist2.stl",
        "urdf/ur5e/meshes/collision/wrist3.stl",
    ]
    object_asset_files = []
    object_collision_files = []
    object_offset = []
    object_common_prefix = "urdf/ycb/"
    with open(asset_root + "urdf/ycb/object_urdf_grasp.txt") as f:
        for line in f:
            object_asset_files.append(object_common_prefix + line[:-1])
    with open(asset_root + "urdf/ycb/object_collision_grasp.txt") as f:
        for line in f:
            object_collision_files.append(object_common_prefix + line[:-1])
    with open(asset_root + "urdf/ycb/object_offset_grasp.txt") as f:
        for line in f:
            object_offset.append([float(x) for x in line[:-1].split(" ")])

    ur5e_collision_models = []
    ur5e_rotations = [
        R.from_euler("x", [90], degrees=True),
        R.from_euler("xy", [90, 180], degrees=True),
        R.from_euler("xy", [180, 180], degrees=True),
        R.from_euler("z", [-180], degrees=True),
        R.from_euler("x", [-180], degrees=True),
        R.from_euler("x", [90], degrees=True),
        R.from_euler("z", [-90], degrees=True),
    ]
    ur5e_translations = [[0, 0, 0], [0, 0, 0], [0, -0.138, 0], [0, -0.007, 0], [0, 0.127, 0], [0, 0, 0], [0, 0, 0]]
    for idx, parts_path in enumerate(ur5e_collision_parts):
        collision_mesh = stl_reader(asset_root + parts_path)
        m = fcl.BVHModel()
        collision_mesh.transform(ur5e_rotations[idx], ur5e_translations[idx])
        verts, tris = collision_mesh.get_vertices(), collision_mesh.get_faces()
        m.beginModel(len(verts), len(tris))
        m.addSubModel(verts, tris)
        m.endModel()
        ur5e_collision_models.append(m)

    object_collision_lib = []
    viewer = None
    if not gym_args.headless:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is None:
            gym_args.headless = True

    spacing = 2
    env_lower = gymapi.Vec3(-spacing, -spacing, 0)
    env_upper = gymapi.Vec3(spacing, spacing, 0)

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)
    asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
    asset_options.use_mesh_materials = True
    ur5e_asset = gym.load_asset(sim, asset_root, ur5e_asset_file, asset_options)
    table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)
    upper_cover_dims = gymapi.Vec3(table_dims.x, table_dims.y, 0.03)
    upper_cover_asset = gym.create_box(sim, upper_cover_dims.x, upper_cover_dims.y, upper_cover_dims.z, asset_options)
    asset_options.fix_base_link = False
    object_assets = [gym.load_asset(sim, asset_root, ob, asset_options) for ob in object_asset_files]

    ur5e_pose = gymapi.Transform()
    ur5e_pose.p = gymapi.Vec3(0, 0, 0)
    ur5e_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0.5 * math.pi)
    table_pose = gymapi.Transform()
    table_pose.p = gymapi.Vec3(table_dims.x * 0.5 + 0.3, 0.0, table_dims.z * 0.5)
    upper_cover_pose = gymapi.Transform()
    upper_cover_pose.p = gymapi.Vec3(table_pose.p.x, 0.0, table_dims.z + drawer_height + 0.015)
    camera_focus = gymapi.Vec3(0, 0, 0)
    camera_props = gymapi.CameraProperties()
    camera_props.horizontal_fov = 70.25
    camera_props.width = 1280
    camera_props.height = 720

    plane_normal = np.array([0.0, 0.0, 1.0])
    col_plane = fcl.Plane(plane_normal, 0)
    plane_obj = fcl.CollisionObject(col_plane, fcl.Transform())
    col_table = fcl.Box(table_dims.x, table_dims.y, table_dims.z)
    trans_table = fcl.Transform(np.array([table_dims.x * 0.5 + 0.3, 0.0, table_dims.z * 0.5]))
    table_obj = fcl.CollisionObject(col_table, trans_table)
    object_collision_models = [table_obj]

    num_of_envs = 1
    row_num_of_envs = 1
    envs = []
    ur5e_handles = []
    object_handles = []
    object_status_list = []
    object_reader_tracker = []
    object_mesh = []
    flex_collision_models = []
    spj = slj = ej = wj1 = wj2 = wj3 = None
    target_file_idx = np.array(TARGET_OBJ_INDEX, dtype=np.int64)
    if target_file_idx.size != NUM_OF_OBJECTS or len(set(target_file_idx.tolist())) != NUM_OF_OBJECTS:
        raise ValueError(f"TARGET_OBJ_INDEX must contain {NUM_OF_OBJECTS} unique object indices.")
    main_cam_handle = None
    spawned_object_world_xyz = []

    for i in range(num_of_envs):
        envs.append(gym.create_env(sim, env_lower, env_upper, row_num_of_envs))
        ur5e_handles.append(gym.create_actor(envs[-1], ur5e_asset, ur5e_pose, "ur5e" + str(i), 0, 32767))
        spj = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "shoulder_pan_joint")
        slj = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "shoulder_lift_joint")
        ej = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "elbow_joint")
        wj1 = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "wrist_1_joint")
        wj2 = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "wrist_2_joint")
        wj3 = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "wrist_3_joint")
        gym.create_actor(envs[-1], table_asset, table_pose, "table" + str(i), 0, 1)
        if ADD_COVER:
            gym.create_actor(envs[-1], upper_cover_asset, upper_cover_pose, "upper_cover" + str(i), 0, 1)

        objs_manager = fcl.DynamicAABBTreeCollisionManager()
        objs_manager.setup()
        obstacle_objs = []
        GT_OBJ_POS_LIST = []
        object_scaling_factor = np.ones(NUM_OF_OBJECTS, dtype=np.float64)

        for k in range(NUM_OF_OBJECTS):
            object_pose = gymapi.Transform()
            fk = int(target_file_idx[k])
            collision_mesh, temp_center, temp_bounding_box, m = _make_object_collision_model(
                asset_root, object_collision_files, object_offset, fk, object_scaling_factor[k]
            )

            if k == 0:
                tx, ty, tz = float(args.object_x), float(args.object_y), float(args.object_z)
            else:
                placed = False
                tx = ty = tz = 0.0
                for _ in range(MAX_RANDOM_PLACEMENT_ATTEMPTS):
                    tx_cand, ty_cand = _sample_random_xy(table_dims)
                    tz_cand = float(table_dims.z + 0.08)
                    t_cand = fcl.Transform(np.array([tx_cand, ty_cand, tz_cand]))

                    req = fcl.CollisionRequest()
                    rdata = fcl.CollisionData(request=req)
                    objs_manager.collide(fcl.CollisionObject(m, t_cand), rdata, fcl.defaultCollisionCallback)
                    is_collision = bool(rdata.result.is_collision)

                    if not is_collision:
                        too_close = False
                        for px, py in GT_OBJ_POS_LIST:
                            if float(np.hypot(tx_cand - px, ty_cand - py)) <= XY_MIN_SEPARATION:
                                too_close = True
                                break
                        if not too_close:
                            tx, ty, tz = tx_cand, ty_cand, tz_cand
                            placed = True
                            break
                if not placed:
                    raise RuntimeError("Failed to place random obstacle objects without overlap.")

            object_pose.p = gymapi.Vec3(tx, ty, tz)
            t = fcl.Transform(np.array([tx, ty, tz]))
            GT_OBJ_POS_LIST.append([tx, ty])
            spawned_object_world_xyz.append([tx, ty, tz])

            object_handles.append(
                gym.create_actor(
                    envs[-1], object_assets[fk], object_pose, "object" + str(k) + str(i), 0, 2 ** (k + 1), k + 1
                )
            )
            gym.set_actor_scale(envs[-1], object_handles[-1], object_scaling_factor[k])
            object_reader_tracker.append(collision_mesh)
            object_status_list.append([temp_center, temp_bounding_box])
            object_collision_lib.append(m)
            obstacle_obj = fcl.CollisionObject(m, t)
            obstacle_objs.append(obstacle_obj)
            objs_manager.registerObjects([obstacle_obj])
            objs_manager.setup()

        main_cam_handle = gym.create_camera_sensor(envs[-1], camera_props)
        main_cam_pos = gymapi.Vec3(3, 0, 0.3)
        gym.set_camera_location(main_cam_handle, envs[-1], main_cam_pos, camera_focus)

    if viewer is not None:
        cam_pos = gymapi.Vec3(2.2, 0, 0.5)
        cam_target = gymapi.Vec3(0, 0, 0.5)
        gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    real_position = False
    env = envs[-1]
    ur = ur5e_handles[-1]
    for t in range(2000):
        if not real_position:
            gym.set_dof_target_position(env, spj, 0)
            gym.set_dof_target_position(env, slj, -math.pi / 2)
            gym.set_dof_target_position(env, ej, 0)
            gym.set_dof_target_position(env, wj1, -math.pi / 2)
            gym.set_dof_target_position(env, wj2, 0)
            gym.set_dof_target_position(env, wj3, 0)
            real_position = True
        if t == 999:
            for ii, element in enumerate(object_handles):
                states = gym.get_actor_rigid_body_states(env, element, 1)
                rotation = np.array(states[0][0][1])
                translation = np.array(states[0][0][0])
                rotation = np.array(rotation.item())
                translation = np.array(translation.item())
                object_status_list[ii][0] += translation
                r1 = R.from_quat(rotation)
                tf = fcl.Transform(r1.as_matrix(), translation)
                flex_collision_models.append([fcl.CollisionObject(object_collision_lib[ii], tf), 0])
                temp_obj = object_reader_tracker[ii]
                temp_obj.set_offset(translation)
                vertices, faces = temp_obj.get_bounding_box_mesh()
                object_mesh.append([vertices, faces])
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        if viewer is not None:
            gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    scene_info = [table_dims.x, table_dims.y, table_dims.z, drawer_height]
    file_path_rac = "./assets/urdf/ur5e/meshes/collision/"
    rac = RC.robot_arm_configuration(file_path_rac, np.array([ur5e_pose.p.x, ur5e_pose.p.y, ur5e_pose.p.z]), scene_info)

    target_idx = 0
    grasp_file = "./assets/" + "/".join(object_asset_files[target_file_idx[target_idx]].split("/")[:-1]) + "/grasp_dict.npy"
    grasp_data = np.load(grasp_file, allow_pickle=True)

    num_grasp = 0
    swept_size = sys.maxsize
    grasp_list = np.arange(len(grasp_data))
    np.random.shuffle(grasp_list)
    init2grasp_path = None
    grasp_target_q = None
    _rrt_plan_s = 0.0
    consecutive_path_failures = 0
    MAX_FAIL = 15

    for grasp_idx in grasp_list:
        if consecutive_path_failures >= MAX_FAIL:
            break
        target_grasp_pos = grasp_data[grasp_idx]["target_pos"].copy()
        target_grasp_quat = grasp_data[grasp_idx]["target_quat"]
        target_grasp_pos[:2] = target_grasp_pos[:2] + GT_OBJ_POS_LIST[target_idx][:2]
        init2grasp_angels_temp = rac.grasp_verify(target_grasp_pos, target_grasp_quat)
        grasp2init_angels_temp = rac.grasp_verify(target_grasp_pos + [0, 0, 0.01], target_grasp_quat)
        if init2grasp_angels_temp is None or grasp2init_angels_temp is None:
            continue
        init2grasp_collision = rac.arm_collision_free(init2grasp_angels_temp, plane_obj, object_collision_models, [])
        grasp2init_collision = rac.arm_collision_free(grasp2init_angels_temp, plane_obj, object_collision_models, [])
        if not init2grasp_collision or not grasp2init_collision:
            continue

        t_rrt0 = time.perf_counter()
        init2grasp_path_temp = RC.get_path2grasp(
            rac,
            init2grasp_angels_temp,
            scene_info,
            target_mesh=object_mesh[target_idx],
            time_limit=30,
            given_static_model=object_collision_models,
        )
        rrt_planning_wall_s = float(time.perf_counter() - t_rrt0)
        if init2grasp_path_temp is None:
            consecutive_path_failures += 1
            continue

        temp_mod_bbox = rac.modify_grasp_bbox(init2grasp_angels_temp, target_mesh=object_mesh[target_idx], visualize=False)
        grasp2init_path_temp = RC.get_path2start(
            rac,
            grasp2init_angels_temp,
            temp_mod_bbox,
            scene_info,
            time_limit=30,
            given_static_model=object_collision_models,
        )
        if grasp2init_path_temp is None:
            consecutive_path_failures += 1
            continue
        consecutive_path_failures = 0

        swept_volume1_temp, swept_verts1_temp = rac.get_swept_volume(
            init2grasp_path_temp, frame_rate=60, scene_info=scene_info, animation=False, static_vi=False
        )
        swept_volume2_temp, swept_verts2_temp = rac.get_swept_volume(
            grasp2init_path_temp, w_target=temp_mod_bbox, frame_rate=60, scene_info=scene_info, animation=False, static_vi=False
        )
        num_grasp += 1
        swept_center_temp, swept_verts_temp = rac.get_swept_center(swept_verts1_temp + swept_verts2_temp, scene_info, 0.6)
        temp_swept_size = get_swept_volume_size(swept_verts_temp)
        if temp_swept_size < swept_size:
            swept_size = temp_swept_size
            init2grasp_path = init2grasp_path_temp
            grasp_target_q = np.asarray(init2grasp_angels_temp, dtype=np.float64).reshape(6)
            _rrt_plan_s = rrt_planning_wall_s
        if num_grasp == 1:
            break

    dof_snapshot = gym.get_actor_dof_states(env, ur, gymapi.STATE_POS)
    q_start_live = np.array(dof_snapshot["pos"][:6], dtype=np.float64)

    result = {
        "timestamp": datetime.now().isoformat(),
        "planner_playback": args.planner_playback,
        "ntfield_waypoint_mode": args.ntfield_waypoint_mode,
        "ntfield_fixed_waypoints": int(args.ntfield_fixed_waypoints),
        "table_dims_m": [TABLE_DIMS_X, TABLE_DIMS_Y, TABLE_DIMS_Z],
        "designated_object_pose_world_m": [args.object_x, args.object_y, args.object_z],
        "all_object_poses_world_m": spawned_object_world_xyz,
        "num_objects": NUM_OF_OBJECTS,
        "target_object_asset_index": int(target_file_idx[0]),
        "spawned_object_asset_indices": target_file_idx.tolist(),
        "spawned_object_urdf_files": [object_asset_files[int(i)] for i in target_file_idx.tolist()],
        "q_start_live": q_start_live.tolist(),
        "goal_configuration_grasp_verify": grasp_target_q.tolist() if grasp_target_q is not None else None,
        "ee_object_pair_collision_required": bool(args.require_ee_object_collision),
        "ee_object_pair_collision_proxy": {
            "shape": "sphere_from_gripper_mesh",
            "center_mode": args.ee_proxy_center_mode,
            "max_radius_m": float(args.ee_proxy_max_radius_m),
            "radius_m": None,
        },
        "ee_object_pair_collision_selected_grasp": {
            "evaluated": False,
            "success": False,
            "target": None,
            "obstacles": [],
            "no_obstacle_collision": False,
            "ee_center_world_m": None,
        },
        "rrtconnect": {},
        "ntfield": {},
        "ntfield_checkpoint": ckpt_abs,
    }

    if init2grasp_path is None or grasp_target_q is None:
        result["error"] = "No valid grasp + RRT path found"
        print(json.dumps(result, indent=2))
        if args.output_json:
            with open(args.output_json, "w") as jf:
                json.dump(result, jf, indent=2)
        gym.destroy_sim(sim)
        if viewer is not None:
            gym.destroy_viewer(viewer)
        os.chdir(_invoke_cwd)
        sys.exit(1)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.abspath(args.record_dir) if args.record_dir else os.path.join(_PI_VLA_ROOT, "output", "trajectory_evaluation", f"multi_benchmark_{stamp}")
    os.makedirs(session_dir, exist_ok=True)
    mp4_rrt = os.path.join(session_dir, "rrt.mp4")
    mp4_nt = os.path.join(session_dir, "ntfield.mp4")
    result["video_session_dir"] = session_dir
    want_video = not args.no_video

    path_rrt = _path_as_6_list(init2grasp_path)
    if grasp_target_q is None:
        grasp_target_q = np.asarray(path_rrt[-1], dtype=np.float64).reshape(6)

    selected_grasp_check = _evaluate_ee_proxy_against_scene(
        rac,
        grasp_target_q,
        flex_collision_models,
        target_idx=target_idx,
        spawned_object_asset_indices=target_file_idx.tolist(),
        center_mode=args.ee_proxy_center_mode,
        max_radius_m=args.ee_proxy_max_radius_m,
    )
    result["ee_object_pair_collision_selected_grasp"] = {
        "evaluated": bool(selected_grasp_check["evaluated"]),
        "success": bool(selected_grasp_check["success"]),
        "target": selected_grasp_check["target"],
        "obstacles": selected_grasp_check["obstacles"],
        "no_obstacle_collision": bool(selected_grasp_check["no_obstacle_collision"]),
        "ee_center_world_m": selected_grasp_check["ee_center_world_m"],
    }
    result["ee_object_pair_collision_proxy"]["radius_m"] = selected_grasp_check["radius_m"]

    result["rrtconnect"]["planning_wall_s_for_get_path2grasp_only"] = _rrt_plan_s
    result["rrtconnect"]["success"] = True
    result["rrtconnect"]["num_waypoints"] = len(path_rrt)
    result["rrtconnect"]["trajectory_waypoints_rad"] = path_rrt
    result["rrtconnect"]["motion"] = joint_metrics(path_rrt, q_start_live, grasp_target_q)
    rrt_terminal_check = planner_terminal_ee_check(
        rac,
        path_rrt,
        flex_collision_models,
        target_idx=target_idx,
        spawned_object_asset_indices=target_file_idx.tolist(),
        center_mode=args.ee_proxy_center_mode,
        max_radius_m=args.ee_proxy_max_radius_m,
    )
    result["rrtconnect"]["ee_object_pair_collision_terminal_waypoint"] = rrt_terminal_check
    if args.require_ee_object_collision and not rrt_terminal_check["success"]:
        result["rrtconnect"]["success"] = False
        result["rrtconnect"]["ee_object_postcheck_failed"] = True

    frames_rrt = [] if want_video else None
    exec_rrt = execute_path_and_time(
        gym, sim, env, ur, spj, slj, ej, wj1, wj2, wj3, viewer, path_rrt, "rrt",
        main_cam_handle=main_cam_handle, camera_props=camera_props, record_rgb=frames_rrt, planner_playback=args.planner_playback
    )
    result["rrtconnect"]["execution"] = exec_rrt
    if want_video and frames_rrt:
        _save_mp4_rgb(frames_rrt, mp4_rrt, fps=args.video_fps)
        result["rrtconnect"]["video_path"] = mp4_rrt
    elif want_video:
        result["rrtconnect"]["video_path"] = None

    reset_arm_to_q(gym, sim, env, ur, spj, slj, ej, wj1, wj2, wj3, viewer, q_start_live, n_steps=200)

    t_nt0 = time.perf_counter()
    path_nt_raw = ntfield_plan(
        nt_model, q_start_live, grasp_target_q, step_size=args.ntfield_step_size, max_steps=args.ntfield_max_steps, tol=args.ntfield_tol, device=ntfield_device_str
    )
    nt_planning_wall_s = float(time.perf_counter() - t_nt0)

    nt_err_pen = None
    if path_nt_raw and len(path_nt_raw) >= 2:
        nt_err_pen = float(np.linalg.norm(np.asarray(path_nt_raw[-2], dtype=np.float64).reshape(6) - grasp_target_q))
    nt_has_path = path_nt_raw is not None and len(path_nt_raw) >= 2
    nt_converged = nt_err_pen is not None and nt_err_pen < goal_eps

    result["ntfield"] = {
        "planning_wall_s": nt_planning_wall_s,
        "success": nt_has_path,
        "converged_within_tol": nt_converged,
        "goal_eps_rad": goal_eps,
        "penultimate_goal_error_rad": nt_err_pen,
        "num_planner_steps_including_appended_goal": len(path_nt_raw) if path_nt_raw else 0,
        "trajectory_waypoints_rad": None,
    }

    if path_nt_raw and len(path_nt_raw) >= 2:
        path_nt = _path_as_6_list(path_nt_raw)
        if args.ntfield_waypoint_mode == "two_point" and len(path_nt) >= 2:
            path_nt = [path_nt[0], path_nt[-1]]
        elif args.ntfield_fixed_waypoints > 0:
            path_nt = _resample_path_fixed_waypoints(path_nt, args.ntfield_fixed_waypoints)
        result["ntfield"]["trajectory_waypoints_rad"] = path_nt
        result["ntfield"]["num_waypoints_after_postprocess"] = len(path_nt)
        result["ntfield"]["motion"] = joint_metrics(path_nt, q_start_live, grasp_target_q)
        nt_terminal_check = planner_terminal_ee_check(
            rac,
            path_nt,
            flex_collision_models,
            target_idx=target_idx,
            spawned_object_asset_indices=target_file_idx.tolist(),
            center_mode=args.ee_proxy_center_mode,
            max_radius_m=args.ee_proxy_max_radius_m,
        )
        result["ntfield"]["ee_object_pair_collision_terminal_waypoint"] = nt_terminal_check
        if args.require_ee_object_collision and not nt_terminal_check["success"]:
            result["ntfield"]["success"] = False
            result["ntfield"]["ee_object_postcheck_failed"] = True
        frames_nt = [] if want_video else None
        exec_nt = execute_path_and_time(
            gym, sim, env, ur, spj, slj, ej, wj1, wj2, wj3, viewer, path_nt, "ntfield",
            main_cam_handle=main_cam_handle, camera_props=camera_props, record_rgb=frames_nt, planner_playback=args.planner_playback
        )
        result["ntfield"]["execution"] = exec_nt
        if want_video and frames_nt:
            _save_mp4_rgb(frames_nt, mp4_nt, fps=args.video_fps)
            result["ntfield"]["video_path"] = mp4_nt
        elif want_video:
            result["ntfield"]["video_path"] = None
    else:
        result["ntfield"]["execution"] = None
        result["ntfield"]["ee_object_pair_collision_terminal_waypoint"] = {
            "evaluated": False,
            "target": None,
            "obstacles": [],
            "no_obstacle_collision": False,
            "success": False,
            "ee_center_world_m": None,
            "radius_m": None,
        }
        if args.require_ee_object_collision:
            result["ntfield"]["ee_object_postcheck_failed"] = True
        if want_video:
            result["ntfield"]["video_path"] = None

    if args.save_final_geometric_debug and grasp_target_q is not None and len(object_mesh) > 0:
        debug_png = (
            os.path.abspath(args.final_geometric_debug_path)
            if args.final_geometric_debug_path
            else os.path.join(session_dir, "final_geometric_debug.png")
        )
        ee_center_for_viz = result["ee_object_pair_collision_selected_grasp"].get("ee_center_world_m")
        ee_radius_for_viz = result["ee_object_pair_collision_proxy"].get("radius_m")
        if ee_center_for_viz is None:
            ee_center_for_viz = list(np.asarray(rac.calculate_transform_from_angles(grasp_target_q)[8][0], dtype=np.float64).reshape(3))
        save_final_geometric_debug_image_multi(
            out_path=debug_png,
            rac=rac,
            dof_result=grasp_target_q,
            object_meshes=object_mesh,
            target_idx=target_idx,
            ee_center_world=ee_center_for_viz,
            ee_radius_m=ee_radius_for_viz,
        )
        result["ee_object_pair_collision_selected_grasp"]["debug_image_path"] = debug_png

    print(json.dumps(result, indent=2))
    if args.output_json:
        with open(args.output_json, "w") as jf:
            json.dump(result, jf, indent=2)

    gym.destroy_sim(sim)
    if viewer is not None:
        gym.destroy_viewer(viewer)
    os.chdir(_invoke_cwd)


if __name__ == "__main__":
    main()
