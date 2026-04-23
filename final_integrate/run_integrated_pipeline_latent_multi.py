#!/usr/bin/env python3
"""
Multi-object test harness for latent-goal NTField planning.

This script keeps the multi-object environment setup from
`run_integrated_pipeline_multi.py` (3 random YCB objects) and replaces the
goal generation with image->latent inference from
`run_integrated_pipeline_latent.py`.

It also adds `--ntfield_delta_clamp_rad` for latent planning stability.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_PI_VLA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HANWEN_GRASPING_ROOT = os.path.join(_PI_VLA_ROOT, "hanwen_grasping")
_COLLECT_DATA_DIR = os.path.join(HANWEN_GRASPING_ROOT, "collect_data")
_UTIL_DIR = os.path.join(_COLLECT_DATA_DIR, "util")
_GRASP_UTIL_DIR = os.path.join(_COLLECT_DATA_DIR, "grasp_util")
_NTRL_DEMO = os.path.join(_PI_VLA_ROOT, "ntrl-demo")

for _p in (HANWEN_GRASPING_ROOT, _UTIL_DIR, _GRASP_UTIL_DIR, _PI_VLA_ROOT, _NTRL_DEMO):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _resolve_under_root(path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(_PI_VLA_ROOT, path))


def _grasp_asset_path_to_display_name(rel_path: str) -> str:
    object_display_names = {
        "002_master_chef_can": "master chef can",
        "004_sugar_box": "sugar box",
        "005_tomato_soup_can": "tomato soup can",
        "006_mustard_bottle": "mustard bottle",
        "036_wood_block": "wood block",
        "011_banana": "banana",
    }
    rel = rel_path.replace("\\", "/")
    folder = rel.split("/")[-2] if "/" in rel else ""
    stem = os.path.splitext(os.path.basename(rel))[0]
    for key in (folder, stem):
        if key and key in object_display_names:
            return object_display_names[key]
    label = stem or folder
    if len(label) > 4 and label[:3].isdigit() and label[3] == "_":
        label = label[4:]
    return label.replace("_", " ").strip()


def _plan_with_goal_latent_delta(
    network: Any,
    q_start: np.ndarray,
    z_goal: np.ndarray,
    scale: float,
    step_size: float,
    max_steps: int,
    grad_tol: float,
    delta_clamp_rad: float,
    device: str,
) -> List[np.ndarray]:
    """Latent planning with optional L-infinity per-step clamp in radians."""
    import torch

    q_start = np.asarray(q_start, dtype=np.float64).reshape(-1)
    dim = int(q_start.shape[0])
    q_start_norm = q_start / scale

    q_tensor = torch.tensor(q_start_norm, dtype=torch.float32, device=device).unsqueeze(0)

    if not isinstance(z_goal, torch.Tensor):
        z_goal = torch.tensor(z_goal, dtype=torch.float32, device=device)
    if z_goal.dim() == 1:
        z_goal = z_goal.unsqueeze(0)

    delta_clamp_norm = 0.0
    if delta_clamp_rad > 0.0:
        delta_clamp_norm = float(delta_clamp_rad / scale)

    path_norm = [q_start_norm.copy()]
    for _ in range(max_steps):
        q_inp = q_tensor.detach().requires_grad_(True)
        tau, _, coords = network.out_with_goal_latent(q_inp, z_goal)
        if tau.item() < 1e-4:
            break

        dtau_tuple = torch.autograd.grad(
            outputs=tau,
            inputs=coords,
            grad_outputs=torch.ones_like(tau),
            only_inputs=True,
            allow_unused=True,
        )
        dtau = dtau_tuple[0]
        if dtau is None or dtau.abs().max().item() < 1e-12:
            break

        grad_q = dtau[:, :dim]
        grad_mag = torch.norm(grad_q, dim=1, keepdim=True)
        if grad_mag.item() < grad_tol:
            break

        update_dir = -grad_q / (grad_mag**2 + 1e-8)
        delta = step_size * update_dir

        if delta_clamp_norm > 0.0:
            max_abs = torch.max(torch.abs(delta)).item()
            if max_abs > delta_clamp_norm:
                delta = delta * (delta_clamp_norm / max_abs)

        with torch.no_grad():
            q_tensor = q_inp.detach() + delta
        path_norm.append(q_tensor[0].detach().cpu().numpy().copy())

    return [p * scale for p in path_norm]


def main() -> None:
    from scipy.spatial.transform import Rotation as R
    from isaacgym import gymapi
    from isaacgym import gymutil

    import cv2
    import fcl
    import torch
    import robot_arm_configuration as RC
    from obj_reader import obj_reader
    from stl_reader import stl_reader

    from final_integrate.run_integrated_pipeline_latent import (
        _compute_z_goal,
        _infer_latent_on_image,
    )
    from final_integrate.run_integrated_pipeline_multi import find_grasp_q_goal
    from trajectory_evaluation.comparison.run_rrt_ntfield_benchmark import (
        DRAWER_HEIGHT,
        TABLE_DIMS_X,
        TABLE_DIMS_Y,
        TABLE_DIMS_Z,
        _path_as_6_list,
        _save_mp4_rgb,
        execute_path_and_time,
        get_swept_volume_size,
        reset_arm_to_q,
        sim_dt,
    )
    from trajectory_evaluation.ntfield.eval_trajectory_ntfield import load_network_and_function
    from planning.gradient_planner_trajectory import SCALE as NTFIELD_SCALE

    num_of_objects = 3
    target_obj_index = [1, 3, 5]
    home_dof = [0.7, -2.0, 2.5, -0.3, 0.7, 0.0]
    start_settle_steps = 30

    parser = argparse.ArgumentParser(description="PI-VLA latent planner in multi-object scene")
    parser.add_argument("--ntfield_checkpoint", type=str, required=True)
    parser.add_argument("--ntfield_experiment_dir", type=str, default=None)
    parser.add_argument("--latent_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--target_object_idx", type=int, default=0, choices=(0, 1, 2))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use_viewer", action="store_true")
    parser.add_argument("--ntfield_device", type=str, default="cuda:0")
    parser.add_argument("--latent_device", type=str, default="auto")
    parser.add_argument("--ntfield_step_size", type=float, default=0.02)
    parser.add_argument("--ntfield_max_steps", type=int, default=200)
    parser.add_argument("--ntfield_grad_tol", type=float, default=1e-3)
    parser.add_argument("--ntfield_delta_clamp_rad", type=float, default=0.0)
    parser.add_argument("--video_fps", type=float, default=60.0)
    parser.add_argument(
        "--planner_playback",
        type=str,
        choices=("direct", "settle"),
        default="direct",
    )
    args, argv_remainder = parser.parse_known_args()

    argv_gym = list(argv_remainder)
    if not args.use_viewer and "--headless" not in argv_gym:
        argv_gym.append("--headless")
    sys.argv = [sys.argv[0]] + argv_gym

    if args.seed is not None:
        np.random.seed(args.seed)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        session_dir = os.path.join(os.path.abspath(args.output_dir), stamp)
    else:
        session_dir = os.path.join(_PI_VLA_ROOT, "output", "final_integrate", stamp)
    os.makedirs(session_dir, exist_ok=True)

    ckpt_abs = _resolve_under_root(args.ntfield_checkpoint)
    if not os.path.isfile(ckpt_abs):
        raise SystemExit(f"NTField checkpoint not found: {ckpt_abs}")
    dev_nt = torch.device(
        "cpu" if args.ntfield_device == "cpu" or not torch.cuda.is_available() else args.ntfield_device
    )
    nt_net, _ = load_network_and_function(ckpt_abs, args.ntfield_experiment_dir, dev_nt, dim=6)
    ntfield_device_str = str(dev_nt) if dev_nt.type == "cuda" else "cpu"

    _cwd_prev = os.getcwd()
    os.chdir(HANWEN_GRASPING_ROOT)

    gym = gymapi.acquire_gym()
    gym_args = gymutil.parse_arguments(description="latent_multi", headless=True, custom_parameters=[])
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

    sim = gym.create_sim(
        gym_args.compute_device_id,
        gym_args.graphics_device_id,
        gym_args.physics_engine,
        sim_params,
    )
    if sim is None:
        raise SystemExit("Failed to create sim")

    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)

    asset_root = "./assets/"
    object_asset_files = []
    object_collision_files = []
    object_offset = []
    object_common_prefix = "urdf/ycb/"
    with open(asset_root + "urdf/ycb/object_urdf_grasp.txt", encoding="utf-8") as f:
        for line in f:
            object_asset_files.append(object_common_prefix + line.rstrip())
    with open(asset_root + "urdf/ycb/object_collision_grasp.txt", encoding="utf-8") as f:
        for line in f:
            object_collision_files.append(object_common_prefix + line.rstrip())
    with open(asset_root + "urdf/ycb/object_offset_grasp.txt", encoding="utf-8") as f:
        for line in f:
            object_offset.append([float(x) for x in line.rstrip().split()])

    ur5e_collision_parts = [
        "urdf/ur5e/meshes/collision/base.stl",
        "urdf/ur5e/meshes/collision/shoulder.stl",
        "urdf/ur5e/meshes/collision/upperarm.stl",
        "urdf/ur5e/meshes/collision/forearm.stl",
        "urdf/ur5e/meshes/collision/wrist1.stl",
        "urdf/ur5e/meshes/collision/wrist2.stl",
        "urdf/ur5e/meshes/collision/wrist3.stl",
    ]
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
    ur5e_collision_models = []
    for idx, parts_path in enumerate(ur5e_collision_parts):
        mesh = stl_reader(asset_root + parts_path)
        mesh.transform(ur5e_rotations[idx], ur5e_translations[idx])
        verts, tris = mesh.get_vertices(), mesh.get_faces()
        m = fcl.BVHModel()
        m.beginModel(len(verts), len(tris))
        m.addSubModel(verts, tris)
        m.endModel()
        ur5e_collision_models.append(m)

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)
    asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
    asset_options.use_mesh_materials = True
    ur5e_asset = gym.load_asset(sim, asset_root, "urdf/ur5e/ur5e_mimic_real_gripper_test.urdf", asset_options)
    table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)
    asset_options.fix_base_link = False
    object_assets = [gym.load_asset(sim, asset_root, ob, asset_options) for ob in object_asset_files]

    ur5e_pose = gymapi.Transform()
    ur5e_pose.p = gymapi.Vec3(0, 0, 0)
    ur5e_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0.5 * math.pi)
    table_pose = gymapi.Transform()
    table_pose.p = gymapi.Vec3(table_dims.x * 0.5 + 0.3, 0.0, table_dims.z * 0.5)

    table_x_min = table_pose.p.x - table_dims.x * 0.5 + 0.05
    table_x_max = table_pose.p.x + table_dims.x * 0.5 - 0.10
    table_y_min = table_pose.p.y - table_dims.y * 0.5 + 0.10
    table_y_max = table_pose.p.y + table_dims.y * 0.5 - 0.20

    camera_focus = gymapi.Vec3(0, 0, 0)
    camera_props = gymapi.CameraProperties()
    camera_props.horizontal_fov = 70.25
    camera_props.width = 1280
    camera_props.height = 720

    col_plane = fcl.Plane(np.array([0.0, 0.0, 1.0]), 0)
    plane_obj = fcl.CollisionObject(col_plane, fcl.Transform())
    col_table = fcl.Box(table_dims.x, table_dims.y, table_dims.z)
    trans_table = fcl.Transform(np.array([table_dims.x * 0.5 + 0.3, 0.0, table_dims.z * 0.5]))
    table_obj = fcl.CollisionObject(col_table, trans_table)
    object_collision_models = [table_obj]

    env = gym.create_env(sim, gymapi.Vec3(-2, -2, 0), gymapi.Vec3(2, 2, 0), 1)
    ur = gym.create_actor(env, ur5e_asset, ur5e_pose, "ur5e", 0, 32767)
    spj = gym.find_actor_dof_handle(env, ur, "shoulder_pan_joint")
    slj = gym.find_actor_dof_handle(env, ur, "shoulder_lift_joint")
    ej = gym.find_actor_dof_handle(env, ur, "elbow_joint")
    wj1 = gym.find_actor_dof_handle(env, ur, "wrist_1_joint")
    wj2 = gym.find_actor_dof_handle(env, ur, "wrist_2_joint")
    wj3 = gym.find_actor_dof_handle(env, ur, "wrist_3_joint")
    gym.create_actor(env, table_asset, table_pose, "table", 0, 1)

    target_file_idx = np.random.choice(target_obj_index, num_of_objects, replace=False)
    object_slot_names = [_grasp_asset_path_to_display_name(object_asset_files[int(target_file_idx[k])]) for k in range(num_of_objects)]
    print(f"Objects: {object_slot_names}")
    print(f"Target: [{args.target_object_idx}] {object_slot_names[args.target_object_idx]}")

    object_scaling_factor = np.ones(num_of_objects)
    object_handles = []
    object_collision_lib = []
    object_status_list = []
    object_reader_tracker = []
    object_mesh = []
    obstacle_objs = []
    gt_obj_pos_list = []
    objs_manager = fcl.DynamicAABBTreeCollisionManager()

    for k in range(num_of_objects):
        is_collision = True
        tx = ty = tz = 0.0
        while is_collision:
            tx = np.random.uniform(table_x_min, table_x_max)
            ty = np.random.uniform(table_y_min, table_y_max)
            tz = TABLE_DIMS_Z + 0.08

            file_path = object_collision_files[target_file_idx[k]]
            collision_mesh = obj_reader(asset_root + file_path)
            collision_mesh.set_scale(object_scaling_factor[k])
            collision_mesh.add_offset(object_offset[target_file_idx[k]])
            verts, tris = collision_mesh.get_bounding_box_mesh()
            temp_center = collision_mesh.get_center()
            temp_bbox = collision_mesh.get_bounding_box()

            m = fcl.BVHModel()
            m.beginModel(len(verts), len(tris))
            m.addSubModel(verts, tris)
            m.endModel()
            temp_co = fcl.CollisionObject(m, fcl.Transform(np.array([tx, ty, tz])))

            req = fcl.CollisionRequest()
            rdata = fcl.CollisionData(request=req)
            objs_manager.collide(temp_co, rdata, fcl.defaultCollisionCallback)
            is_collision = rdata.result.is_collision
            if not is_collision:
                for obj_pos in gt_obj_pos_list:
                    if np.sqrt((tx - obj_pos[0]) ** 2 + (ty - obj_pos[1]) ** 2) <= 0.16:
                        is_collision = True
                        break

        object_pose = gymapi.Transform()
        object_pose.p = gymapi.Vec3(tx, ty, tz)
        handle = gym.create_actor(env, object_assets[target_file_idx[k]], object_pose, f"object{k}", 0, 2 ** (k + 1), k + 1)
        gym.set_actor_scale(env, handle, object_scaling_factor[k])
        object_handles.append(handle)
        object_collision_lib.append(m)
        object_status_list.append([temp_center, temp_bbox])
        object_reader_tracker.append(collision_mesh)
        obstacle_objs.append(temp_co)
        gt_obj_pos_list.append([tx, ty])
        objs_manager.registerObjects(obstacle_objs)
        objs_manager.setup()

    top_cam_handle = gym.create_camera_sensor(env, camera_props)
    top_cam_pos = gymapi.Vec3(table_pose.p.x, table_pose.p.y + 0.001, 2.0)
    top_cam_target = gymapi.Vec3(table_pose.p.x - 0.5, table_pose.p.y, table_pose.p.z)
    gym.set_camera_location(top_cam_handle, env, top_cam_pos, top_cam_target)

    main_cam_handle = gym.create_camera_sensor(env, camera_props)
    gym.set_camera_location(main_cam_handle, env, gymapi.Vec3(3, 0, 0.3), camera_focus)

    viewer = None
    if not gym_args.headless:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())

    gym.set_light_parameters(sim, 0, gymapi.Vec3(0.3, 0.3, 0.3), gymapi.Vec3(1, 1, 1), gymapi.Vec3(-1, 0, 0))
    gym.set_light_parameters(sim, 1, gymapi.Vec3(0.3, 0.3, 0.3), gymapi.Vec3(1, 1, 1), gymapi.Vec3(1, 0, 0))

    original_centers = [s[0].copy() for s in object_status_list]
    real_position = False
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
                rotation = np.array(np.array(states[0][0][1]).item())
                translation = np.array(np.array(states[0][0][0]).item())
                object_status_list[ii][0] = original_centers[ii] + translation
                r1 = R.from_quat(rotation)
                tf = fcl.Transform(r1.as_matrix(), translation)
                object_collision_models.append(fcl.CollisionObject(object_collision_lib[ii], tf))
                tmp = object_reader_tracker[ii]
                tmp.set_offset(translation)
                verts, faces = tmp.get_bounding_box_mesh()
                object_mesh.append([verts, faces])

        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        if viewer is not None:
            gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    for _ in range(start_settle_steps):
        gym.set_dof_target_position(env, spj, home_dof[0])
        gym.set_dof_target_position(env, slj, home_dof[1])
        gym.set_dof_target_position(env, ej, home_dof[2])
        gym.set_dof_target_position(env, wj1, home_dof[3])
        gym.set_dof_target_position(env, wj2, home_dof[4])
        gym.set_dof_target_position(env, wj3, home_dof[5])
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        if viewer is not None:
            gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    gym.render_all_camera_sensors(sim)
    raw_top = gym.get_camera_image(sim, env, top_cam_handle, gymapi.IMAGE_COLOR)
    rgba_top = raw_top.reshape(camera_props.height, camera_props.width, 4)
    rgb_top = rgba_top[..., :3].copy()
    top_view_path = os.path.join(session_dir, "top_view.png")
    cv2.imwrite(top_view_path, cv2.cvtColor(rgb_top, cv2.COLOR_RGB2BGR))
    print(f"Top view saved: {top_view_path}")

    true_locations = [object_status_list[k][0].tolist() for k in range(num_of_objects)]
    with open(os.path.join(session_dir, "object_locations_true.json"), "w", encoding="utf-8") as f:
        json.dump({"objects": object_slot_names, "xyz_m": true_locations}, f, indent=2)

    scene_info = [table_dims.x, table_dims.y, table_dims.z, drawer_height]
    rac = RC.robot_arm_configuration(
        "./assets/urdf/ur5e/meshes/collision/",
        np.array([ur5e_pose.p.x, ur5e_pose.p.y, ur5e_pose.p.z]),
        scene_info,
    )

    target_idx = args.target_object_idx
    grasp_file = "./assets/" + "/".join(object_asset_files[target_file_idx[target_idx]].split("/")[:-1]) + "/grasp_dict.npy"
    grasp_data = np.load(grasp_file, allow_pickle=True)
    grasp_list = np.arange(len(grasp_data))
    np.random.shuffle(grasp_list)

    true_xy = np.array(true_locations[target_idx][:2], dtype=np.float64)
    q_goal_true, _, _ = find_grasp_q_goal(
        rac,
        RC,
        scene_info,
        grasp_data,
        grasp_list,
        true_xy,
        0,
        object_mesh,
        object_collision_models,
        plane_obj,
        get_swept_volume_size,
    )

    dof_snap = gym.get_actor_dof_states(env, ur, gymapi.STATE_POS)
    q_start_live = np.array(dof_snap["pos"][:6], dtype=np.float64)

    latent_goal_pred = _infer_latent_on_image(
        rgb_top,
        _resolve_under_root(args.latent_checkpoint),
        args.latent_device,
    )
    latent_goal_true: Optional[np.ndarray] = None
    if q_goal_true is not None:
        latent_goal_true = _compute_z_goal(
            nt_net,
            q_start_live.reshape(1, -1),
            q_goal_true.reshape(1, -1),
            dev_nt,
        )

    summary: Dict[str, Any] = {
        "session_dir": session_dir,
        "objects": object_slot_names,
        "target_object": object_slot_names[target_idx],
        "q_start_live": q_start_live.tolist(),
        "q_goal_true_found": q_goal_true is not None,
        "videos": {},
        "ntfield_delta_clamp_rad": float(args.ntfield_delta_clamp_rad),
    }

    if latent_goal_true is not None:
        latent_diff = latent_goal_pred - latent_goal_true
        summary["latent_goal_comparison"] = {
            "mse": float(np.mean(np.square(latent_diff))),
            "l2": float(np.linalg.norm(latent_diff)),
            "max_abs": float(np.max(np.abs(latent_diff))),
        }
    else:
        summary["latent_goal_comparison"] = None

    def _run_latent_video(z_goal: Optional[np.ndarray], out_mp4: str, label: str) -> None:
        if z_goal is None:
            summary["videos"][label] = None
            return
        reset_arm_to_q(gym, sim, env, ur, spj, slj, ej, wj1, wj2, wj3, viewer, q_start_live, n_steps=200)
        path_raw = _plan_with_goal_latent_delta(
            network=nt_net,
            q_start=q_start_live,
            z_goal=z_goal.reshape(1, -1),
            scale=float(NTFIELD_SCALE),
            step_size=float(args.ntfield_step_size),
            max_steps=int(args.ntfield_max_steps),
            grad_tol=float(args.ntfield_grad_tol),
            delta_clamp_rad=float(args.ntfield_delta_clamp_rad),
            device=ntfield_device_str,
        )
        if not path_raw or len(path_raw) < 2:
            summary["videos"][label] = None
            return
        path = _path_as_6_list(path_raw)
        frames: List[np.ndarray] = []
        execute_path_and_time(
            gym,
            sim,
            env,
            ur,
            spj,
            slj,
            ej,
            wj1,
            wj2,
            wj3,
            viewer,
            path,
            label,
            main_cam_handle=main_cam_handle,
            camera_props=camera_props,
            record_rgb=frames,
            planner_playback=args.planner_playback,
        )
        _save_mp4_rgb(frames, out_mp4, fps=args.video_fps)
        summary["videos"][label] = out_mp4
        print(f"[{label}] saved: {out_mp4}")

    _run_latent_video(latent_goal_pred, os.path.join(session_dir, "ntfield_predicted_latent_goal.mp4"), "predicted_latent_goal")
    _run_latent_video(latent_goal_true, os.path.join(session_dir, "ntfield_true_latent_goal.mp4"), "true_latent_goal")

    with open(os.path.join(session_dir, "q_goal_true.json"), "w", encoding="utf-8") as f:
        json.dump({"joint_rad": None if q_goal_true is None else q_goal_true.tolist()}, f, indent=2)
    with open(os.path.join(session_dir, "latent_goal_pred.json"), "w", encoding="utf-8") as f:
        json.dump({"latent_goal": latent_goal_pred.tolist()}, f, indent=2)
    with open(os.path.join(session_dir, "latent_goal_true.json"), "w", encoding="utf-8") as f:
        json.dump({"latent_goal": None if latent_goal_true is None else latent_goal_true.tolist()}, f, indent=2)
    with open(os.path.join(session_dir, "pipeline_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    gym.destroy_sim(sim)
    if viewer is not None:
        gym.destroy_viewer(viewer)
    os.chdir(_cwd_prev)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
