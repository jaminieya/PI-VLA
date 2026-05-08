#!/usr/bin/env python3
"""
PI-VLA integration — true latent goal (z_goal_truth) only.

Skips image capture, student model inference, and predicted-latent planning.
Runs NTField gradient descent directly toward z(q_start, q_goal_true).

Usage::

  python final_integrate/ntfield_planning.py \
    --ntfield_checkpoint "/home/hojinsohn/VLM-NT/PI-VLA/teacher_model.pt" \
    --output_dir "output/true_latent_run" \
    --seed 1007 \
    --ntfield_step_size 0.02 \
    --ntfield_max_steps 200 \
    --ntfield_tol 0.01
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
_STUDENT_TRAINING_DIR = os.path.join(_PI_VLA_ROOT, "student_model_training")

for _p in (
    HANWEN_GRASPING_ROOT,
    _UTIL_DIR,
    _GRASP_UTIL_DIR,
    _PI_VLA_ROOT,
    _NTRL_DEMO,
    _STUDENT_TRAINING_DIR,
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OBJECT_HEIGHTS_M: Dict[int, float] = {
    1: 0.176887,  # 004_sugar_box
    3: 0.192159,  # 006_mustard_bottle
    5: 0.037270,  # 011_banana
}
OBJECT_HEIGHT_DEFAULT_M: float = 0.10
_GRIPPER_FCL_MODEL_INDEX = 8


def _resolve_under_root(path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(_PI_VLA_ROOT, path))


def _sim_ee_pose_for_gripper_fcl(gym, env, ur):
    from isaacgym import gymapi

    rb = gym.get_actor_rigid_body_dict(env, ur)
    st = gym.get_actor_rigid_body_states(env, ur, gymapi.STATE_POS)

    def _body_T(name):
        idx = rb.get(name)
        if idx is None:
            return None
        T = gymapi.Transform.from_buffer(st["pose"][int(idx)])
        trans = np.array([T.p.x, T.p.y, T.p.z], dtype=np.float64)
        quat  = np.array([T.r.x, T.r.y, T.r.z, T.r.w], dtype=np.float64)
        return trans, quat

    for link in ("robotiq_arg2f_base_link", "wrist_3_link", "tool0", "ee_link"):
        got = _body_T(link)
        if got is not None:
            return got[0], got[1], link

    li, wi = rb.get("left_inner_finger"), rb.get("wrist_3_link")
    ri = rb.get("right_inner_finger")
    if li is not None and ri is not None and wi is not None:
        pl = gymapi.Transform.from_buffer(st["pose"][int(li)]).p
        pr = gymapi.Transform.from_buffer(st["pose"][int(ri)]).p
        trans = np.array(
            [0.5*(pl.x+pr.x), 0.5*(pl.y+pr.y), 0.5*(pl.z+pr.z)],
            dtype=np.float64,
        )
        Tw   = gymapi.Transform.from_buffer(st["pose"][int(wi)])
        quat = np.array([Tw.r.x, Tw.r.y, Tw.r.z, Tw.r.w], dtype=np.float64)
        return trans, quat, "finger_midpoint+wrist3_quat"

    return None, None, "no_ee_link"




def main() -> None:
    from scipy.spatial.transform import Rotation as R
    from isaacgym import gymapi, gymutil

    import fcl
    import torch
    import cv2
    import robot_arm_configuration as RC
    from stl_reader import stl_reader
    from obj_reader import obj_reader

    from trajectory_evaluation.comparison.run_rrt_ntfield_benchmark import (
        TABLE_DIMS_X, TABLE_DIMS_Y, TABLE_DIMS_Z, DRAWER_HEIGHT,
        execute_path_and_time, reset_arm_to_q, _save_mp4_rgb, _path_as_6_list, sim_dt,
    )
    from trajectory_evaluation.ntfield.eval_trajectory_ntfield import (
        _ModelShim, load_network_and_function,
    )
    from final_integrate.run_integrated_pipeline_latent import _compute_z_goal
    from planning.gradient_planner_trajectory import SCALE as NTFIELD_SCALE

    parser = argparse.ArgumentParser(description="PI-VLA — true latent goal only")
    parser.add_argument("--ntfield_checkpoint",    type=str, required=True)
    parser.add_argument("--ntfield_experiment_dir", type=str, default=None)
    parser.add_argument("--output_dir",            type=str, default=None)
    parser.add_argument("--object_z",              type=float, default=0.18)
    parser.add_argument("--target_obj_indices",    type=str,   default="1,3,5")
    parser.add_argument("--num_objects",           type=int,   default=3)
    parser.add_argument("--ox_min",  type=float, default=0.42)
    parser.add_argument("--ox_max",  type=float, default=0.98)
    parser.add_argument("--oy_min",  type=float, default=-0.38)
    parser.add_argument("--oy_max",  type=float, default=0.38)
    parser.add_argument("--seed",    type=int,   default=None)
    parser.add_argument("--use_viewer", action="store_true")
    parser.add_argument("--ntfield_device", type=str, default="cuda:0")
    parser.add_argument("--physx_cpu",      action="store_true")
    parser.add_argument("--ntfield_step_size",  type=float, default=0.02)
    parser.add_argument("--ntfield_max_steps",  type=int,   default=200)
    parser.add_argument("--ntfield_tol",        type=float, default=0.01)
    parser.add_argument("--ntfield_delta_clamp_rad",      type=float, default=0.0)
    parser.add_argument("--ntfield_refine_max_steps",     type=int,   default=-1)
    parser.add_argument("--ntfield_refine_step_size",     type=float, default=None)
    parser.add_argument("--ntfield_refine_step_size_factor", type=float, default=None)
    parser.add_argument("--ntfield_refine_delta_clamp_rad",  type=float, default=None)
    parser.add_argument("--ntfield_stagnate_max_steps",   type=int,   default=400)
    parser.add_argument("--ntfield_stagnate_patience",    type=int,   default=30)
    parser.add_argument("--ntfield_stagnate_rel_eps",     type=float, default=5e-4)
    parser.add_argument("--ntfield_stagnate_step_size",   type=float, default=None)
    parser.add_argument("--video_fps", type=float, default=60.0)
    parser.add_argument("--planner_playback", type=str, choices=("direct", "settle"), default="direct")
    parser.add_argument("--between_axis_margin_m",   type=float, default=0.005)
    parser.add_argument("--between_lateral_thresh_m", type=float, default=0.02)
    parser.add_argument("--between_z_thresh_m",      type=float, default=0.05)
    parser.add_argument("--no_isaac_hard_exit", action="store_true")
    args, argv_remainder = parser.parse_known_args()

    argv_gym = list(argv_remainder)
    if not args.use_viewer and "--headless" not in argv_gym:
        argv_gym.append("--headless")
    sys.argv = [sys.argv[0]] + argv_gym

    if args.seed is not None:
        np.random.seed(args.seed)

    target_obj_index: List[int] = [
        int(x.strip()) for x in str(args.target_obj_indices).split(",") if x.strip()
    ]
    num_objects = int(args.num_objects)
    if num_objects <= 0:
        raise SystemExit("--num_objects must be >= 1")
    if len(target_obj_index) < num_objects:
        raise SystemExit(
            f"Need at least {num_objects} indices in --target_obj_indices, "
            f"got {len(target_obj_index)}"
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(
        os.path.abspath(args.output_dir) if args.output_dir
        else os.path.join(_PI_VLA_ROOT, "output", "true_latent_only"),
        stamp,
    )
    os.makedirs(session_dir, exist_ok=True)

    oz       = float(args.object_z)
    ckpt_abs = _resolve_under_root(args.ntfield_checkpoint)
    if not os.path.isfile(ckpt_abs):
        raise SystemExit(f"NTField checkpoint not found: {ckpt_abs}")

    _cwd_prev = os.getcwd()
    os.chdir(HANWEN_GRASPING_ROOT)

    # ── Isaac Gym setup ──────────────────────────────────────────────────────
    gym = gymapi.acquire_gym()
    gym_args = gymutil.parse_arguments(description="true_latent_only", headless=True, custom_parameters=[])
    gym_args.headless = not args.use_viewer
    if args.physx_cpu:
        gym_args.use_gpu = False

    table_dims   = gymapi.Vec3(TABLE_DIMS_X, TABLE_DIMS_Y, TABLE_DIMS_Z)
    drawer_height = DRAWER_HEIGHT

    sim_params = gymapi.SimParams()
    sim_params.substeps = 2
    sim_params.dt       = sim_dt
    sim_params.up_axis  = gymapi.UP_AXIS_Z
    sim_params.gravity  = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.physx.solver_type            = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = gym_args.num_threads
    sim_params.physx.use_gpu     = gym_args.use_gpu
    sim_params.use_gpu_pipeline  = False

    sim = gym.create_sim(
        gym_args.compute_device_id, gym_args.graphics_device_id,
        gym_args.physics_engine, sim_params,
    )
    if sim is None:
        raise SystemExit("Failed to create sim")

    plane_params        = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)

    dev_nt = torch.device(
        "cpu" if args.ntfield_device == "cpu" or not torch.cuda.is_available()
        else args.ntfield_device
    )
    nt_net, ntfield_fn = load_network_and_function(ckpt_abs, args.ntfield_experiment_dir, dev_nt, dim=6)
    nt_model           = _ModelShim(ntfield_fn)
    ntfield_device_str = str(dev_nt) if dev_nt.type == "cuda" else "cpu"

    asset_root       = "./assets/"
    ur5e_asset_file  = "urdf/ur5e/ur5e_mimic_real_gripper_test.urdf"
    ur5e_collision_parts = [
        "urdf/ur5e/meshes/collision/base.stl",
        "urdf/ur5e/meshes/collision/shoulder.stl",
        "urdf/ur5e/meshes/collision/upperarm.stl",
        "urdf/ur5e/meshes/collision/forearm.stl",
        "urdf/ur5e/meshes/collision/wrist1.stl",
        "urdf/ur5e/meshes/collision/wrist2.stl",
        "urdf/ur5e/meshes/collision/wrist3.stl",
    ]
    object_asset_files:      List[str]         = []
    object_collision_files:  List[str]         = []
    object_offset:           List[List[float]] = []
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
        R.from_euler("x",  [90],        degrees=True),
        R.from_euler("xy", [90, 180],   degrees=True),
        R.from_euler("xy", [180, 180],  degrees=True),
        R.from_euler("z",  [-180],      degrees=True),
        R.from_euler("x",  [-180],      degrees=True),
        R.from_euler("x",  [90],        degrees=True),
        R.from_euler("z",  [-90],       degrees=True),
    ]
    ur5e_translations = [
        [0, 0, 0], [0, 0, 0], [0, -0.138, 0], [0, -0.007, 0],
        [0, 0.127, 0], [0, 0, 0], [0, 0, 0],
    ]
    for idx, parts_path in enumerate(ur5e_collision_parts):
        collision_mesh = stl_reader(asset_root + parts_path)
        m = fcl.BVHModel()
        collision_mesh.transform(ur5e_rotations[idx], ur5e_translations[idx])
        verts, tris = collision_mesh.get_vertices(), collision_mesh.get_faces()
        m.beginModel(len(verts), len(tris))
        m.addSubModel(verts, tris)
        m.endModel()
        ur5e_collision_models.append(m)

    viewer = None
    if not gym_args.headless:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is None:
            gym_args.headless = True

    spacing   = 2
    env_lower = gymapi.Vec3(-spacing, -spacing, 0)
    env_upper = gymapi.Vec3(spacing,  spacing,  0)

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link         = True
    asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)
    asset_options.mesh_normal_mode      = gymapi.COMPUTE_PER_VERTEX
    asset_options.use_mesh_materials    = True
    ur5e_asset  = gym.load_asset(sim, asset_root, ur5e_asset_file, asset_options)
    table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)
    asset_options.fix_base_link = False
    object_assets = [gym.load_asset(sim, asset_root, ob, asset_options) for ob in object_asset_files]

    ur5e_pose   = gymapi.Transform()
    ur5e_pose.p = gymapi.Vec3(0, 0, 0)
    ur5e_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0.5 * math.pi)
    table_pose  = gymapi.Transform()
    table_pose.p = gymapi.Vec3(table_dims.x * 0.5 + 0.3, 0.0, table_dims.z * 0.5)
    camera_focus = gymapi.Vec3(0, 0, 0)
    camera_props = gymapi.CameraProperties()
    camera_props.horizontal_fov = 70.25
    camera_props.width  = 1280
    camera_props.height = 720

    col_table   = fcl.Box(table_dims.x, table_dims.y, table_dims.z)
    trans_table = fcl.Transform(np.array([table_dims.x * 0.5 + 0.3, 0.0, table_dims.z * 0.5]))
    table_obj   = fcl.CollisionObject(col_table, trans_table)
    object_collision_models = [table_obj]
    table_x_min = table_pose.p.x - table_dims.x * 0.5 + 0.05
    table_x_max = table_pose.p.x + table_dims.x * 0.5 - 0.10
    table_y_min = table_pose.p.y - table_dims.y * 0.5 + 0.10
    table_y_max = table_pose.p.y + table_dims.y * 0.5 - 0.20
    plane_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    col_plane    = fcl.Plane(plane_normal, 0.0)
    plane_obj    = fcl.CollisionObject(col_plane, fcl.Transform())

    object_mesh:           List[Any] = []
    flex_collision_models: List[Any] = []
    envs:                  List[Any] = []
    ur5e_handles:          List[Any] = []
    object_handles:        List[Any] = []
    object_status_list:    List[Any] = []
    object_reader_tracker: List[Any] = []
    object_collision_lib:  List[Any] = []
    placed_object_locations: List[List[float]] = []
    spj = slj = ej = wj1 = wj2 = wj3 = None

    target_file_idx = np.random.choice(target_obj_index, num_objects, replace=False)
    main_cam_handle = None

    for i in range(1):
        envs.append(gym.create_env(sim, env_lower, env_upper, 1))
        ur5e_handles.append(gym.create_actor(envs[-1], ur5e_asset, ur5e_pose, "ur5e" + str(i), 0, 32767))
        spj = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "shoulder_pan_joint")
        slj = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "shoulder_lift_joint")
        ej  = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "elbow_joint")
        wj1 = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "wrist_1_joint")
        wj2 = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "wrist_2_joint")
        wj3 = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "wrist_3_joint")
        gym.create_actor(envs[-1], table_asset, table_pose, "table" + str(i), 0, 1)

        objs_manager    = fcl.DynamicAABBTreeCollisionManager()
        objs_manager.setup()
        obstacle_objs:      List[Any]         = []
        gt_obj_pos_list:    List[List[float]] = []
        gt_target_pos = [
            float(np.random.uniform(max(table_x_min, 0.20 + table_dims.x / 2), table_x_max)),
            float(np.random.uniform(table_y_min, table_y_max)),
            float(oz),
        ]
        object_scaling_factor = np.ones(num_objects, dtype=np.float64)

        for k in range(num_objects):
            object_pose = gymapi.Transform()
            file_path   = object_collision_files[target_file_idx[k]]
            collision_mesh = obj_reader(asset_root + file_path)
            collision_mesh.set_scale(object_scaling_factor[k])
            collision_mesh.add_offset(object_offset[target_file_idx[k]])
            verts, tris       = collision_mesh.get_bounding_box_mesh()
            temp_center       = collision_mesh.get_center()
            temp_bounding_box = collision_mesh.get_bounding_box()
            m = fcl.BVHModel()
            m.beginModel(len(verts), len(tris))
            m.addSubModel(verts, tris)
            m.endModel()

            is_collision = True
            tx = ty = 0.0
            while is_collision:
                tx = float(np.random.uniform(table_x_min, table_x_max))
                ty = float(np.random.uniform(table_y_min, table_y_max))
                t  = fcl.Transform(np.array([tx, ty, oz]))
                req   = fcl.CollisionRequest()
                rdata = fcl.CollisionData(request=req)
                objs_manager.collide(
                    fcl.CollisionObject(m, t), rdata, fcl.defaultCollisionCallback
                )
                is_collision = rdata.result.is_collision
                if not is_collision:
                    if float(np.sqrt((tx - gt_target_pos[0])**2 + (ty - gt_target_pos[1])**2)) <= 0.2:
                        is_collision = True
                        continue
                    for obj_xy in gt_obj_pos_list:
                        if float(np.sqrt((tx - obj_xy[0])**2 + (ty - obj_xy[1])**2)) <= 0.16:
                            is_collision = True
                            break

            object_pose.p = gymapi.Vec3(tx, ty, oz)
            gt_obj_pos_list.append([tx, ty])
            placed_object_locations.append([tx, ty, float(oz)])
            object_handles.append(
                gym.create_actor(
                    envs[-1], object_assets[target_file_idx[k]], object_pose,
                    "object" + str(k) + str(i), 0, 2 ** (k + 1), k + 1,
                )
            )
            gym.set_actor_scale(envs[-1], object_handles[-1], object_scaling_factor[k])
            object_reader_tracker.append(collision_mesh)
            object_status_list.append([temp_center, temp_bounding_box])
            object_collision_lib.append(m)
            obstacle_objs.append(fcl.CollisionObject(m, t))
            objs_manager.registerObjects(obstacle_objs)
            objs_manager.setup()

        main_cam_handle = gym.create_camera_sensor(envs[-1], camera_props)
        main_cam_pos    = gymapi.Vec3(3, 0, 0.3)
        gym.set_camera_location(main_cam_handle, envs[-1], main_cam_pos, camera_focus)

    if viewer is not None:
        gym.viewer_camera_look_at(
            viewer, None, gymapi.Vec3(2.2, 0, 0.5), gymapi.Vec3(0, 0, 0.5)
        )

    gym.set_light_parameters(sim, 0, gymapi.Vec3(0.3,0.3,0.3), gymapi.Vec3(1,1,1), gymapi.Vec3(-1,0,0))
    gym.set_light_parameters(sim, 1, gymapi.Vec3(0.3,0.3,0.3), gymapi.Vec3(1,1,1), gymapi.Vec3(1,0,0))

    env = envs[-1]
    ur  = ur5e_handles[-1]
    real_position = False

    # ── Warm-up ──────────────────────────────────────────────────────────────
    for t in range(2000):
        if not real_position:
            for handle, angle in zip(
                [spj, slj, ej, wj1, wj2, wj3],
                [0, -math.pi/2, 0, -math.pi/2, 0, 0],
            ):
                gym.set_dof_target_position(env, handle, angle)
            real_position = True
        if t == 999:
            for ii, element in enumerate(object_handles):
                states      = gym.get_actor_rigid_body_states(env, element, 1)
                rotation    = np.array(np.array(states[0][0][1]).item())
                translation = np.array(np.array(states[0][0][0]).item())
                object_status_list[ii][0] += translation
                r1 = R.from_quat(rotation)
                tf = fcl.Transform(r1.as_matrix(), translation)
                flex_collision_models.append(
                    [fcl.CollisionObject(object_collision_lib[ii], tf), 0]
                )
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

    # ── Home DOF settle ───────────────────────────────────────────────────────
    _HOME_DOF         = [0.7, -2.0, 2.5, -0.3, 0.7, 0.0]
    _START_SETTLE_STEPS = 30
    for _ in range(_START_SETTLE_STEPS):
        for handle, angle in zip([spj, slj, ej, wj1, wj2, wj3], _HOME_DOF):
            gym.set_dof_target_position(env, handle, angle)
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        if viewer is not None:
            gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    # ── Save object locations ─────────────────────────────────────────────────
    object_location = placed_object_locations[0] if placed_object_locations else [0.0, 0.0, float(oz)]
    with open(os.path.join(session_dir, "object_location.json"), "w") as f:
        json.dump({"xyz_m": object_location, "xyz_m_all": placed_object_locations}, f, indent=2)

    # ── Live q_start ──────────────────────────────────────────────────────────
    dof_snapshot = gym.get_actor_dof_states(env, ur, gymapi.STATE_POS)
    q_start_live = np.array(dof_snapshot["pos"][:6], dtype=np.float64)

    # ── Solve grasp q_goal_true ───────────────────────────────────────────────
    rac         = None
    q_goal_true = None

    if object_mesh:
        scene_info = [table_dims.x, table_dims.y, table_dims.z, drawer_height]
        file_path_rac = "./assets/urdf/ur5e/meshes/collision/"
        rac = RC.robot_arm_configuration(
            file_path_rac,
            np.array([ur5e_pose.p.x, ur5e_pose.p.y, ur5e_pose.p.z]),
            scene_info,
        )
        grasp_file = (
            "./assets/"
            + "/".join(object_asset_files[target_file_idx[0]].split("/")[:-1])
            + "/grasp_dict.npy"
        )
        grasp_data = np.load(grasp_file, allow_pickle=True)
        grasp_list = np.arange(len(grasp_data))
        np.random.shuffle(grasp_list)
        true_xy = np.array(object_location[:2], dtype=np.float64)

        for grasp_idx in grasp_list:
            target_grasp_pos  = grasp_data[grasp_idx]["target_pos"].copy()
            target_grasp_quat = grasp_data[grasp_idx]["target_quat"]
            target_grasp_pos[:2] += true_xy[:2]
            q_candidate = rac.grasp_verify(target_grasp_pos, target_grasp_quat)
            if q_candidate is None:
                continue
            if not rac.arm_collision_free(q_candidate, plane_obj, object_collision_models, []):
                continue
            q_goal_true = np.asarray(q_candidate, dtype=np.float64).reshape(6)
            break

    if q_goal_true is None:
        raise SystemExit(
            "[error] Could not find a collision-free grasp for the target object. "
            "Try a different --seed or check grasp_dict.npy."
        )

    with open(os.path.join(session_dir, "q_goal_true.json"), "w") as f:
        json.dump({"joint_rad": q_goal_true.tolist()}, f, indent=2)
    print(f"[true_latent] q_goal_true: {q_goal_true}", flush=True)

    # ── Compute true latent z(q_start, q_goal_true) ───────────────────────────
    latent_goal_true = _compute_z_goal(
        nt_net,
        q_start_live.reshape(1, -1),
        q_goal_true.reshape(1, -1),
        dev_nt,
    )
    with open(os.path.join(session_dir, "latent_goal_true.json"), "w") as f:
        json.dump({"latent_goal_true": latent_goal_true.reshape(-1).tolist()}, f, indent=2)

    # ── Refine / stagnation schedule (mirrors original script) ────────────────
    refine_max = args.ntfield_refine_max_steps
    if refine_max < 0:
        refine_max = 100 if args.ntfield_delta_clamp_rad > 0.0 else 0

    if args.ntfield_refine_step_size is not None:
        refine_step = float(args.ntfield_refine_step_size)
    elif args.ntfield_refine_step_size_factor is not None:
        refine_step = float(args.ntfield_step_size) * float(args.ntfield_refine_step_size_factor)
    else:
        refine_step = float(args.ntfield_step_size) * 0.5

    refine_delta_clamp_rad: Optional[float] = None
    if args.ntfield_refine_delta_clamp_rad is not None and float(args.ntfield_refine_delta_clamp_rad) > 0.0:
        refine_delta_clamp_rad = float(args.ntfield_refine_delta_clamp_rad)

    stagnate_step = (
        float(args.ntfield_stagnate_step_size)
        if args.ntfield_stagnate_step_size is not None
        else float(args.ntfield_step_size) * 0.25
    )

    # ── Gradient planner toward true latent goal ──────────────────────────────
    def ntfield_plan_gradient_with_goal_latent(
        teacher_network,
        q_start: np.ndarray,
        z_goal: np.ndarray,
        step_size: float    = 0.02,
        max_steps: int      = 200,
        tol: float          = 0.01,
        device: str         = "cuda",
        delta_clamp_rad: float       = 0.0,
        refine_max_steps: int        = 0,
        refine_step_size: float      = 0.01,
        refine_delta_clamp_rad: Optional[float] = None,
        stagnate_max_steps: int      = 0,
        stagnate_patience: int       = 30,
        stagnate_rel_eps: float      = 5e-4,
        stagnate_step_size: float    = 0.005,
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        import torch

        def _latent_dist(qn: torch.Tensor, zg: torch.Tensor) -> float:
            with torch.no_grad():
                d, _, _ = teacher_network.out_with_goal_latent(qn.detach(), zg)
                return float(d.item())

        def _grad_step(q_t, zg, stp, clamp_rad):
            q_t  = q_t.detach().requires_grad_(True)
            dist, _, coords_out = teacher_network.out_with_goal_latent(q_t, zg)
            d_pre = float(dist.item())
            if d_pre < tol:
                return q_t.detach(), d_pre, True
            grad_out   = torch.autograd.grad(dist, coords_out)[0]
            grad_start = grad_out[:, :6]
            with torch.no_grad():
                delta = -stp * grad_start
                if clamp_rad > 0.0:
                    cap_norm = float(clamp_rad) / float(NTFIELD_SCALE)
                    peak     = torch.amax(torch.abs(delta))
                    if float(peak.item()) > cap_norm + 1e-12:
                        delta = delta * (cap_norm / (peak + 1e-12))
                q_next = q_t + delta
            d_post    = _latent_dist(q_next, zg)
            converged = d_post < tol
            return q_next.detach(), d_post, converged

        q_start      = np.asarray(q_start, dtype=np.float32).reshape(-1)
        q_curr_norm  = q_start / NTFIELD_SCALE
        q_curr_t     = torch.tensor(q_curr_norm, dtype=torch.float32, device=device).unsqueeze(0)

        if isinstance(z_goal, np.ndarray):
            z_goal = torch.tensor(z_goal.reshape(1, -1), dtype=torch.float32, device=device)
        else:
            z_goal = z_goal.reshape(1, -1).to(device)

        path_norm  = [q_curr_norm.copy()]
        final_dist: Optional[float] = None
        converged  = False
        stopped    = "max_main_steps"

        main_clamp = float(delta_clamp_rad) if delta_clamp_rad > 0.0 else 0.0
        for _ in range(max_steps):
            q_curr_t, final_dist, converged = _grad_step(q_curr_t, z_goal, step_size, main_clamp)
            path_norm.append(q_curr_t.detach().cpu().numpy()[0].copy())
            if converged:
                stopped = "latent_tol_main"
                break

        refine_clamp = (
            float(refine_delta_clamp_rad)
            if refine_delta_clamp_rad is not None and refine_delta_clamp_rad > 0.0
            else 0.0
        )
        if refine_max_steps > 0 and not converged:
            for _ in range(refine_max_steps):
                q_curr_t, final_dist, converged = _grad_step(
                    q_curr_t, z_goal, refine_step_size, refine_clamp
                )
                path_norm.append(q_curr_t.detach().cpu().numpy()[0].copy())
                if converged:
                    stopped = "latent_tol_refine"
                    break
            else:
                if not converged:
                    stopped = "max_refine_steps"

        if stagnate_max_steps > 0 and not converged:
            best_d = final_dist if final_dist is not None else _latent_dist(q_curr_t, z_goal)
            stall  = 0
            for _ in range(stagnate_max_steps):
                q_curr_t, final_dist, converged = _grad_step(
                    q_curr_t, z_goal, stagnate_step_size, 0.0
                )
                path_norm.append(q_curr_t.detach().cpu().numpy()[0].copy())
                if converged:
                    stopped = "latent_tol_stagnate"
                    break
                imp = (best_d - final_dist) / max(best_d, 1e-8)
                if imp > stagnate_rel_eps:
                    best_d = min(best_d, final_dist)
                    stall  = 0
                else:
                    stall += 1
                    if stall >= stagnate_patience:
                        stopped = "latent_dist_stagnated"
                        break
            else:
                if not converged and stopped != "latent_tol_stagnate":
                    stopped = "max_stagnate_steps"

        if len(path_norm) < 2:
            path_norm.append(q_curr_t.detach().cpu().numpy()[0].copy())

        meta = {"final_latent_dist": final_dist, "stopped": stopped}
        return [p * NTFIELD_SCALE for p in path_norm], meta

    path_true, meta_true = ntfield_plan_gradient_with_goal_latent(
        nt_net,
        q_start_live.reshape(1, -1),
        latent_goal_true.reshape(1, -1),
        step_size             = args.ntfield_step_size,
        max_steps             = args.ntfield_max_steps,
        tol                   = args.ntfield_tol,
        device                = ntfield_device_str,
        delta_clamp_rad       = args.ntfield_delta_clamp_rad,
        refine_max_steps      = refine_max,
        refine_step_size      = refine_step,
        refine_delta_clamp_rad = refine_delta_clamp_rad,
        stagnate_max_steps    = int(args.ntfield_stagnate_max_steps),
        stagnate_patience     = int(args.ntfield_stagnate_patience),
        stagnate_rel_eps      = float(args.ntfield_stagnate_rel_eps),
        stagnate_step_size    = stagnate_step,
    )
    # ── Record video ──────────────────────────────────────────────────────────
    mp4_path = os.path.join(session_dir, "ntfield_trajectory_true_latent_goal.mp4")
    if path_true and len(path_true) >= 2:
        reset_arm_to_q(
            gym, sim, env, ur, spj, slj, ej, wj1, wj2, wj3, viewer, q_start_live, n_steps=200
        )
        frames: List[np.ndarray] = []
        execute_path_and_time(
            gym, sim, env, ur, spj, slj, ej, wj1, wj2, wj3, viewer,
            _path_as_6_list(path_true), "true_latent_goal",
            main_cam_handle=main_cam_handle,
            camera_props=camera_props,
            record_rgb=frames,
            planner_playback=args.planner_playback,
        )
        _save_mp4_rgb(frames, mp4_path, fps=args.video_fps)
    else:
        print("[warn] Planner returned an empty path; no video saved.")
        mp4_path = None

    # ── Joint error vs true goal ───────────────────────────────────────────────
    q_end     = np.asarray(path_true[-1], dtype=np.float64).reshape(6)
    d_raw     = q_end - q_goal_true
    d_wrapped = np.arctan2(np.sin(d_raw), np.cos(d_raw))
    joint_err = {
        "joint_err_l2_rad":             float(np.linalg.norm(d_raw)),
        "joint_err_linf_rad":           float(np.max(np.abs(d_raw))),
        "joint_err_l2_wrapped_rad":     float(np.linalg.norm(d_wrapped)),
        "joint_err_linf_wrapped_rad":   float(np.max(np.abs(d_wrapped))),
    }

    # ── Summary ───────────────────────────────────────────────────────────────
    status  = "Success" if isinstance(meta_true["stopped"], str) and meta_true["stopped"].startswith("latent_tol") else "Failure"
    summary = {
        "session_dir":              session_dir,
        "status":                   status,
        "planner_stop_reason":      meta_true["stopped"],
        "ntfield_final_latent_dist": meta_true["final_latent_dist"],
        "video":                    mp4_path,
        "q_start":                  q_start_live.tolist(),
        "q_goal_true":              q_goal_true.tolist(),
        "q_end":                    q_end.tolist(),
        "joint_error":              joint_err
    }

    with open(os.path.join(session_dir, "pipeline_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if viewer is not None:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

    import gc
    gc.collect()
    if dev_nt.type == "cuda":
        torch.cuda.synchronize()

    os.chdir(_cwd_prev)
    print(json.dumps(summary, indent=2))

    if not args.no_isaac_hard_exit:
        os._exit(0)


if __name__ == "__main__":
    main()