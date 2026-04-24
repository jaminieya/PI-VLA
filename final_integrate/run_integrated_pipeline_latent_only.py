#!/usr/bin/env python3
"""
End-to-end PI-VLA integration (Isaac Gym + NTField + latent goal only).

Run from PI-VLA root:
  python final_integrate/run_integrated_pipeline_latent_only.py \
    --ntfield_checkpoint ntrl-demo/Experiments/UR5_trajectory_no_wall_accuracy_check/trajectory_03_25_20_28/Model_Epoch_05000_ValLoss_7.820605e-01.pt \
    --latent_checkpoint final_integrate/best_z_goal_model_wonorm_mse_cos.pth

Outputs under output/final_integrate/<timestamp>/ (see --output_dir).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

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


def _process_img(img: np.ndarray, img_size: int):
    """Resize + to tensor."""
    import torch

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]

    h, w = img.shape[:2]
    if h != img_size or w != img_size:
        ys = np.linspace(0, h - 1, img_size).astype(int)
        xs = np.linspace(0, w - 1, img_size).astype(int)
        img = img[np.ix_(ys, xs)]

    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


# ---------------------------------------------------------------------------
# Latent model
# ---------------------------------------------------------------------------

def _get_latent_model(output_dim: int = 256):
    """Matches StudentModelWonorm from train_config_wonorm.py."""
    import torch.nn as nn
    import torchvision.models as models

    class StudentHead(nn.Module):
        def __init__(self, in_features, output_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, output_dim),
            )

        def forward(self, x):
            return self.net(x)

    class StudentModel(nn.Module):
        def __init__(self, output_dim):
            super().__init__()
            try:
                backbone = models.resnet18(weights=None)
            except TypeError:
                backbone = models.resnet18(pretrained=False)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.backbone = backbone
            self.head = StudentHead(in_features, output_dim)

        def forward(self, x):
            return self.head(self.backbone(x))

    return StudentModel(output_dim)


def _infer_latent_on_image(image: np.ndarray, checkpoint_path: str, device: str) -> np.ndarray:
    import torch
    from torchvision import transforms

    dev = torch.device(
        "cuda" if torch.cuda.is_available() and device == "auto" else device
    )
    ckpt = torch.load(os.path.abspath(checkpoint_path), map_location=dev)
    z_dim = ckpt.get("z_dim", 256)

    model = _get_latent_model(output_dim=z_dim).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    x = _process_img(image, 224).unsqueeze(0)
    x = normalize(x).to(dev)
    with torch.no_grad():
        pred = model(x).squeeze(0).cpu().numpy()
    return pred


def main() -> None:
    from scipy.spatial.transform import Rotation as R
    from isaacgym import gymapi
    from isaacgym import gymutil

    import fcl
    import torch
    import cv2
    import robot_arm_configuration as RC
    from stl_reader import stl_reader
    from obj_reader import obj_reader

    from trajectory_evaluation.comparison.run_rrt_ntfield_benchmark import (
        TABLE_DIMS_X,
        TABLE_DIMS_Y,
        TABLE_DIMS_Z,
        DRAWER_HEIGHT,
        NUM_OF_OBJECTS,
        TARGET_OBJ_INDEX,
        execute_path_and_time,
        reset_arm_to_q,
        _save_mp4_rgb,
        _path_as_6_list,
        sim_dt,
    )
    from trajectory_evaluation.ntfield.eval_trajectory_ntfield import load_network_and_function
    from planning.gradient_planner_trajectory import SCALE as NTFIELD_SCALE

    parser = argparse.ArgumentParser(description="PI-VLA integration — latent goal only")
    parser.add_argument("--ntfield_checkpoint", type=str, required=True)
    parser.add_argument("--ntfield_experiment_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Session dir (default: output/final_integrate/TIMESTAMP)")
    parser.add_argument("--object_z", type=float, default=0.18)
    parser.add_argument("--ox_min", type=float, default=0.42)
    parser.add_argument("--ox_max", type=float, default=0.98)
    parser.add_argument("--oy_min", type=float, default=-0.38)
    parser.add_argument("--oy_max", type=float, default=0.38)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use_viewer", action="store_true")
    parser.add_argument("--ntfield_device", type=str, default="cuda:0")
    parser.add_argument("--latent_checkpoint", type=str, required=True,
                        help="Path to image→latent goal checkpoint (ResNet18, output_dim=256)")
    parser.add_argument("--latent_device", type=str, default="auto")
    parser.add_argument("--ntfield_step_size", type=float, default=0.02)
    parser.add_argument("--ntfield_max_steps", type=int, default=200)
    parser.add_argument("--ntfield_tol", type=float, default=0.01)
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
    session_dir = (
        os.path.abspath(args.output_dir)
        if args.output_dir
        else os.path.join(_PI_VLA_ROOT, "output", "final_integrate", stamp)
    )
    os.makedirs(session_dir, exist_ok=True)

    ox = float(np.random.uniform(args.ox_min, args.ox_max))
    oy = float(np.random.uniform(args.oy_min, args.oy_max))
    oz = float(args.object_z)

    # ── Load NTField ─────────────────────────────────────────────────────────
    ckpt_abs = _resolve_under_root(args.ntfield_checkpoint)
    if not os.path.isfile(ckpt_abs):
        raise SystemExit(f"NTField checkpoint not found: {ckpt_abs}")

    dev_nt = torch.device(
        "cpu"
        if args.ntfield_device == "cpu" or not torch.cuda.is_available()
        else args.ntfield_device
    )
    nt_net, _ = load_network_and_function(
        ckpt_abs, args.ntfield_experiment_dir, dev_nt, dim=6
    )
    ntfield_device_str = str(dev_nt) if dev_nt.type == "cuda" else "cpu"

    _cwd_prev = os.getcwd()
    os.chdir(HANWEN_GRASPING_ROOT)

    # ── Isaac Gym setup ──────────────────────────────────────────────────────
    gym = gymapi.acquire_gym()
    gym_args = gymutil.parse_arguments(description="final_integrate", headless=True, custom_parameters=[])
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
        gym_args.compute_device_id, gym_args.graphics_device_id,
        gym_args.physics_engine, sim_params,
    )
    if sim is None:
        raise SystemExit("Failed to create sim")

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
    object_asset_files: List[str] = []
    object_collision_files: List[str] = []
    object_offset: List[List[float]] = []
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
    gym.create_box(sim, upper_cover_dims.x, upper_cover_dims.y, upper_cover_dims.z, asset_options)

    asset_options.fix_base_link = False
    object_assets = [gym.load_asset(sim, asset_root, ob, asset_options) for ob in object_asset_files]

    ur5e_pose = gymapi.Transform()
    ur5e_pose.p = gymapi.Vec3(0, 0, 0)
    ur5e_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0.5 * math.pi)
    table_pose = gymapi.Transform()
    table_pose.p = gymapi.Vec3(table_dims.x * 0.5 + 0.3, 0.0, table_dims.z * 0.5)
    camera_focus = gymapi.Vec3(0, 0, 0)
    camera_props = gymapi.CameraProperties()
    camera_props.horizontal_fov = 70.25
    camera_props.width = 1280
    camera_props.height = 720

    col_table = fcl.Box(table_dims.x, table_dims.y, table_dims.z)
    trans_table = fcl.Transform(np.array([table_dims.x * 0.5 + 0.3, 0.0, table_dims.z * 0.5]))
    table_obj = fcl.CollisionObject(col_table, trans_table)
    object_collision_models = [table_obj]

    envs: List[Any] = []
    ur5e_handles: List[Any] = []
    object_handles: List[Any] = []
    object_status_list: List[Any] = []
    object_reader_tracker: List[Any] = []
    object_collision_lib: List[Any] = []
    spj = slj = ej = wj1 = wj2 = wj3 = None
    target_file_idx = np.array(TARGET_OBJ_INDEX)
    main_cam_handle = None
    top_cam_handle = None

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
        objs_manager = fcl.DynamicAABBTreeCollisionManager()
        objs_manager.setup()
        obstacle_objs: List[Any] = []

        object_scaling_factor = np.ones(NUM_OF_OBJECTS, dtype=np.float64)

        for k in range(NUM_OF_OBJECTS):
            object_pose = gymapi.Transform()
            object_pose.p = gymapi.Vec3(ox, oy, oz)
            file_path = object_collision_files[target_file_idx[k]]
            collision_mesh = obj_reader(asset_root + file_path)
            collision_mesh.set_scale(object_scaling_factor[k])
            collision_mesh.add_offset(object_offset[target_file_idx[k]])
            verts, tris = collision_mesh.get_bounding_box_mesh()
            temp_center = collision_mesh.get_center()
            temp_bounding_box = collision_mesh.get_bounding_box()
            m = fcl.BVHModel()
            m.beginModel(len(verts), len(tris))
            m.addSubModel(verts, tris)
            m.endModel()
            t = fcl.Transform(np.array([ox, oy, oz]))
            object_handles.append(
                gym.create_actor(
                    envs[-1],
                    object_assets[target_file_idx[k]],
                    object_pose,
                    "object" + str(k) + str(i),
                    0,
                    2 ** (k + 1),
                    k + 1,
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
        main_cam_pos = gymapi.Vec3(3, 0, 0.3)
        gym.set_camera_location(main_cam_handle, envs[-1], main_cam_pos, camera_focus)

        top_cam_handle = gym.create_camera_sensor(envs[-1], camera_props)
        top_cam_pos = gymapi.Vec3(table_pose.p.x, table_pose.p.y + 0.001, 2.2)
        top_cam_target = gymapi.Vec3(table_pose.p.x - 0.5, table_pose.p.y, table_pose.p.z)
        gym.set_camera_location(top_cam_handle, envs[-1], top_cam_pos, top_cam_target)

    if viewer is not None:
        cam_pos = gymapi.Vec3(2.2, 0, 0.5)
        cam_target = gymapi.Vec3(0, 0, 0.5)
        gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    gym.set_light_parameters(sim, 0, gymapi.Vec3(0.3, 0.3, 0.3), gymapi.Vec3(1.0, 1.0, 1.0), gymapi.Vec3(-1.0, 0.0, 0.0))
    gym.set_light_parameters(sim, 1, gymapi.Vec3(0.3, 0.3, 0.3), gymapi.Vec3(1.0, 1.0, 1.0), gymapi.Vec3(1.0, 0.0, 0.0))

    env = envs[-1]
    ur = ur5e_handles[-1]
    real_position = False

    # ── Warm-up simulation ───────────────────────────────────────────────────
    for t in range(2000):
        if not real_position:
            gym.set_dof_target_position(env, spj, 0)
            gym.set_dof_target_position(env, slj, -math.pi / 2)
            gym.set_dof_target_position(env, ej, 0)
            gym.set_dof_target_position(env, wj1, -math.pi / 2)
            gym.set_dof_target_position(env, wj2, 0)
            gym.set_dof_target_position(env, wj3, 0)
            real_position = True
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        if viewer is not None:
            gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    # ── Capture top-view image ───────────────────────────────────────────────
    assert top_cam_handle is not None
    gym.render_all_camera_sensors(sim)
    raw_top = gym.get_camera_image(sim, env, top_cam_handle, gymapi.IMAGE_COLOR)
    rgba_top = raw_top.reshape(camera_props.height, camera_props.width, 4)
    rgb_top = rgba_top[..., :3].copy()

    top_view_path = os.path.join(session_dir, "top_view.png")
    cv2.imwrite(top_view_path, cv2.cvtColor(rgb_top, cv2.COLOR_RGB2BGR))

    object_location = [float(ox), float(oy), float(oz)]
    with open(os.path.join(session_dir, "object_location.json"), "w", encoding="utf-8") as f:
        json.dump({"xyz_m": object_location}, f, indent=2)

    # ── Get live q_start ─────────────────────────────────────────────────────
    dof_snapshot = gym.get_actor_dof_states(env, ur, gymapi.STATE_POS)
    q_start_live = np.array(dof_snapshot["pos"][:6], dtype=np.float64)

    # ── Infer latent goal from image ─────────────────────────────────────────
    latent_goal_pred = _infer_latent_on_image(
        rgb_top,
        _resolve_under_root(args.latent_checkpoint),
        args.latent_device,
    )

    summary: Dict[str, Any] = {
        "session_dir": session_dir,
        "object_pose_world_m": object_location,
        "q_start_live": q_start_live.tolist(),
        "videos": {},
    }

    # ── Gradient planner using predicted latent goal ─────────────────────────
    def ntfield_plan_gradient_with_goal_latent(
        teacher_network, q_start: np.ndarray, z_goal_hat: np.ndarray,
        step_size: float = 0.02, max_steps: int = 200,
        tol: float = 0.01, device: str = "cuda",
    ):
        import torch

        q_start = np.asarray(q_start, dtype=np.float32).reshape(-1)
        q_curr_norm = q_start / NTFIELD_SCALE
        q_curr_t = torch.tensor(
            q_curr_norm, dtype=torch.float32, device=device
        ).unsqueeze(0)  # (1, 6)

        if isinstance(z_goal_hat, np.ndarray):
            z_goal_hat = torch.tensor(
                z_goal_hat.reshape(1, -1), dtype=torch.float32, device=device
            )
        else:
            z_goal_hat = z_goal_hat.reshape(1, -1).to(device)

        path_norm = [q_curr_norm.copy()]

        for _ in range(max_steps):
            q_curr_t = q_curr_t.detach().requires_grad_(True)
            dist, _, coords_out = teacher_network.out_with_goal_latent(q_curr_t, z_goal_hat)

            if dist.item() < tol:
                break

            grad_out = torch.autograd.grad(dist, coords_out)[0]
            grad_start = grad_out[:, :6]

            with torch.no_grad():
                q_curr_t = q_curr_t - step_size * grad_start

            path_norm.append(q_curr_t.detach().cpu().numpy()[0].copy())

        return [p * NTFIELD_SCALE for p in path_norm]

    # ── Success-checking helpers ─────────────────────────────────────────────
    def _get_end_effector_position(
        gym_inst, sim_inst, env_inst, ur_handle
    ) -> np.ndarray:
        """Return world-space XYZ of the end-effector (last rigid body = tool0 link)."""
        rigid_states = gym_inst.get_actor_rigid_body_states(
            env_inst, ur_handle, gymapi.STATE_POS
        )
        ee_state = rigid_states[-1]
        return np.array(
            [ee_state["pose"]["p"]["x"],
             ee_state["pose"]["p"]["y"],
             ee_state["pose"]["p"]["z"]],
            dtype=np.float64,
        )

    SUCCESS_THRESHOLD_M = 0.10   # metres — tune to your task

    # ── Execute and record trajectory ────────────────────────────────────────
    reset_arm_to_q(gym, sim, env, ur, spj, slj, ej, wj1, wj2, wj3, viewer, q_start_live, n_steps=200)

    path_nt_raw = ntfield_plan_gradient_with_goal_latent(
        nt_net,
        q_start_live.reshape(1, -1),
        latent_goal_pred.reshape(1, -1),
        step_size=args.ntfield_step_size,
        max_steps=args.ntfield_max_steps,
        tol=args.ntfield_tol,
        device=ntfield_device_str,
    )

    mp4_path = os.path.join(session_dir, "ntfield_trajectory_predicted_latent_goal.mp4")
    if path_nt_raw and len(path_nt_raw) >= 2:
        path_nt = _path_as_6_list(path_nt_raw)
        frames_nt: List[np.ndarray] = []
        execute_path_and_time(
            gym, sim, env, ur, spj, slj, ej, wj1, wj2, wj3, viewer,
            path_nt, "predicted_latent_goal",
            main_cam_handle=main_cam_handle,
            camera_props=camera_props,
            record_rgb=frames_nt,
            planner_playback=args.planner_playback,
        )
        _save_mp4_rgb(frames_nt, mp4_path, fps=args.video_fps)
        summary["videos"]["predicted_latent_goal"] = mp4_path

        # ── Distance / success check ─────────────────────────────────────────
        ee_pos = _get_end_effector_position(gym, sim, env, ur)
        target_pos = np.array([ox, oy, oz], dtype=np.float64)
        ee_to_target_dist = float(np.linalg.norm(ee_pos - target_pos))
        success = ee_to_target_dist <= SUCCESS_THRESHOLD_M

        summary["success_check"] = {
            "ee_position_m":       ee_pos.tolist(),
            "target_position_m":   target_pos.tolist(),
            "ee_to_target_dist_m": ee_to_target_dist,
            "threshold_m":         SUCCESS_THRESHOLD_M,
            "success":             success,
        }
        print(
            f"[success_check]  EE={ee_pos.round(4).tolist()}  "
            f"target={target_pos.round(4).tolist()}  "
            f"dist={ee_to_target_dist:.4f} m  "
            f"{'✓ SUCCESS' if success else '✗ FAILURE'}  (thr={SUCCESS_THRESHOLD_M} m)"
        )
    else:
        print("[warn] NTField planner returned an empty path.")
        summary["videos"]["predicted_latent_goal"] = None
        summary["success_check"] = {
            "ee_position_m":       None,
            "target_position_m":   [ox, oy, oz],
            "ee_to_target_dist_m": None,
            "threshold_m":         SUCCESS_THRESHOLD_M,
            "success":             False,
        }

    # ── Save outputs ─────────────────────────────────────────────────────────
    with open(os.path.join(session_dir, "latent_goal_pred.json"), "w", encoding="utf-8") as f:
        json.dump({"latent_goal": latent_goal_pred.tolist()}, f, indent=2)
    with open(os.path.join(session_dir, "pipeline_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    gym.destroy_sim(sim)
    if viewer is not None:
        gym.destroy_viewer(viewer)
    os.chdir(_cwd_prev)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()