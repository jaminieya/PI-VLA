#!/usr/bin/env python3
"""
End-to-end PI-VLA integration (Isaac Gym + obj-loc + TracIK + NTField).

Run from PI-VLA root:
  python final_integrate/run_integrated_pipeline_latent.py \
    --ntfield_checkpoint ntrl-demo/Experiments/UR5_trajectory_no_wall_accuracy_check/trajectory_03_25_20_28/Model_Epoch_05000_ValLoss_7.820605e-01.pt \
    --objloc_checkpoint final_integrate/best_obj_loc_model.pth \
    --latent_checkpoint final_integrate/best_z_goal_model_original.pth

Outputs under output/final_integrate/<timestamp>/ (see --output_dir).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_PI_VLA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HANWEN_GRASPING_ROOT = os.path.join(_PI_VLA_ROOT, "hanwen_grasping")
_COLLECT_DATA_DIR = os.path.join(HANWEN_GRASPING_ROOT, "collect_data")
_UTIL_DIR = os.path.join(_COLLECT_DATA_DIR, "util")
_GRASP_UTIL_DIR = os.path.join(_COLLECT_DATA_DIR, "grasp_util")
_NTRL_DEMO = os.path.join(_PI_VLA_ROOT, "ntrl-demo")
_IMG2OBJ = os.path.join(_PI_VLA_ROOT, "img2objloc_model")

NORMALIZE_COORDS = True
# Same as planning.gradient_planner_trajectory.SCALE; defined here (not imported) because
# that module imports torch, which must load after isaacgym in main().
SCALE = float(np.pi / 0.5)

for _p in (HANWEN_GRASPING_ROOT, _UTIL_DIR, _GRASP_UTIL_DIR, _PI_VLA_ROOT, _NTRL_DEMO):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _resolve_under_root(path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(_PI_VLA_ROOT, path))


def _build_coords_batch(
    q_start: torch.Tensor, q_goal: torch.Tensor, use_scale: bool
) -> torch.Tensor:
    """q_start, q_goal: (B, 6) radians -> (B, 12) teacher input."""
    import torch

    q_start = torch.as_tensor(q_start, dtype=torch.float32)
    q_goal = torch.as_tensor(q_goal, dtype=torch.float32)
    if use_scale:
        q_start = q_start / SCALE
        q_goal = q_goal / SCALE
    return torch.cat([q_start, q_goal], dim=1)


def _compute_z_goal(
    teacher: torch.nn.Module,
    qs: torch.Tensor,
    qg: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """
    qs, qg: (B, D) joint configs (numpy or tensor) — teacher runs on ``device``.
    Returns z_goal: (H,) float32 on CPU (batch B must be 1 for this pipeline).

    Torch is imported inside the body so this module can load before isaacgym
    (Isaac Gym requires ``import isaacgym`` before ``import torch``).
    """
    import torch

    with torch.no_grad():
        coords = _build_coords_batch(qs, qg, NORMALIZE_COORDS).to(device)
        _, zg = teacher.encode_pair_latents(coords)
    return zg.detach().cpu().float().numpy().reshape(-1)

def _get_model(output_dim: int = 3):
    import torch.nn as nn
    import torchvision.models as models

    # No ImageNet weights — checkpoint supplies all weights (older torchvision
    # has no ResNet18_Weights; use pretrained=False).
    try:
        model = models.resnet18(weights=None)
    except TypeError:
        model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, output_dim)
    return model

def _get_latent_model(output_dim: int = 256):
    import torch.nn as nn
    import torchvision.models as models

    # No ImageNet weights — checkpoint supplies all weights (older torchvision
    # has no ResNet18_Weights; use pretrained=False).
    try:
        model = models.resnet18(weights=None)
    except TypeError:
        model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, output_dim)
    return model
    
def _process_img(img: np.ndarray, img_size: int):
    """Resize + to tensor. Matches what we did in training."""
    import torch

    if img.dtype!= np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]

    h, w = img.shape[:2]
    if h!= img_size or w!= img_size:
        ys = np.linspace(0, h - 1, img_size).astype(int)
        xs = np.linspace(0, w - 1, img_size).astype(int)
        img = img[np.ix_(ys, xs)]

    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

def _infer_latent_on_image(
    image: np.ndarray,
    checkpoint_path: str,
    device: str,
) -> Tuple[float, float, float]:
    """
    Run the latent_model.py-style model (get_model / output_dim=128) on a raw HWC
    uint8 RGB numpy array.  Returns (pred_latent).
    """
    import torch
    from torchvision import transforms

    ckpt_path = os.path.abspath(checkpoint_path)
    dev = torch.device(
        "cuda" if torch.cuda.is_available() and device == "auto" else device
    )

    model = _get_latent_model(output_dim=256).to(dev)
    model.load_state_dict(torch.load(ckpt_path, map_location=dev))
    model.eval()

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    x = _process_img(image, 224).unsqueeze(0)
    x = normalize(x).to(dev)

    with torch.no_grad():
        pred = model(x).squeeze(0).cpu().numpy()  # shape (256,)

    return pred

def _infer_objloc_on_image(
    image: np.ndarray,
    checkpoint_path: str,
    device: str,
) -> Tuple[float, float, float]:
    """
    Run the view_data.py-style model (get_model / output_dim=3) on a raw HWC
    uint8 RGB numpy array.  Returns (pred_x, pred_y, pred_z).
    """
    import torch
    from torchvision import transforms

    if _IMG2OBJ not in sys.path:
        sys.path.insert(0, _IMG2OBJ)


    ckpt_path = os.path.abspath(checkpoint_path)
    dev = torch.device(
        "cuda" if torch.cuda.is_available() and device == "auto" else device
    )

    model = _get_model(output_dim=3).to(dev)
    model.load_state_dict(torch.load(ckpt_path, map_location=dev))
    model.eval()

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    x = _process_img(image, 224).unsqueeze(0)
    x = normalize(x).to(dev)

    with torch.no_grad():
        pred = model(x).squeeze(0).cpu().numpy()  # shape (3,)

    return float(pred[0]), float(pred[1]), float(pred[2])


def find_grasp_q_goal(
    rac: Any,
    RC_mod: Any,
    scene_info: List[float],
    grasp_data: np.ndarray,
    grasp_list: np.ndarray,
    obj_world_xy: np.ndarray,
    target_idx: int,
    object_mesh: List[Any],
    object_collision_models: List[Any],
    plane_obj: Any,
    get_swept_volume_size_fn: Any,
) -> Tuple[Optional[np.ndarray], Optional[List[np.ndarray]], float]:
    """
    Same selection logic as run_rrt_ntfield_benchmark: TracIK (grasp_verify) + RRT path + swept volume.
    ``obj_world_xy`` is (2,) added to grasp template XY (world frame).
    """
    GT_OBJ_POS_LIST = [obj_world_xy.tolist()]
    num_grasp = 0
    swept_size = sys.maxsize
    init2grasp_path: Optional[List[np.ndarray]] = None
    grasp_target_q: Optional[np.ndarray] = None
    rrt_plan_s = 0.0
    consecutive_path_failures = 0
    MAX_FAIL = 15

    for grasp_idx in grasp_list:
        if consecutive_path_failures >= MAX_FAIL:
            break
        target_grasp_pos = grasp_data[grasp_idx]["target_pos"].copy()
        target_grasp_quat = grasp_data[grasp_idx]["target_quat"]
        target_grasp_pos[:2] = target_grasp_pos[:2] + GT_OBJ_POS_LIST[target_idx][:2]
        init2grasp_angels_temp = rac.grasp_verify(target_grasp_pos, target_grasp_quat)
        grasp2init_angels_temp = rac.grasp_verify(
            target_grasp_pos + [0, 0, 0.01], target_grasp_quat
        )
        if init2grasp_angels_temp is None or grasp2init_angels_temp is None:
            continue
        init2grasp_collision = rac.arm_collision_free(
            init2grasp_angels_temp, plane_obj, object_collision_models, []
        )
        grasp2init_collision = rac.arm_collision_free(
            grasp2init_angels_temp, plane_obj, object_collision_models, []
        )
        if not init2grasp_collision or not grasp2init_collision:
            continue

        t_rrt0 = time.perf_counter()
        init2grasp_path_temp = RC_mod.get_path2grasp(
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
        temp_mod_bbox = rac.modify_grasp_bbox(
            init2grasp_angels_temp, target_mesh=object_mesh[target_idx], visualize=False
        )
        grasp2init_path_temp = RC_mod.get_path2start(
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
            grasp2init_path_temp,
            w_target=temp_mod_bbox,
            frame_rate=60,
            scene_info=scene_info,
            animation=False,
            static_vi=False,
        )
        num_grasp += 1
        swept_center_temp, swept_verts_temp = rac.get_swept_center(
            swept_verts1_temp + swept_verts2_temp, scene_info, 0.6
        )
        temp_swept_size = get_swept_volume_size_fn(swept_verts_temp)
        if temp_swept_size < swept_size:
            swept_size = temp_swept_size
            init2grasp_path = init2grasp_path_temp
            grasp_target_q = np.asarray(init2grasp_angels_temp, dtype=np.float64).reshape(6)
            rrt_plan_s = rrt_planning_wall_s
        if num_grasp == 1:
            break

    return grasp_target_q, init2grasp_path, rrt_plan_s


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
        get_swept_volume_size,
        reset_arm_to_q,
        _save_mp4_rgb,
        _path_as_6_list,
        sim_dt,
    )
    from trajectory_evaluation.ntfield.eval_trajectory_ntfield import _ModelShim, load_network_and_function
    from planning.gradient_planner_trajectory import SCALE as NTFIELD_SCALE
    from planning.gradient_planner_trajectory import plan as ntfield_plan
    from planning.gradient_planner_trajectory import plan_with_goal_latent as ntfield_plan_with_goal_latent

    parser = argparse.ArgumentParser(description="PI-VLA final integration pipeline")
    parser.add_argument("--ntfield_checkpoint", type=str, required=True)
    parser.add_argument("--ntfield_experiment_dir", type=str, default=None)
    parser.add_argument("--objloc_checkpoint", type=str, required=True,
                        help="Path to best_obj_loc_model.pth (get_model / output_dim=3)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Session dir (default: output/final_integrate/TIMESTAMP)")
    parser.add_argument("--object_z", type=float, default=0.18,
                        help="Actor Z (m), same order as benchmark")
    parser.add_argument("--ox_min", type=float, default=0.42)
    parser.add_argument("--ox_max", type=float, default=0.98)
    parser.add_argument("--oy_min", type=float, default=-0.38)
    parser.add_argument("--oy_max", type=float, default=0.38)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use_viewer", action="store_true")
    parser.add_argument("--ntfield_device", type=str, default="cuda:0")
    parser.add_argument("--objloc_device", type=str, default="auto")
    parser.add_argument("--latent_checkpoint", type=str, required=True,
                        help="Path to image→latent goal checkpoint (ResNet18 fc, output_dim=256)")
    parser.add_argument("--latent_device", type=str, default="auto",
                        help="Device for latent model (e.g. cuda:0, cpu, auto)")
    parser.add_argument("--ntfield_step_size", type=float, default=0.02)
    parser.add_argument("--ntfield_max_steps", type=int, default=200)
    parser.add_argument("--ntfield_tol", type=float, default=0.01)
    parser.add_argument("--ntfield_goal_eps_rad", type=float, default=None)
    parser.add_argument("--video_fps", type=float, default=60.0)
    parser.add_argument(
        "--planner_playback",
        type=str,
        choices=("direct", "settle"),
        default="direct",
    )
    args, argv_remainder = parser.parse_known_args()

    # ------------------------------------------------------------------
    # Removed: --sam_checkpoint, --sam_device  (SAM no longer used)
    # ------------------------------------------------------------------

    argv_gym = list(argv_remainder)
    if not args.use_viewer and "--headless" not in argv_gym:
        argv_gym.append("--headless")
    sys.argv = [sys.argv[0]] + argv_gym

    if args.seed is not None:
        np.random.seed(args.seed)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        session_dir = os.path.abspath(args.output_dir)
    else:
        session_dir = os.path.join(_PI_VLA_ROOT, "output", "final_integrate", stamp)
    os.makedirs(session_dir, exist_ok=True)
    # sam_work directory no longer needed

    ox = float(np.random.uniform(args.ox_min, args.ox_max))
    oy = float(np.random.uniform(args.oy_min, args.oy_max))
    oz = float(args.object_z)

    ckpt_abs = _resolve_under_root(args.ntfield_checkpoint)
    if not os.path.isfile(ckpt_abs):
        raise SystemExit(f"NTField checkpoint not found: {ckpt_abs}")

    dev_nt = torch.device(
        "cpu"
        if args.ntfield_device == "cpu" or not torch.cuda.is_available()
        else args.ntfield_device
    )
    nt_net, ntfield_fn = load_network_and_function(
        ckpt_abs, args.ntfield_experiment_dir, dev_nt, dim=6
    )
    nt_model = _ModelShim(ntfield_fn)
    ntfield_device_str = str(dev_nt) if dev_nt.type == "cuda" else "cpu"
    goal_eps = (
        float(args.ntfield_goal_eps_rad)
        if args.ntfield_goal_eps_rad is not None
        else float(args.ntfield_tol * NTFIELD_SCALE)
    )

    _cwd_prev = os.getcwd()
    os.chdir(HANWEN_GRASPING_ROOT)

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

    sim = gym.create_sim(gym_args.compute_device_id, gym_args.graphics_device_id, gym_args.physics_engine, sim_params)
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

    envs: List[Any] = []
    ur5e_handles: List[Any] = []
    object_handles: List[Any] = []
    object_status_list: List[Any] = []
    object_reader_tracker: List[Any] = []
    object_mesh: List[Any] = []
    flex_collision_models: List[Any] = []
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
        ej = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "elbow_joint")
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

    # ADD THESE — match collect_data lighting exactly
    gym.set_light_parameters(sim, 0, gymapi.Vec3(0.3, 0.3, 0.3), gymapi.Vec3(1.0, 1.0, 1.0), gymapi.Vec3(-1.0, 0.0, 0.0))
    gym.set_light_parameters(sim, 1, gymapi.Vec3(0.3, 0.3, 0.3), gymapi.Vec3(1.0, 1.0, 1.0), gymapi.Vec3(1.0, 0.0, 0.0))
    
    env = envs[-1]
    ur = ur5e_handles[-1]
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

    assert top_cam_handle is not None
    gym.render_all_camera_sensors(sim)
    raw_top = gym.get_camera_image(sim, env, top_cam_handle, gymapi.IMAGE_COLOR)
    rgba_top = raw_top.reshape(camera_props.height, camera_props.width, 4)
    rgb_top = rgba_top[..., :3].copy()  # HWC uint8, matches view_data.py expectations

    top_view_path = os.path.join(session_dir, "top_view.png")
    cv2.imwrite(top_view_path, cv2.cvtColor(rgb_top, cv2.COLOR_RGB2BGR))

    # ------------------------------------------------------------------
    # Object localisation: get_model (output_dim=3) — same as view_data.py
    # Replaces the previous SAM + separate objloc_xy pipeline entirely.
    # pred_z is taken directly from the model (no z_fixed fallback needed).
    # ------------------------------------------------------------------
    pred_x, pred_y, pred_z = _infer_objloc_on_image(
        rgb_top,
        _resolve_under_root(args.objloc_checkpoint),
        args.objloc_device,
    )

    object_location_original = [float(ox), float(oy), float(oz)]
    object_location_predicted = [float(pred_x), float(pred_y), float(pred_z)]

    with open(os.path.join(session_dir, "object_location_original.json"), "w", encoding="utf-8") as f:
        json.dump({"xyz_m": object_location_original}, f, indent=2)
    with open(os.path.join(session_dir, "object_location_predicted.json"), "w", encoding="utf-8") as f:
        json.dump({"xyz_m": object_location_predicted}, f, indent=2)

    scene_info = [table_dims.x, table_dims.y, table_dims.z, drawer_height]
    file_path_rac = "./assets/urdf/ur5e/meshes/collision/"
    rac = RC.robot_arm_configuration(
        file_path_rac, np.array([ur5e_pose.p.x, ur5e_pose.p.y, ur5e_pose.p.z]), scene_info
    )

    grasp_file = "./assets/" + "/".join(object_asset_files[target_file_idx[0]].split("/")[:-1]) + "/grasp_dict.npy"
    grasp_data = np.load(grasp_file, allow_pickle=True)
    target_idx = 0
    grasp_list = np.arange(len(grasp_data))
    np.random.shuffle(grasp_list)

    true_xy = np.array([ox, oy], dtype=np.float64)
    pred_xy = np.array([pred_x, pred_y], dtype=np.float64)

    q_goal_true, _, _ = find_grasp_q_goal(
        rac, RC, scene_info, grasp_data, grasp_list,
        true_xy, target_idx, object_mesh, object_collision_models,
        plane_obj, get_swept_volume_size,
    )
    q_goal_pred, _, _ = find_grasp_q_goal(
        rac, RC, scene_info, grasp_data, grasp_list,
        pred_xy, target_idx, object_mesh, object_collision_models,
        plane_obj, get_swept_volume_size,
    )

    with open(os.path.join(session_dir, "q_goal_original.json"), "w", encoding="utf-8") as f:
        json.dump({"joint_rad": None if q_goal_true is None else q_goal_true.tolist()}, f, indent=2)
    with open(os.path.join(session_dir, "q_goal_predicted.json"), "w", encoding="utf-8") as f:
        json.dump({"joint_rad": None if q_goal_pred is None else q_goal_pred.tolist()}, f, indent=2)

    dof_snapshot = gym.get_actor_dof_states(env, ur, gymapi.STATE_POS)
    q_start_live = np.array(dof_snapshot["pos"][:6], dtype=np.float64)

    mp4_pred = os.path.join(session_dir, "ntfield_trajectory_predicted_goal.mp4")
    mp4_true = os.path.join(session_dir, "ntfield_trajectory_original_goal.mp4")

    summary: Dict[str, Any] = {
        "session_dir": session_dir,
        "object_pose_world_m": object_location_original,
        "object_location_predicted_m": object_location_predicted,
        "q_start_live": q_start_live.tolist(),
        "q_goal_original_found": q_goal_true is not None,
        "q_goal_predicted_found": q_goal_pred is not None,
        "videos": {},
    }

    def _run_ntfield_video(q_goal: Optional[np.ndarray], out_mp4: str, label: str) -> None:
        if q_goal is None:
            summary["videos"][label] = None
            return
        reset_arm_to_q(gym, sim, env, ur, spj, slj, ej, wj1, wj2, wj3, viewer, q_start_live, n_steps=200)
        path_nt_raw = ntfield_plan(
            nt_model,
            q_start_live,
            q_goal,
            step_size=args.ntfield_step_size,
            max_steps=args.ntfield_max_steps,
            tol=args.ntfield_tol,
            device=ntfield_device_str,
        )
        if not path_nt_raw or len(path_nt_raw) < 2:
            summary["videos"][label] = None
            return
        path_nt = _path_as_6_list(path_nt_raw)
        frames_nt: List[np.ndarray] = []
        execute_path_and_time(
            gym, sim, env, ur, spj, slj, ej, wj1, wj2, wj3, viewer,
            path_nt, label,
            main_cam_handle=main_cam_handle,
            camera_props=camera_props,
            record_rgb=frames_nt,
            planner_playback=args.planner_playback,
        )
        _save_mp4_rgb(frames_nt, out_mp4, fps=args.video_fps)
        summary["videos"][label] = out_mp4

    _run_ntfield_video(q_goal_pred, mp4_pred, "predicted_goal")
    _run_ntfield_video(q_goal_true, mp4_true, "original_goal")

    with open(os.path.join(session_dir, "pipeline_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    
    # ------------------------------------------------------------------
    # Latent goal: get_model (output_dim=256) — same as view_data.py
    # pred_latent is taken directly from the model.
    # Compute the true latent goal from the true q_start and q_goal with the teacher model.
    # Run the gradient planner trajectory with the predicted latent goal.
    # Run the gradient planner trajectory with the true latent goal.
    # Save the videos.
    # Save the summary.
    # ------------------------------------------------------------------


    def ntfield_latent_plan_gradient(teacher_network, q_start, z_goal_hat, step_size=0.02, max_steps=200, tol=0.01, device="cuda"):
        """Gradient descent planner minimizing distance to a goal latent vector directly."""
        q_curr_norm = np.asarray(q_start, dtype=np.float32) / NTFIELD_SCALE
        q_curr_t = torch.tensor(q_curr_norm, dtype=torch.float32, device=device).unsqueeze(0)
        q_curr_t.requires_grad_(True)

        path_norm = [q_curr_norm.copy()]

        for step in range(max_steps):
            dist, _, coords_out = teacher_network.out_with_goal_latent(q_curr_t, z_goal_hat)

            if dist.item() < tol:
                break

            # Get gradient of distance with respect to coords
            grad_out = torch.autograd.grad(dist, coords_out)[0]
            # We only care about the start configuration's gradient (first 6 dims)
            grad_start = grad_out[:, :6]

            with torch.no_grad():
                # Step in negative gradient direction to MINIMIZE distance
                q_curr_t = q_curr_t - step_size * grad_start

            q_curr_t = q_curr_t.detach().requires_grad_(True)
            path_norm.append(q_curr_t.detach().cpu().numpy()[0].copy())

        path_rad = [p * NTFIELD_SCALE for p in path_norm]
        return path_rad

    def _run_ntfield_video_with_goal_latent(z_goal: Optional[np.ndarray], out_mp4: str, label: str) -> None:
        if z_goal is None:
            summary["videos"][label] = None
            return
        reset_arm_to_q(gym, sim, env, ur, spj, slj, ej, wj1, wj2, wj3, viewer, q_start_live, n_steps=200)
        path_nt_raw = ntfield_latent_plan_gradient(
            nt_net,
            q_start_live.reshape(1, -1),
            z_goal.reshape(1, -1)
        )
        if not path_nt_raw or len(path_nt_raw) < 2:
            summary["videos"][label] = None
            return
        path_nt = _path_as_6_list(path_nt_raw)
        frames_nt: List[np.ndarray] = []
        execute_path_and_time(
            gym, sim, env, ur, spj, slj, ej, wj1, wj2, wj3, viewer,
            path_nt, label,
            main_cam_handle=main_cam_handle,
            camera_props=camera_props,
            record_rgb=frames_nt,
            planner_playback=args.planner_playback,
        )
        _save_mp4_rgb(frames_nt, out_mp4, fps=args.video_fps)
        summary["videos"][label] = out_mp4

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

    mp4_pred_latent = os.path.join(session_dir, "ntfield_trajectory_predicted_goal_latent.mp4")
    mp4_true_latent = os.path.join(session_dir, "ntfield_trajectory_original_goal_latent.mp4")
    
    _run_ntfield_video_with_goal_latent(latent_goal_pred, mp4_pred_latent, "predicted_latent_goal")
    _run_ntfield_video_with_goal_latent(latent_goal_true, mp4_true_latent, "original_latent_goal")

    with open(os.path.join(session_dir, "latent_goal_pred.json"), "w", encoding="utf-8") as f:
        json.dump({"latent_goal": latent_goal_pred.tolist()}, f, indent=2)
    with open(os.path.join(session_dir, "latent_goal_true.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"latent_goal": None if latent_goal_true is None else latent_goal_true.tolist()},
            f,
            indent=2,
        )

    gym.destroy_sim(sim)
    if viewer is not None:
        gym.destroy_viewer(viewer)
    os.chdir(_cwd_prev)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()