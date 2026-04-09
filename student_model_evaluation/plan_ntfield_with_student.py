#
# trajectory_evaluation/comparison/run_rrt_ntfield_benchmark.py
#
# Fixed scene: table (0.8, 1.0, 0.1) m, single YCB 006_mustard_bottle at user-provided world pose.
# 1) Grasp pipeline -> goal joints q_goal (same as collect_data / integrated).
# 2) RRTConnect get_path2grasp from current sim q_start; record planning time, path joint motion, execution time.
# 3) Reset arm to q_start; NTField gradient plan; same metrics.
# 4) Reset arm to q_start; Student visual latent plan; same metrics.
#
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime

import numpy as np
from scipy.spatial.transform import Rotation as R
from isaacgym import gymapi
from isaacgym import gymutil
import fcl
from PIL import Image

_PI_VLA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HANWEN_GRASPING_ROOT = os.path.join(_PI_VLA_ROOT, "hanwen_grasping")
util_dir = os.path.join(HANWEN_GRASPING_ROOT, "util")
grasp_util_dir = os.path.join(HANWEN_GRASPING_ROOT, "grasp_util")
sys.path.insert(0, HANWEN_GRASPING_ROOT)
sys.path.append(util_dir)
sys.path.append(grasp_util_dir)
sys.path.insert(0, _PI_VLA_ROOT)
sys.path.insert(0, os.path.join(_PI_VLA_ROOT, "ntrl-demo"))

import torch
import torchvision.transforms as T
import robot_arm_configuration as RC
from stl_reader import stl_reader
from obj_reader import obj_reader
from trajectory_evaluation.ntfield.eval_trajectory_ntfield import _ModelShim, load_network_and_function
from planning.gradient_planner_trajectory import SCALE as NTFIELD_SCALE
from planning.gradient_planner_trajectory import plan as ntfield_plan

# --- Fixed benchmark layout (matches user request) ---
TABLE_DIMS_X = 0.8
TABLE_DIMS_Y = 1.0
TABLE_DIMS_Z = 0.10
DRAWER_HEIGHT = 0.40
NUM_OF_OBJECTS = 1
MUSTARD_ASSET_IDX = 3
TARGET_OBJ_INDEX = [MUSTARD_ASSET_IDX]
ADD_COVER = False
max_scaling_factor = 0

sim_dt = 1.0 / 60.0
SETTLE_STEPS = 15
FINAL_HOLD_STEPS = 80
RAD_PER_SIM_STEP_HEURISTIC = 0.018

# -------------------------------------------------------------------------
# STUDENT MODEL HELPERS
# -------------------------------------------------------------------------

def preprocess_collect_data_rgb(rgb_uint8: np.ndarray, device: torch.device) -> torch.Tensor:
    if rgb_uint8.dtype != np.uint8:
        rgb_uint8 = np.clip(rgb_uint8, 0, 255).astype(np.uint8)
    tfm = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return tfm(rgb_uint8).unsqueeze(0).to(device)


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


def mppi_plan_latent_space(teacher_network, q_start, z_goal_hat, scale, steps=200, sample_num=50, horizon=5, device=None) -> np.ndarray:
    """MPPI planner evaluating paths directly against the goal latent vector."""
    q_curr = torch.tensor(q_start, dtype=torch.float32, device=device).unsqueeze(0) / scale
    dp_prior = torch.zeros((1, 6), device=device)
    path = [q_curr.clone() * scale]

    for _ in range(steps):
        q_tmp = q_curr.unsqueeze(0).repeat(sample_num, horizon, 1)

        dp = 0.015 * torch.randn((sample_num, 1, 6), device=device) + 0.015 * torch.randn((sample_num, horizon, 6), device=device)
        dp = dp + 2.0 * dp_prior
        dp_norm = torch.norm(dp, dim=2, keepdim=True)
        dp = dp / (torch.clamp(dp_norm, min=0.015) / 0.015)

        dp_cumsum = torch.cumsum(dp, dim=1)
        q_tmp = q_tmp + dp_cumsum

        q_horizon_start = q_tmp[:, 0, :]
        q_horizon_end = q_tmp[:, -1, :]
        z_target_exp = z_goal_hat.expand(sample_num, -1)

        cost_0, _, _ = teacher_network.out_with_goal_latent(q_horizon_start, z_target_exp)
        cost_1, _, _ = teacher_network.out_with_goal_latent(q_horizon_end, z_target_exp)

        total_cost = 10.0 * cost_0.squeeze(1) + cost_1.squeeze(1)

        weight = torch.softmax(-50.0 * total_cost, dim=0)
        dp_prior = (weight @ dp[:, 0, :]).unsqueeze(0)
        q_curr = q_curr + dp_prior
        path.append(q_curr.clone() * scale)

        with torch.no_grad():
            tau_curr, _, _ = teacher_network.out_with_goal_latent(q_curr, z_goal_hat)
        if tau_curr.item() < 0.05:
            break

    return np.stack([p.squeeze(0).detach().cpu().numpy() for p in path], axis=0)


# -------------------------------------------------------------------------

def _resolve_pi_vla_checkpoint(path: str) -> str:
    if not path: return path
    if os.path.isabs(path): return os.path.normpath(path)
    return os.path.normpath(os.path.join(_PI_VLA_ROOT, path))

def _path_as_6_list(path):
    if not path: return None
    return [np.asarray(p, dtype=np.float64).reshape(-1)[:6].tolist() for p in path]

def get_swept_volume_size(main_swept):
    min_x, min_y, min_z = sys.maxsize, sys.maxsize, sys.maxsize
    max_x, max_y, max_z = -sys.maxsize, -sys.maxsize, -sys.maxsize
    for tx, ty, tz in main_swept:
        min_x = min(min_x, tx); min_y = min(min_y, ty); min_z = min(min_z, tz)
        max_x = max(max_x, tx); max_y = max(max_y, ty); max_z = max(max_z, tz)
    return max_y - min_y

def joint_metrics(path_6, q_start, q_goal):
    if not path_6: return {}
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
    if not path_local or waypoint_idx <= 0: return SETTLE_STEPS
    q0 = np.asarray(path_local[waypoint_idx - 1], dtype=np.float64)
    q1 = np.asarray(path_local[waypoint_idx], dtype=np.float64)
    dq = float(np.max(np.abs(q1 - q0)))
    return max(SETTLE_STEPS, min(600, int(math.ceil(dq / RAD_PER_SIM_STEP_HEURISTIC))))

def _append_cam_rgb(gym, sim, env, main_cam_handle, camera_props, record_rgb):
    if record_rgb is None or main_cam_handle is None: return
    gym.render_all_camera_sensors(sim)
    raw_main = gym.get_camera_image(sim, env, main_cam_handle, gymapi.IMAGE_COLOR)
    rgba_main = raw_main.reshape(camera_props.height, camera_props.width, 4)
    record_rgb.append(rgba_main[..., :3].copy())

def _save_mp4_rgb(frames, out_mp4, fps=60.0):
    if not frames: return
    os.makedirs(os.path.dirname(os.path.abspath(out_mp4)), exist_ok=True)
    try:
        import imageio
        imageio.mimsave(out_mp4, frames, fps=fps)
    except Exception:
        import cv2
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        wri = cv2.VideoWriter(out_mp4, fourcc, fps, (w, h))
        for fr in frames:
            wri.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        wri.release()

def execute_path_and_time(
    gym, sim, env, ur_handle, spj, slj, ej, wj1, wj2, wj3, viewer, path_local, label,
    main_cam_handle=None, camera_props=None, record_rgb=None, planner_playback="direct",
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
            if viewer is not None: gym.draw_viewer(viewer, sim, True)
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
        if viewer is not None: gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)
        n_sub += 1
        _append_cam_rgb(gym, sim, env, main_cam_handle, camera_props, record_rgb)
    t1 = time.perf_counter()
    return {
        "label": label, "success": True, "execution_wall_s": float(t1 - t0),
        "execution_sim_s": float(n_sub * sim_dt), "physics_steps": int(n_sub),
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
        if viewer is not None: gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)


def main():
    parser = argparse.ArgumentParser(description="RRTConnect vs NTField benchmark (fixed 0.8x1.0 table, mustard).")
    parser.add_argument("--object_x", type=float, required=True, help="Mustard bottle actor position x (world m)")
    parser.add_argument("--object_y", type=float, required=True, help="Mustard bottle actor position y (world m)")
    parser.add_argument("--object_z", type=float, required=True, help="Mustard bottle actor position z (world m)")
    
    # Teacher Args
    parser.add_argument("--ntfield_checkpoint", type=str, required=True, help="Trajectory NTField Model_Epoch_*.pt")
    parser.add_argument("--ntfield_experiment_dir", type=str, default=None)
    parser.add_argument("--ntfield_device", type=str, default="cuda:0")
    parser.add_argument("--ntfield_step_size", type=float, default=0.02)
    parser.add_argument("--ntfield_max_steps", type=int, default=200)
    parser.add_argument("--ntfield_tol", type=float, default=0.01)
    parser.add_argument("--ntfield_goal_eps_rad", type=float, default=None)
    
    # Student Args
    parser.add_argument("--student_checkpoint", type=str, default=None, help="Student model .pt")
    parser.add_argument("--student_planner", type=str, default="mppi_latent", choices=["mppi_latent", "gradient_latent"])
    parser.add_argument("--student_prompt", type=str, default="006_mustard_bottle", help="Prompt condition")

    parser.add_argument("--use_viewer", action="store_true")
    parser.add_argument("--output_json", type=str, default=None, help="Write result JSON to this path")
    parser.add_argument("--record_dir", type=str, default=None)
    parser.add_argument("--video_fps", type=float, default=60.0, help="FPS for saved MP4s")
    parser.add_argument("--no_video", action="store_true", help="Skip MP4 recording")
    parser.add_argument("--planner_playback", type=str, choices=("direct", "settle"), default="direct")
    parser.add_argument("--seed", type=int, default=None)
    
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
    
    # Load Teacher
    ntfield_network, ntfield_fn = load_network_and_function(ckpt_abs, args.ntfield_experiment_dir, dev, dim=6)
    nt_model = _ModelShim(ntfield_fn)
    ntfield_network.eval() 
    ntfield_device_str = str(dev) if dev.type == "cuda" else "cpu"
    goal_eps = float(args.ntfield_goal_eps_rad) if args.ntfield_goal_eps_rad is not None else float(args.ntfield_tol * NTFIELD_SCALE)

    # Load Student Model
    student_model = None
    student_start_cond = "joints"
    if args.student_checkpoint:
        try:
            from train_goal_rep_alignment import GoalLatentPredictorWithFiLM
            student_impl = "legacy_film"
        except ImportError:
            from img2goal_config.train_goal_rep_alignment import GoalLatentPredictorImageOnly
            student_impl = "image_only"

        stu_payload = torch.load(args.student_checkpoint, map_location="cpu")
        ntfield_h = int(stu_payload["ntfield_h"])
        student_start_cond = str(stu_payload.get("start_cond", "joints"))

        if student_impl == "legacy_film":
            student_model = GoalLatentPredictorWithFiLM(ntfield_h=ntfield_h, start_cond=student_start_cond).to(dev)
        else:
            student_model = GoalLatentPredictorImageOnly(ntfield_h=ntfield_h).to(dev)
            
        student_model.load_state_dict(stu_payload["student_state_dict"], strict=True)
        student_model.eval()
        print(f"Loaded Student Model: {args.student_checkpoint}")


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
    target_file_idx = np.array(TARGET_OBJ_INDEX)
    main_cam_handle = None

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
        objs_manager = fcl.DynamicAABBTreeCollisionManager()
        objs_manager.setup()
        obstacle_objs = []
        GT_OBJ_POS_LIST = []

        object_scaling_factor = np.ones(NUM_OF_OBJECTS, dtype=np.float64)

        for k in range(NUM_OF_OBJECTS):
            object_pose = gymapi.Transform()
            tx, ty, tz = float(args.object_x), float(args.object_y), float(args.object_z)
            object_pose.p = gymapi.Vec3(tx, ty, tz)
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
            t = fcl.Transform(np.array([tx, ty, tz]))
            GT_OBJ_POS_LIST.append([tx, ty])
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

        # Top-view camera for data collection / visual prompting
        top_cam_handle = gym.create_camera_sensor(envs[-1], camera_props)
        top_cam_pos = gymapi.Vec3(table_pose.p.x, table_pose.p.y + 0.001, 2.2)
        top_cam_target = gymapi.Vec3(table_pose.p.x - 0.5, table_pose.p.y, table_pose.p.z)
        gym.set_camera_location(top_cam_handle, envs[-1], top_cam_pos, top_cam_target)
        main_cam_handle = top_cam_handle

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
    rac = RC.robot_arm_configuration(
        file_path_rac, np.array([ur5e_pose.p.x, ur5e_pose.p.y, ur5e_pose.p.z]), scene_info
    )

    grasp_file = "./assets/" + "/".join(object_asset_files[target_file_idx[0]].split("/")[:-1]) + "/grasp_dict.npy"
    grasp_data = np.load(grasp_file, allow_pickle=True)
    target_idx = 0

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
        init2grasp_collision = rac.arm_collision_free(
            init2grasp_angels_temp, plane_obj, object_collision_models, []
        )
        grasp2init_collision = rac.arm_collision_free(
            grasp2init_angels_temp, plane_obj, object_collision_models, []
        )
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
        temp_mod_bbox = rac.modify_grasp_bbox(
            init2grasp_angels_temp, target_mesh=object_mesh[target_idx], visualize=False
        )
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
        "table_dims_m": [TABLE_DIMS_X, TABLE_DIMS_Y, TABLE_DIMS_Z],
        "object_pose_world_m": [args.object_x, args.object_y, args.object_z],
        "object": "006_mustard_bottle",
        "q_start_live": q_start_live.tolist(),
        "goal_configuration_grasp_verify": grasp_target_q.tolist() if grasp_target_q is not None else None,
        "rrtconnect": {},
        "ntfield": {},
        "student": {},
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
    if args.record_dir:
        session_dir = os.path.abspath(args.record_dir)
    else:
        session_dir = os.path.join(_PI_VLA_ROOT, "output", "trajectory_evaluation", f"benchmark_{stamp}")
    os.makedirs(session_dir, exist_ok=True)
    mp4_rrt = os.path.join(session_dir, "rrt.mp4")
    mp4_nt = os.path.join(session_dir, "ntfield.mp4")
    mp4_stu = os.path.join(session_dir, "ntfield_student.mp4")
    result["video_session_dir"] = session_dir
    want_video = not args.no_video

    # ----------------------------------------------------
    # RRT Execution
    # ----------------------------------------------------
    path_rrt = _path_as_6_list(init2grasp_path)
    if grasp_target_q is None:
        grasp_target_q = np.asarray(path_rrt[-1], dtype=np.float64).reshape(6)

    result["rrtconnect"]["planning_wall_s_for_get_path2grasp_only"] = _rrt_plan_s
    result["rrtconnect"]["success"] = True
    result["rrtconnect"]["num_waypoints"] = len(path_rrt)
    result["rrtconnect"]["trajectory_waypoints_rad"] = path_rrt
    result["rrtconnect"]["motion"] = joint_metrics(path_rrt, q_start_live, grasp_target_q)

    frames_rrt = [] if want_video else None
    exec_rrt = execute_path_and_time(
        gym, sim, env, ur, spj, slj, ej, wj1, wj2, wj3, viewer, path_rrt, "rrt",
        main_cam_handle, camera_props, frames_rrt, args.planner_playback
    )
    result["rrtconnect"]["execution"] = exec_rrt
    if want_video and frames_rrt:
        _save_mp4_rgb(frames_rrt, mp4_rrt, fps=args.video_fps)
        result["rrtconnect"]["video_path"] = mp4_rrt
    elif want_video:
        result["rrtconnect"]["video_path"] = None

    # ----------------------------------------------------
    # Teacher NTField Planning
    # ----------------------------------------------------
    reset_arm_to_q(gym, sim, env, ur, spj, slj, ej, wj1, wj2, wj3, viewer, q_start_live, n_steps=200)

    t_nt0 = time.perf_counter()
    path_nt_raw = ntfield_plan(nt_model, q_start_live, grasp_target_q, step_size=args.ntfield_step_size, max_steps=args.ntfield_max_steps, tol=args.ntfield_tol, device=ntfield_device_str)
    nt_planning_wall_s = float(time.perf_counter() - t_nt0)

    nt_err_pen = float(np.linalg.norm(np.asarray(path_nt_raw[-2], dtype=np.float64).reshape(6) - grasp_target_q)) if path_nt_raw and len(path_nt_raw) >= 2 else None
    nt_has_path = path_nt_raw is not None and len(path_nt_raw) >= 2

    result["ntfield"] = {
        "planning_wall_s": nt_planning_wall_s,
        "success": nt_has_path,
        "converged_within_tol": nt_err_pen is not None and nt_err_pen < goal_eps,
        "goal_eps_rad": goal_eps,
        "penultimate_goal_error_rad": nt_err_pen,
        "num_planner_steps_including_appended_goal": len(path_nt_raw) if path_nt_raw else 0,
    }

    if nt_has_path:
        path_nt = _path_as_6_list(path_nt_raw)
        result["ntfield"]["trajectory_waypoints_rad"] = path_nt
        result["ntfield"]["motion"] = joint_metrics(path_nt, q_start_live, grasp_target_q)
        frames_nt = [] if want_video else None
        exec_nt = execute_path_and_time(
            gym, sim, env, ur, spj, slj, ej, wj1, wj2, wj3, viewer, path_nt, "ntfield",
            main_cam_handle, camera_props, frames_nt, args.planner_playback
        )
        result["ntfield"]["execution"] = exec_nt
        if want_video and frames_nt:
            _save_mp4_rgb(frames_nt, mp4_nt, fps=args.video_fps)
            result["ntfield"]["video_path"] = mp4_nt
    else:
        result["ntfield"]["execution"] = None


    # ----------------------------------------------------
    # Student NTField Planning
    # ----------------------------------------------------
    if student_model is not None:
        reset_arm_to_q(gym, sim, env, ur, spj, slj, ej, wj1, wj2, wj3, viewer, q_start_live, n_steps=200)
        t_stu0 = time.perf_counter()

        # Capture Image
        gym.render_all_camera_sensors(sim)
        raw_top = gym.get_camera_image(sim, env, main_cam_handle, gymapi.IMAGE_COLOR)
        rgb_top = raw_top.reshape(camera_props.height, camera_props.width, 4)[..., :3]
        img_t = preprocess_collect_data_rgb(rgb_top, dev)

        qs_t = torch.tensor(q_start_live, dtype=torch.float32, device=dev).unsqueeze(0)
        with torch.no_grad():
            try:
                # Legacy FiLM signature
                z_goal_hat = student_model(img_t, [args.student_prompt], qs_t)
            except TypeError:
                # Image-only signature
                z_goal_hat = student_model(img_t)

        # Plan using latent goal directly
        if args.student_planner == "gradient_latent":
            path_stu_raw = ntfield_latent_plan_gradient(
                teacher_network=ntfield_network,
                q_start=q_start_live,
                z_goal_hat=z_goal_hat,
                step_size=args.ntfield_step_size,
                max_steps=args.ntfield_max_steps,
                tol=args.ntfield_tol,
                device=dev
            )
        else:
            path_stu_raw = mppi_plan_latent_space(
                teacher_network=ntfield_network,
                q_start=q_start_live,
                z_goal_hat=z_goal_hat,
                scale=NTFIELD_SCALE,
                device=dev
            )

        stu_planning_wall_s = float(time.perf_counter() - t_stu0)
        stu_has_path = path_stu_raw is not None and len(path_stu_raw) >= 2

        result["student"] = {
            "planning_wall_s": stu_planning_wall_s,
            "success": stu_has_path,
            "num_planner_steps": len(path_stu_raw) if path_stu_raw else 0,
        }

        if stu_has_path:
            path_stu = _path_as_6_list(path_stu_raw)
            result["student"]["trajectory_waypoints_rad"] = path_stu
            result["student"]["reached_destination_rad"] = path_stu[-1]
            
            frames_stu = [] if want_video else None
            exec_stu = execute_path_and_time(
                gym, sim, env, ur, spj, slj, ej, wj1, wj2, wj3, viewer, path_stu, "student",
                main_cam_handle, camera_props, frames_stu, args.planner_playback
            )
            result["student"]["execution"] = exec_stu
            if want_video and frames_stu:
                _save_mp4_rgb(frames_stu, mp4_stu, fps=args.video_fps)
                result["student"]["video_path"] = mp4_stu
        else:
            result["student"]["execution"] = None


    # Finish
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