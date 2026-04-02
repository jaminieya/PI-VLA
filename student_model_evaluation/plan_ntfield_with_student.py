#
# File:        run_retrieval.py
# Brief:       Comparison test on active sensing methods and rearrangement methods
# Author:      Junyoung Kim -- kim3722@purdue.edu, Hanwen Ren -- ren221@purdue.edu
# Date:        2024-05-04
# Last Modified: 2025-05-17
#
# Lives under PI-VLA/student_model_evaluation/; Isaac assets and local imports are in hanwen_grasping/.
#

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Outputs (recordings, debug frames) — cwd is hanwen_grasping/ for assets.
_STUDENT_EVAL_RECORDINGS = os.path.join(_SCRIPT_DIR, "recordings")

file_dir = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "hanwen_grasping"))
os.chdir(file_dir)
if file_dir not in sys.path:
    sys.path.insert(0, file_dir)

from scipy.spatial.transform import Rotation as R
import math
import re
import time
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from PIL import Image
import numpy as np
from trac_ik_python.trac_ik import IK
import open3d as o3d
import fcl
import cv2
import copy

import torch
import torchvision.transforms as T
import json

import robot_arm_configuration as RC

#import MCTS_algo_ICRA as mct
#from rearrangement_planning_util_ICRA import write_result

util_dir = os.path.join(file_dir, './util')
grasp_util_dir = os.path.join(file_dir, './grasp_util')
pi_vla_root = os.path.dirname(file_dir)
ntrl_demo_path = os.path.abspath(os.path.join(pi_vla_root, 'ntrl-demo'))
if os.path.isdir(ntrl_demo_path) and ntrl_demo_path not in sys.path:
    sys.path.insert(0, ntrl_demo_path)
sys.path.append(util_dir)
sys.path.append(grasp_util_dir)

try:
    import ompl.base as ob
    import ompl.util as ou
    import ompl.geometric as og
except ImportError:
    from os.path import abspath, dirname, join
    sys.path.insert(
        0, join(dirname(dirname(dirname(abspath(__file__)))), 'py-bindings'))
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og

from stl_reader import stl_reader
from obj_reader import obj_reader

#define parameters
#*************************************************************************************************#
num_of_envs = 1
row_num_of_envs = int(math.sqrt(num_of_envs))
SCALE = float(np.pi / 0.5)

choose = np.random.randint(2)
if choose == 0:
    max_drawer_height = 0.40
    min_drawer_height = 0.40
    MIN_NUM_OBSTACLES = 5
    MAX_NUM_OBSTACLES = 8
    table_dims = gymapi.Vec3(np.random.uniform(0.5, 0.7), np.random.uniform(0.8, 1.0), 0.10)
else:
    max_drawer_height = 0.55
    min_drawer_height = 0.55
    MIN_NUM_OBSTACLES = 7
    MAX_NUM_OBSTACLES = 11
    table_dims = gymapi.Vec3(np.random.uniform(0.7, 0.9), np.random.uniform(1.0, 1.2), 0.10)

piece_width = 0.03
max_scaling_factor = 0
fall_height = table_dims.z
ADD_COVER = False

TARGET_OBJ_INDEX = [1, 3, 5]
MIN_RADIUS = 0.03471716871486391

NUM_OF_OBJECTS = 1
#*************************************************************************************************#

#helper functions
#*************************************************************************************************#
def plan_to_student_latent(
    teacher_network: torch.nn.Module,
    q_start_rad: np.ndarray,
    z_goal_hat: torch.Tensor,
    *,
    steps: int = 200,
    lr: float = 0.05,
    device: torch.device,
    use_scale: bool,
    metric: str = "cosine",
    clip_rad: float = float(np.pi),
) -> tuple:
    """Optimize q_goal to match student latent using teacher's z_goal embedding."""
    q_start = np.asarray(q_start_rad, dtype=np.float32).reshape(6)
    q_start_t = torch.tensor(q_start, dtype=torch.float32, device=device).unsqueeze(0)
    if use_scale:
        q_start_n = q_start_t / SCALE
    else:
        q_start_n = q_start_t

    # Add small noise to avoid getting stuck at a zero-gradient saddle near q_start.
    noise = torch.randn_like(q_start_n) * 0.05
    q_goal_n = (q_start_n.clone().detach() + noise).requires_grad_(True)
    opt = torch.optim.Adam([q_goal_n], lr=lr)

    path = [q_start.copy()]
    loss_history = []

    for _ in range(steps):
        opt.zero_grad()
        coords = torch.cat([q_start_n, q_goal_n], dim=1)
        _, z_goal_cur = teacher_network.encode_pair_latents(coords)

        if metric == "mse":
            loss = torch.nn.functional.mse_loss(z_goal_cur, z_goal_hat)
        else:
            zc = torch.nn.functional.normalize(z_goal_cur, dim=1)
            zh = torch.nn.functional.normalize(z_goal_hat, dim=1)
            loss = (1.0 - (zc * zh).sum(dim=1)).mean()

        loss.backward()
        opt.step()

        with torch.no_grad():
            lim = clip_rad / SCALE if use_scale else clip_rad
            q_goal_n.clamp_(-lim, lim)
            qg = q_goal_n.detach().clone()
            if use_scale:
                qg = qg * SCALE
            path.append(qg[0].cpu().numpy().astype(np.float32))
            loss_history.append(float(loss.item()))

    with torch.no_grad():
        coords = torch.cat([q_start_n, q_goal_n], dim=1)
        _, z_goal_cur = teacher_network.encode_pair_latents(coords)
        zc = torch.nn.functional.normalize(z_goal_cur, dim=1)
        zh = torch.nn.functional.normalize(z_goal_hat, dim=1)
        cos_sim = float((zc * zh).sum(dim=1).mean().item())
        mse = float(torch.nn.functional.mse_loss(z_goal_cur, z_goal_hat).item())

    stats = {
        "steps": steps,
        "planner_lr": lr,
        "metric": metric,
        "final_cosine_similarity": cos_sim,
        "final_mse": mse,
        "loss_history": loss_history,
    }
    return np.stack(path, axis=0), stats


def preprocess_collect_data_rgb(rgb_uint8: np.ndarray, device: str) -> torch.Tensor:
    """
    Align exactly with collect_data.py + training dataset pipeline:
      collect_data: rgb = rgba[..., :3].copy() saved as uint8
      training: ToPILImage -> Resize((224,224)) -> ToTensor -> Normalize(ImageNet)
    """
    if rgb_uint8.dtype != np.uint8:
        rgb_uint8 = np.clip(rgb_uint8, 0, 255).astype(np.uint8)
    tfm = T.Compose(
        [
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return tfm(rgb_uint8).unsqueeze(0).to(device)

def _build_record_frames(path_rad: np.ndarray, loss_history: list, fps: int) -> list:
    """Render a simple trajectory/loss visualization into RGB frames."""
    import matplotlib.pyplot as plt
    n_steps = path_rad.shape[0]
    n_joints = path_rad.shape[1]
    xs = np.arange(n_steps)
    frames = []

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), dpi=120)
    fig.tight_layout(pad=2.0)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]

    for t in range(n_steps):
        ax1.clear()
        for j in range(n_joints):
            ax1.plot(xs[: t + 1], path_rad[: t + 1, j], color=colors[j % len(colors)], linewidth=1.8, label=f"q{j+1}")
        ax1.set_title("Student-conditioned NTField planned trajectory")
        ax1.set_ylabel("Joint angle (rad)")
        ax1.set_xlim(0, max(1, n_steps - 1))
        ax1.grid(True, alpha=0.3)
        if t == 0:
            ax1.legend(loc="upper right", ncol=3, fontsize=8)

        ax2.clear()
        if loss_history:
            ax2.plot(np.arange(1, len(loss_history) + 1), loss_history, color="black", linewidth=1.8)
            cur = min(t, len(loss_history) - 1)
            ax2.scatter([cur + 1], [loss_history[cur]], color="red", s=30)
            ax2.set_xlim(1, max(2, len(loss_history)))
        ax2.set_title("Latent matching loss during optimization")
        ax2.set_xlabel("Optimization step")
        ax2.set_ylabel("Loss")
        ax2.grid(True, alpha=0.3)

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        rgb = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
        frames.append(rgb.copy())
    plt.close(fig)
    return frames

def global_coord_converter(coord1, coord2, coord3, offset1, offset2, offset3):
    return (coord1 - offset1, coord3 - offset3, -coord2 + offset2)

def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0.w, quaternion0.x, quaternion0.y, quaternion0.z
    w1, x1, y1, z1 = quaternion1.w, quaternion1.x, quaternion1.y, quaternion1.z
    return gymapi.Quat(x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                       -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                       x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0, 
                       -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0)

def interpolate_path_ntfield(path, steps_between=4):
    if path is None or len(path) < 2:
        return path
    interpolated = []
    for i in range(len(path) - 1):
        start = np.array(path[i], dtype=np.float64)
        end = np.array(path[i + 1], dtype=np.float64)
        for k in range(steps_between + 1):
            t = k / (steps_between + 1)
            pt = start + t * (end - start)
            interpolated.append(pt.tolist())
    interpolated.append(np.array(path[-1], dtype=np.float64).tolist())
    return interpolated

def _student_eval_output_path(path: str) -> str:
    """Relative paths -> student_model_evaluation/recordings ; absolute paths unchanged."""
    if not path:
        return path
    if os.path.isabs(path):
        return path
    os.makedirs(_STUDENT_EVAL_RECORDINGS, exist_ok=True)
    return os.path.join(_STUDENT_EVAL_RECORDINGS, path)


def parse_q_ntfield(s):
    s = s.strip().replace("pi", "math.pi")
    parts = [x.strip() for x in s.split(",")]
    if len(parts) != 6:
        raise ValueError(f"Expected 6 comma-separated values, got {len(parts)}")
    return [float(eval(p, {"math": math})) for p in parts]

def resolve_collected_h5_path(h5_path, script_dir, repo_root):
    if not h5_path:
        return None
    if os.path.isfile(h5_path):
        return os.path.abspath(h5_path)
    cand = os.path.join(script_dir, h5_path)
    if os.path.isfile(cand):
        return os.path.abspath(cand)
    cand2 = os.path.join(repo_root, h5_path)
    if os.path.isfile(cand2):
        return os.path.abspath(cand2)
    cand3 = os.path.join(repo_root, "collected_data", os.path.basename(h5_path))
    if os.path.isfile(cand3):
        return os.path.abspath(cand3)
    return os.path.abspath(h5_path)

def read_h5_object_world_xyz(h5_path):
    import h5py
    if not h5_path or not os.path.isfile(h5_path):
        return None
    with h5py.File(h5_path, "r") as f:
        if "object_actor_world" in f:
            v = np.array(f["object_actor_world"][:], dtype=np.float64).reshape(-1)
            if v.size >= 3:
                return v[:3].copy()
            return None
        if "object_location" in f:
            v = np.array(f["object_location"][:], dtype=np.float64).reshape(-1)
            if v.size < 3:
                return None
            if float(v[0]) < 0.28:
                return None
            return v[:3].copy()
    return None


def resolve_planning_torch_device(args) -> torch.device:
    """
    Isaac PhysX uses --compute_device_id (often 0) and leaves little free VRAM on that GPU.
    When multiple CUDA devices exist, default PyTorch to a *different* index so teacher/student
    load and run without contending with the sim on the same device.
    """
    override = getattr(args, "torch_device", None)
    if override is not None and str(override).strip():
        return torch.device(str(override).strip())
    if not torch.cuda.is_available():
        return torch.device("cpu")
    phys = int(getattr(args, "compute_device_id", 0))
    n = torch.cuda.device_count()
    if n <= 1:
        return torch.device("cuda")
    alt = (phys + 1) % n
    return torch.device(f"cuda:{alt}")


#*************************************************************************************************#

if __name__ == '__main__':
    gym = gymapi.acquire_gym()

    args = gymutil.parse_arguments(
        description="ur5e example",
        custom_parameters=[
            {'name': '--env_id', 'type': int, 'help': 'env_id', 'default': 0},
            {'name': '--ntfield', 'action': 'store_true', 'help': 'Use NTField to plan and animate robot path'},
            {'name': '--checkpoint', 'type': str, 'default': None, 'help': 'NTField checkpoint path'},
            {'name': '--student', 'type': str, 'default': None, 'help': 'Student model checkpoint path (.pt)'},
            {'name': '--q_start', 'type': str, 'default': None, 'help': 'Start joint config'},
            {'name': '--q_goal', 'type': str, 'default': None, 'help': 'Goal joint config'},
            {'name': '--h5_path', 'type': str, 'default': None, 'help': 'H5 path for configs and prompt'},
            {'name': '--record', 'action': 'store_true', 'help': 'Record video of the simulation'},
            {
                'name': '--record_output',
                'type': str,
                'default': 'ntfield_record.mp4',
                'help': 'Output video path (relative -> student_model_evaluation/recordings/)',
            },
            {'name': '--no_walls', 'action': 'store_true', 'help': 'Remove side walls'},
            {'name': '--no_h5_spawn_object', 'action': 'store_true', 'help': 'Do not place object at HDF5 location'},
            {'name': '--headless', 'action': 'store_true', 'help': 'No interactive viewer'},
            {
                'name': '--torch_device',
                'type': str,
                'default': '',
                'help': 'PyTorch device for NTField teacher/student (e.g. cuda:1, cpu). Empty = auto: use another GPU than --compute_device_id when possible.',
            },
        ],
    )
    args._session_eval_dir = None
    env_id = int(args.env_id)
    ntfield_h5_object_xyz = None
    if getattr(args, 'ntfield', False) and args.h5_path:
        args.h5_path = resolve_collected_h5_path(args.h5_path, file_dir, pi_vla_root)
        ntfield_h5_object_xyz = read_h5_object_world_xyz(args.h5_path)

    sim_params = gymapi.SimParams()
    sim_params.substeps = 2
    sim_params.dt = 1.0 / 60.0
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0, 0, -9.8)

    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
    sim_params.use_gpu_pipeline = False

    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
    if sim is None:
        quit()

    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)

    asset_root = "./assets/"
    ur5e_asset_file = "urdf/ur5e/ur5e_mimic_real_gripper_test.urdf"

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

    import h5_episode_viz as h5viz
    _nt_h5_resolved = None
    _nt_h5_ol = None
    _nt_h5_prompt = ""
    _nt_h5_obj_idx = None
    if getattr(args, 'ntfield', False) and args.h5_path and not getattr(args, 'no_h5_spawn_object', False):
        _nt_h5_resolved = h5viz.resolve_h5_path(args.h5_path, pi_vla_root, file_dir)
        if _nt_h5_resolved:
            _nt_h5_ol, _nt_h5_prompt = h5viz.read_h5_object_and_prompt(_nt_h5_resolved)
            _nt_h5_obj_idx = h5viz.infer_ycb_index_from_prompt(_nt_h5_prompt, object_asset_files)

    # Calculate Inverse Kinematics string
    with open("./assets/urdf/ur5e/ur5e_mimic_real_gripper_test.urdf") as f:
        urdf_str = f.read()

    viewer = None
    if not getattr(args, 'headless', False):
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())

    spacing = 2
    env_lower = gymapi.Vec3(-spacing, -spacing, 0)
    env_upper = gymapi.Vec3(spacing, spacing, 0)

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
    asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
    asset_options.use_mesh_materials = True

    ur5e_asset = gym.load_asset(sim, asset_root, ur5e_asset_file, asset_options)
    table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

    drawer_height = np.random.random()*(max_drawer_height - min_drawer_height) + min_drawer_height
    side_cover_dims = gymapi.Vec3(table_dims.x, piece_width, drawer_height)
    left_cover_asset = gym.create_box(sim, side_cover_dims.x, side_cover_dims.y, side_cover_dims.z, asset_options)
    right_cover_asset = gym.create_box(sim, side_cover_dims.x, side_cover_dims.y, side_cover_dims.z, asset_options)

    ur5e_pose = gymapi.Transform()
    ur5e_pose.p = gymapi.Vec3(0, 0, 0)
    ur5e_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0.5*math.pi)

    table_pose = gymapi.Transform()
    table_pose.p = gymapi.Vec3(table_dims.x*0.5 + 0.3, 0.0, table_dims.z*0.5)

    left_cover_pose = gymapi.Transform()
    left_cover_pose.p = gymapi.Vec3(table_pose.p.x, table_dims.y*0.5 - 0.015, table_dims.z + side_cover_dims.z/2.0)
    right_cover_pose = gymapi.Transform()
    right_cover_pose.p = gymapi.Vec3(table_pose.p.x, -table_dims.y*0.5 + 0.015, table_dims.z + side_cover_dims.z/2.0)

    camera_props = gymapi.CameraProperties()
    camera_props.horizontal_fov = 70.25
    camera_props.width = 1280
    camera_props.height = 720

    envs = []
    ur5e_handles = []
    body_cam_handles = []
    top_cam_handles = []

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
        if not getattr(args, 'no_walls', False):
            gym.create_actor(envs[-1], left_cover_asset, left_cover_pose, "left_cover" + str(i), 0, 1)
            gym.create_actor(envs[-1], right_cover_asset, right_cover_pose, "right_cover" + str(i), 0, 1)

        # Place objects
        h5_use_fixed = (_nt_h5_obj_idx is not None and _nt_h5_ol is not None and _nt_h5_ol.size >= 3)
        target_file_idx = [_nt_h5_obj_idx] if h5_use_fixed else np.random.choice(TARGET_OBJ_INDEX, NUM_OF_OBJECTS)
        object_scaling_factor = np.ones(NUM_OF_OBJECTS, dtype=np.float64) if h5_use_fixed else np.random.randint(0, max_scaling_factor+1, size = NUM_OF_OBJECTS)/10.0 + 1.0

        for k in range(NUM_OF_OBJECTS):
            object_pose = gymapi.Transform()
            if h5_use_fixed:
                object_pose.p = gymapi.Vec3(float(_nt_h5_ol[0]), float(_nt_h5_ol[1]), float(_nt_h5_ol[2]))
            else:
                object_pose.p = gymapi.Vec3(np.random.uniform(0.35, table_dims.x + 0.2), np.random.uniform(-table_dims.y/2 + 0.1, table_dims.y/2 - 0.2), table_dims.z + 0.08)

            fk = target_file_idx[k]
            gym.create_actor(envs[-1], gym.load_asset(sim, asset_root, object_asset_files[fk], asset_options), object_pose, "object" + str(k) + str(i), 0, 2**(k+1), k+1)

        # Top-down camera
        top_cam_handles.append(gym.create_camera_sensor(envs[-1], camera_props))
        top_cam_pos = gymapi.Vec3(table_pose.p.x, table_pose.p.y + 0.001, 2)
        top_cam_target = gymapi.Vec3(table_pose.p.x - 0.5, table_pose.p.y, table_pose.p.z)
        gym.set_camera_location(top_cam_handles[-1], envs[-1], top_cam_pos, top_cam_target)

    if viewer is not None:
        gym.viewer_camera_look_at(viewer, None, top_cam_pos, top_cam_target)

    # Initialization & settling
    real_position = False
    for t in range(50):
        if not real_position:
            gym.set_dof_target_position(envs[-1], spj, 0)
            gym.set_dof_target_position(envs[-1], slj, -math.pi/2)
            gym.set_dof_target_position(envs[-1], ej,  0)
            gym.set_dof_target_position(envs[-1], wj1, -math.pi/2)
            gym.set_dof_target_position(envs[-1], wj2, 0)
            gym.set_dof_target_position(envs[-1], wj3, 0)
            real_position = True

        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        if viewer is not None:
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)

    robot_path = None
    plot_frames = []
    
    if getattr(args, 'ntfield', False):
        import h5py
        from models.metric_arm import model_test_metric as md
        from planning import plan as gradient_plan
        sys.path.append(ntrl_demo_path)

        device = resolve_planning_torch_device(args)
        phys_id = int(getattr(args, "compute_device_id", 0))
        print(f"[NTField] PyTorch device: {device} (Isaac PhysX compute_device_id={phys_id})")

        # Parse start config
        with h5py.File(args.h5_path, "r") as f:
            q_start = np.array(f["joint_configs"][0], dtype=np.float64)
            print(q_start)

        # Load Teacher Model
        model_path = os.path.dirname(os.path.abspath(args.checkpoint))
        data_path = os.path.join(ntrl_demo_path, "datasets", "arm", "UR5_trajectory")
        teacher = md.Model(model_path, data_path, dim=6, source=[0.0] * 6, device=str(device))
        teacher.load(os.path.abspath(args.checkpoint))
        teacher.network.eval()

        if getattr(args, 'student', None):
            print(f"--- Student Model Mode Activated ---")
            
            # 1. Capture Top-View Image for Student Prediction
            gym.render_all_camera_sensors(sim)
            raw = gym.get_camera_image(sim, envs[-1], top_cam_handles[-1], gymapi.IMAGE_COLOR)
            rgba = raw.reshape(camera_props.height, camera_props.width, 4)
            rgb_top = rgba[..., :3].copy()
            # Debug: raw RGB exactly as collect_data would save to H5 "images".
            Image.fromarray(rgb_top).save(_student_eval_output_path("debug_full_camera_view.png"))
            img_t = preprocess_collect_data_rgb(rgb_top, str(device))

            # Save raw RGB exactly from simulator (same format as collected_data images)
            Image.fromarray(rgb_top).save(_student_eval_output_path("debug_raw_rgb_input.png"))
            # Save the exact model input after preprocess (denormalized for visualization)
            img_vis = img_t[0].detach().cpu().permute(1, 2, 0).numpy()
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_vis = (img_vis * std + mean)
            img_vis = np.clip(img_vis * 255.0, 0, 255).astype(np.uint8)
            Image.fromarray(img_vis).save(_student_eval_output_path("debug_model_input_224.png"))


            # 2. Load Student Architecture
            from train_goal_rep_alignment import GoalLatentPredictorWithFiLM
            student_payload = torch.load(args.student, map_location="cpu")
            ntfield_h = int(student_payload["ntfield_h"])
            use_scale = bool(student_payload.get("normalize_coords", False))

            student = GoalLatentPredictorWithFiLM(ntfield_h=ntfield_h).to(device)
            student.load_state_dict(student_payload["student_state_dict"], strict=True)
            student.eval()

            # 3. Predict Goal Latent Space
            qs_t = torch.tensor(q_start, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                z_goal_hat = student(img_t, [_nt_h5_prompt], qs_t)

            # 3.5 Compare student latent against ground-truth goal latent from dataset.
            with h5py.File(args.h5_path, "r") as f:
                if "final_joint_config" in f:
                    q_goal_gt = np.array(f["final_joint_config"][:6], dtype=np.float32)
                else:
                    q_goal_gt = np.array(f["joint_configs"][-1, :6], dtype=np.float32)

            q_goal_gt_t = torch.tensor(q_goal_gt, dtype=torch.float32, device=device).unsqueeze(0)
            if use_scale:
                qs_n = qs_t / SCALE
                qg_n = q_goal_gt_t / SCALE
            else:
                qs_n = qs_t
                qg_n = q_goal_gt_t
            coords_gt = torch.cat([qs_n, qg_n], dim=1)
            with torch.no_grad():
                _, z_goal_gt = teacher.network.encode_pair_latents(coords_gt)

            z_hat_n = torch.nn.functional.normalize(z_goal_hat, dim=1)
            z_gt_n = torch.nn.functional.normalize(z_goal_gt, dim=1)
            cos_sim_gt = float((z_hat_n * z_gt_n).sum(dim=1).mean().item())
            cos_loss_gt = 1.0 - cos_sim_gt
            mse_gt = float(torch.nn.functional.mse_loss(z_goal_hat, z_goal_gt).item())
            print(f"[GT compare] cosine similarity: {cos_sim_gt:.6f}")
            print(f"[GT compare] cosine loss (1-cos): {cos_loss_gt:.6f}")
            print(f"[GT compare] mse: {mse_gt:.6f}")

            # 4. Use Adam search to find q_goal_hat matching student latent target.
            opt_trace_rad, stats = plan_to_student_latent(
                teacher.network, q_start, z_goal_hat,
                steps=200, lr=0.05, device=device, use_scale=use_scale, metric="cosine"
            )

            # Final optimization point is the predicted goal coordinate.
            q_goal_hat = opt_trace_rad[-1]

            # 5. Plan physical path using native NTField planner to q_goal_hat.
            path = gradient_plan(teacher, q_start, q_goal_hat, step_size=0.02, max_steps=200, tol=0.01, device=device)
            robot_path = interpolate_path_ntfield(path, steps_between=4)
            print("num waypoints:", len(robot_path))
            print("start:", np.array(robot_path[0]))
            print("end:", np.array(robot_path[-1]))
            print("delta L2:", float(np.linalg.norm(np.array(robot_path[-1]) - np.array(robot_path[0]))))
            
            if getattr(args, 'record', False):
                plot_frames = _build_record_frames(opt_trace_rad, stats.get("loss_history", []), fps=30)

        else:
            # Native NTField Plan
            with h5py.File(args.h5_path, "r") as f:
                q_goal = np.array(f["final_joint_config"][:], dtype=np.float64) if "final_joint_config" in f else np.array(f["joint_configs"][-1], dtype=np.float64)
            path = gradient_plan(teacher, q_start, q_goal, step_size=0.02, max_steps=200, tol=0.01, device=device)
            robot_path = interpolate_path_ntfield(path, steps_between=4)


    if robot_path is None or len(robot_path) == 0:
        print("Error: No path to animate.")
        sys.exit(1)

    # Animate Path Loop
    path_id_box = [0]
    record_frames = [] if getattr(args, 'record', False) else None
    
    def _animate_robot_path_step():
        path_id = min(path_id_box[0], len(robot_path) - 1)
        dof_result = robot_path[path_id]

        gym.set_dof_target_position(envs[-1], spj, dof_result[0])
        gym.set_dof_target_position(envs[-1], slj, dof_result[1])
        gym.set_dof_target_position(envs[-1], ej,  dof_result[2])
        gym.set_dof_target_position(envs[-1], wj1, dof_result[3])
        gym.set_dof_target_position(envs[-1], wj2, dof_result[4])
        gym.set_dof_target_position(envs[-1], wj3, dof_result[5])

        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)

        if viewer is not None:
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)

        if record_frames is not None:
            gym.render_all_camera_sensors(sim)
            raw = gym.get_camera_image(sim, envs[-1], top_cam_handles[-1], gymapi.IMAGE_COLOR)
            rgba = raw.reshape(camera_props.height, camera_props.width, 4)
            record_frames.append(rgba[..., :3].copy())

        path_id_box[0] = path_id + 1

    hold_frames = 120
    for _ in range(len(robot_path) + hold_frames):
        _animate_robot_path_step()

    # Save Video Outputs
    if getattr(args, 'record', False):
        try:
            import imageio
            out_3d = _student_eval_output_path(getattr(args, "record_output", "ntfield_student_3d.mp4"))
            imageio.mimsave(out_3d, record_frames, fps=60)
            print(f"Saved 3D Simulation video to {out_3d}")
            
            if plot_frames:
                out_plot = out_3d.replace('.mp4', '_loss_plot.mp4')
                imageio.mimsave(out_plot, plot_frames, fps=30)
                print(f"Saved Latent Loss Plot video to {out_plot}")
        except (ImportError, ValueError):
            import cv2
            out_3d = _student_eval_output_path(getattr(args, "record_output", "ntfield_student_3d.mp4"))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            h, w = record_frames[0].shape[:2]
            writer = cv2.VideoWriter(out_3d, fourcc, 60.0, (w, h))
            for f in record_frames:
                writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            writer.release()
            print(f"Saved 3D Simulation video to {out_3d} (via OpenCV)")

    print('Test Completed Successfully!!')
    if viewer is not None:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
    sys.exit(1)