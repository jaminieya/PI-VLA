#
# student_model_evaluation/plan_ntfield_with_student.py
#
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R
from isaacgym import gymapi
from isaacgym import gymutil
import fcl

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
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import robot_arm_configuration as RC
from stl_reader import stl_reader
from obj_reader import obj_reader
from trajectory_evaluation.ntfield.eval_trajectory_ntfield import _ModelShim, load_network_and_function
from planning.gradient_planner_trajectory import SCALE as NTFIELD_SCALE
from planning.gradient_planner_trajectory import plan as ntfield_plan

# --- Fixed benchmark layout ---
TABLE_DIMS_X = 0.8
TABLE_DIMS_Y = 1.0
TABLE_DIMS_Z = 0.10
DRAWER_HEIGHT = 0.40
NUM_OF_OBJECTS = 1
BANANA_ASSET_IDX = 5
TARGET_OBJ_INDEX = [BANANA_ASSET_IDX]

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


def ntfield_latent_plan_gradient(teacher_network, q_start, z_goal_hat,
                                  q_goal_norm=None,
                                  step_size=0.2, max_steps=200, tol=0.01, device="cuda"):
    q_curr_norm = np.asarray(q_start, dtype=np.float32) / NTFIELD_SCALE
    q_curr_t = torch.tensor(q_curr_norm, dtype=torch.float32, device=device).unsqueeze(0)
    q_curr_t.requires_grad_(True)
    path_norm = [q_curr_norm.copy()]

    for step in range(max_steps):
        dist, _, coords_out = teacher_network.out_with_goal_latent(
            q_curr_t, z_goal_hat, q_goal_norm=q_goal_norm
        )

        if dist.item() < tol:
            print(f"[latent planner] converged at step {step}, dist={dist.item():.6f}")
            break

        grad_out = torch.autograd.grad(dist, coords_out)[0]
        grad_start = grad_out[:, :6]   # raw dtau w.r.t. start side

        # Mirror Gradient() exactly:
        # Ypred0 = -dtau[:, :dim]
        # Spred0 = norm(Ypred0)
        # step = +step_size * 1/Spred0^2 * Ypred0
        direction = -grad_start
        norm_dir = torch.norm(direction, dim=1, keepdim=True)
        direction = direction / (norm_dir ** 2 + 1e-8)

        with torch.no_grad():
            q_curr_t = q_curr_t + step_size * direction   # positive step, matches plan()

        q_curr_t = q_curr_t.detach().requires_grad_(True)
        path_norm.append(q_curr_t.detach().cpu().numpy()[0].copy())

    else:
        print(f"[latent planner] reached max_steps={max_steps}, final dist={dist.item():.6f}")

    return [p * NTFIELD_SCALE for p in path_norm]
    
def mppi_plan_latent_space(teacher_network, q_start, z_goal_hat, scale, steps=200, sample_num=50, horizon=5, device=None) -> np.ndarray:
    """MPPI planner evaluating paths against the goal latent vector."""
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


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class StudentHead(nn.Module):
    def __init__(self, in_features, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, output_dim),
        )
        # Running stats for stable eval — fixes batch_size=1 issue
        self.register_buffer("_running_mean", torch.zeros(output_dim))
        self.register_buffer("_running_var",  torch.ones(output_dim))
        self._momentum = 0.01

    def _apply_encoder_norm(self, y):
        if self.training:
            mean = y.mean(dim=0)
            var  = y.var(dim=0, unbiased=False)
            with torch.no_grad():
                self._running_mean.mul_(1 - self._momentum).add_(mean, alpha=self._momentum)
                self._running_var.mul_(1 - self._momentum).add_(var,  alpha=self._momentum)
        else:
            mean = self._running_mean
            var  = self._running_var
        return (y - mean) / torch.sqrt(var + 1e-5)

    def forward(self, x):
        return self._apply_encoder_norm(self.net(x))


class StudentModel(nn.Module):
    """Defined at module level so torch.save/load and pickle work correctly."""

    def __init__(self, output_dim: int):
        super().__init__()
        try:
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except (TypeError, AttributeError):
            backbone = models.resnet18(pretrained=True)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = StudentHead(in_features, output_dim)

    def forward(self, x):
        return self.head(self.backbone(x))


def _get_latent_model_new(output_dim: int) -> nn.Module:
    return StudentModel(output_dim)


def _get_latent_model(output_dim: int = 256):
    import torch.nn as nn
    import torchvision.models as models

    try:
        model = models.resnet18(weights=None)
    except TypeError:
        model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, output_dim)
    return model


def _normalize_student_checkpoint(ckpt):
    """Return (state_dict, metadata) for various save formats."""
    meta = {}
    if isinstance(ckpt, dict) and "student_state_dict" in ckpt:
        meta = {k: v for k, v in ckpt.items() if k != "student_state_dict"}
        return ckpt["student_state_dict"], meta
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        meta = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
        return ckpt["model_state_dict"], meta
    if isinstance(ckpt, dict):
        return ckpt, meta
    return ckpt, meta


def _infer_latent_output_dim(meta: dict, sd: dict) -> int:
    if "head.net.3.weight" in sd:
        return int(sd["head.net.3.weight"].shape[0])
    if "fc.weight" in sd:
        return int(sd["fc.weight"].shape[0])
    for key in ("z_dim", "ntfield_h"):
        if key in meta and meta[key] is not None:
            return int(meta[key])
    return 256


def _load_image_student_from_checkpoint(sd: dict, meta: dict, dev: torch.device) -> nn.Module:
    """Load ResNet18 + latent head weights used by Isaac image inference."""
    output_dim = _infer_latent_output_dim(meta, sd)
    if any(k.startswith("backbone.") for k in sd):
        model = _get_latent_model(output_dim=output_dim).to(dev)
        model.load_state_dict(sd, strict=True)
        return model
    if "fc.weight" in sd:
        model = _get_latent_model(output_dim=output_dim).to(dev)
        model.load_state_dict(sd, strict=True)
        return model
    sample = ", ".join(list(sd.keys())[:12])
    raise RuntimeError(
        "Unrecognized student weights for image-only NTField inference (expected "
        "StudentModel `backbone.*` + `head.*`, or ResNet18 + `fc.*`). "
        f"First keys: {sample}. "
        "GoalLatentPredictorWithFiLM / ResNet50 image-only checkpoints need a different inference path."
    )


def _process_img(img: np.ndarray, img_size: int) -> torch.Tensor:
    from torchvision import transforms

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


def _infer_latent_on_image(image: np.ndarray, checkpoint_path: str, device: str) -> np.ndarray:
    """Run image→latent student on a HWC uint8 RGB array. Returns (z_dim,) numpy (default 256)."""
    from torchvision import transforms

    dev = torch.device("cuda" if torch.cuda.is_available() and device == "auto" else device)
    raw = torch.load(os.path.abspath(checkpoint_path), map_location=dev)
    sd, meta = _normalize_student_checkpoint(raw)
    model = _load_image_student_from_checkpoint(sd, meta, dev)
    model.eval()

    x = _process_img(image, 224).unsqueeze(0)
    x = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x).to(dev)

    with torch.no_grad():
        return model(x).squeeze(0).cpu().numpy()


# -------------------------------------------------------------------------

def _resolve_pi_vla_checkpoint(path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(_PI_VLA_ROOT, path))


def _path_as_6_list(path):
    if not path:
        return None
    return [np.asarray(p, dtype=np.float64).reshape(-1)[:6].tolist() for p in path]


def get_swept_volume_size(main_swept):
    min_y, max_y = sys.maxsize, -sys.maxsize
    for _, ty, _ in main_swept:
        min_y = min(min_y, ty)
        max_y = max(max_y, ty)
    return max_y - min_y


def joint_metrics(path_6, q_start, q_goal):
    if not path_6:
        return {}
    arr = np.array(path_6, dtype=np.float64).reshape(-1, 6)
    q0 = np.asarray(q_start, dtype=np.float64).reshape(6)
    qg = np.asarray(q_goal, dtype=np.float64).reshape(6)
    out = {
        "joint_net_abs_delta_rad": np.abs(qg - q0).tolist(),
        "joint_net_abs_delta_l1_rad": float(np.sum(np.abs(qg - q0))),
        "joint_net_abs_delta_l2_rad": float(np.linalg.norm(qg - q0)),
    }
    if len(arr) < 2:
        out["joint_cumulative_abs_delta_per_joint_rad"] = [0.0] * 6
        out["path_segment_l1_sum_rad"] = 0.0
        out["path_segment_l2_sum_rad"] = 0.0
        return out
    d = np.abs(np.diff(arr, axis=0))
    out["joint_cumulative_abs_delta_per_joint_rad"] = np.sum(d, axis=0).tolist()
    out["path_segment_l1_sum_rad"] = float(np.sum(d))
    out["path_segment_l2_sum_rad"] = float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))
    return out


def _settle_steps_at_waypoint(path_local, waypoint_idx):
    if not path_local or waypoint_idx <= 0:
        return SETTLE_STEPS
    dq = float(np.max(np.abs(
        np.asarray(path_local[waypoint_idx], dtype=np.float64) -
        np.asarray(path_local[waypoint_idx - 1], dtype=np.float64)
    )))
    return max(SETTLE_STEPS, min(600, int(math.ceil(dq / RAD_PER_SIM_STEP_HEURISTIC))))


def _append_cam_rgb(gym, sim, env, main_cam_handle, camera_props, record_rgb):
    if record_rgb is None or main_cam_handle is None:
        return
    gym.render_all_camera_sensors(sim)
    raw = gym.get_camera_image(sim, env, main_cam_handle, gymapi.IMAGE_COLOR)
    record_rgb.append(raw.reshape(camera_props.height, camera_props.width, 4)[..., :3].copy())


def _save_mp4_rgb(frames, out_mp4, fps=60.0):
    if not frames:
        return
    os.makedirs(os.path.dirname(os.path.abspath(out_mp4)), exist_ok=True)
    try:
        import imageio
        imageio.mimsave(out_mp4, frames, fps=fps)
    except Exception:
        import cv2
        h, w = frames[0].shape[:2]
        wri = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for fr in frames:
            wri.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        wri.release()


def execute_path_and_time(
    gym, sim, env, ur_handle, spj, slj, ej, wj1, wj2, wj3, viewer, path_local, label,
    main_cam_handle=None, camera_props=None, record_rgb=None, planner_playback="direct",
):
    if not path_local:
        return {"label": label, "success": False, "execution_wall_s": None, "execution_sim_s": None, "physics_steps": 0}

    def _set_and_step(q):
        gym.set_dof_target_position(env, spj, float(q[0]))
        gym.set_dof_target_position(env, slj, float(q[1]))
        gym.set_dof_target_position(env, ej,  float(q[2]))
        gym.set_dof_target_position(env, wj1, float(q[3]))
        gym.set_dof_target_position(env, wj2, float(q[4]))
        gym.set_dof_target_position(env, wj3, float(q[5]))
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        if viewer is not None:
            gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)
        _append_cam_rgb(gym, sim, env, main_cam_handle, camera_props, record_rgb)

    t0 = time.perf_counter()
    n_sub = 0
    for idx, q in enumerate(path_local):
        n_hold = _settle_steps_at_waypoint(path_local, idx) if planner_playback == "settle" else 1
        for _ in range(n_hold):
            _set_and_step(q)
            n_sub += 1
    for _ in range(FINAL_HOLD_STEPS):
        _set_and_step(path_local[-1])
        n_sub += 1
    t1 = time.perf_counter()

    return {
        "label": label, "success": True,
        "execution_wall_s": float(t1 - t0),
        "execution_sim_s": float(n_sub * sim_dt),
        "physics_steps": int(n_sub),
    }


def reset_arm_to_q(gym, sim, env, ur_handle, spj, slj, ej, wj1, wj2, wj3, viewer, q, n_steps=200):
    for _ in range(n_steps):
        gym.set_dof_target_position(env, spj, float(q[0]))
        gym.set_dof_target_position(env, slj, float(q[1]))
        gym.set_dof_target_position(env, ej,  float(q[2]))
        gym.set_dof_target_position(env, wj1, float(q[3]))
        gym.set_dof_target_position(env, wj2, float(q[4]))
        gym.set_dof_target_position(env, wj3, float(q[5]))
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        if viewer is not None:
            gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)


def main():
    parser = argparse.ArgumentParser(description="RRTConnect vs NTField benchmark.")
    parser.add_argument("--object_x", type=float, required=True)
    parser.add_argument("--object_y", type=float, required=True)
    parser.add_argument("--object_z", type=float, required=True)
    parser.add_argument("--ntfield_checkpoint", type=str, required=True)
    parser.add_argument("--ntfield_experiment_dir", type=str, default=None)
    parser.add_argument("--ntfield_device", type=str, default="cuda:0")
    parser.add_argument("--ntfield_step_size", type=float, default=0.02)
    parser.add_argument("--ntfield_max_steps", type=int, default=200)
    parser.add_argument("--ntfield_tol", type=float, default=0.01)
    parser.add_argument("--ntfield_goal_eps_rad", type=float, default=None)
    parser.add_argument("--student_checkpoint", type=str, default=None)
    parser.add_argument("--student_planner", type=str, default="gradient_latent", choices=["mppi_latent", "gradient_latent"])
    parser.add_argument("--use_viewer", action="store_true")
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--record_dir", type=str, default=None)
    parser.add_argument("--video_fps", type=float, default=60.0)
    parser.add_argument("--no_video", action="store_true")
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

    ntfield_network, ntfield_fn = load_network_and_function(ckpt_abs, args.ntfield_experiment_dir, dev, dim=6)
    nt_model = _ModelShim(ntfield_fn)
    ntfield_network.eval()

    # Print the running mean and variance of the encoder norm
    print(ntfield_network.encoder_norm.running_mean)  # None if not tracked
    print(ntfield_network.encoder_norm.running_var)


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
    object_asset_files, object_collision_files, object_offset = [], [], []
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
        R.from_euler("x",  [90],       degrees=True),
        R.from_euler("xy", [90, 180],  degrees=True),
        R.from_euler("xy", [180, 180], degrees=True),
        R.from_euler("z",  [-180],     degrees=True),
        R.from_euler("x",  [-180],     degrees=True),
        R.from_euler("x",  [90],       degrees=True),
        R.from_euler("z",  [-90],      degrees=True),
    ]
    ur5e_translations = [[0,0,0],[0,0,0],[0,-0.138,0],[0,-0.007,0],[0,0.127,0],[0,0,0],[0,0,0]]
    for idx, parts_path in enumerate(ur5e_collision_parts):
        collision_mesh = stl_reader(asset_root + parts_path)
        collision_mesh.transform(ur5e_rotations[idx], ur5e_translations[idx])
        verts, tris = collision_mesh.get_vertices(), collision_mesh.get_faces()
        m = fcl.BVHModel()
        m.beginModel(len(verts), len(tris))
        m.addSubModel(verts, tris)
        m.endModel()
        ur5e_collision_models.append(m)

    viewer = None
    if not gym_args.headless:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is None:
            gym_args.headless = True

    env_lower = gymapi.Vec3(-2, -2, 0)
    env_upper = gymapi.Vec3( 2,  2, 0)

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)
    asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
    asset_options.use_mesh_materials = True
    ur5e_asset  = gym.load_asset(sim, asset_root, ur5e_asset_file, asset_options)
    table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

    asset_options.fix_base_link = False
    object_assets = [gym.load_asset(sim, asset_root, ob, asset_options) for ob in object_asset_files]

    ur5e_pose  = gymapi.Transform()
    ur5e_pose.p = gymapi.Vec3(0, 0, 0)
    ur5e_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0.5 * math.pi)
    table_pose = gymapi.Transform()
    table_pose.p = gymapi.Vec3(table_dims.x * 0.5 + 0.3, 0.0, table_dims.z * 0.5)

    camera_props = gymapi.CameraProperties()
    camera_props.horizontal_fov = 70.25
    camera_props.width  = 1280
    camera_props.height = 720

    col_plane = fcl.Plane(np.array([0.0, 0.0, 1.0]), 0)
    plane_obj = fcl.CollisionObject(col_plane, fcl.Transform())
    col_table  = fcl.Box(table_dims.x, table_dims.y, table_dims.z)
    table_obj  = fcl.CollisionObject(col_table, fcl.Transform(np.array([table_dims.x * 0.5 + 0.3, 0.0, table_dims.z * 0.5])))
    object_collision_models = [table_obj]

    envs, ur5e_handles, object_handles = [], [], []
    object_status_list, object_reader_tracker, object_mesh = [], [], []
    flex_collision_models, object_collision_lib = [], []
    spj = slj = ej = wj1 = wj2 = wj3 = None
    target_file_idx = np.array(TARGET_OBJ_INDEX)
    GT_OBJ_POS_LIST = []

    for i in range(1):
        envs.append(gym.create_env(sim, env_lower, env_upper, 1))
        ur5e_handles.append(gym.create_actor(envs[-1], ur5e_asset, ur5e_pose, f"ur5e{i}", 0, 32767))
        spj = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "shoulder_pan_joint")
        slj = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "shoulder_lift_joint")
        ej  = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "elbow_joint")
        wj1 = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "wrist_1_joint")
        wj2 = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "wrist_2_joint")
        wj3 = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "wrist_3_joint")
        gym.create_actor(envs[-1], table_asset, table_pose, f"table{i}", 0, 1)

        object_scaling_factor = np.ones(NUM_OF_OBJECTS, dtype=np.float64)
        for k in range(NUM_OF_OBJECTS):
            tx, ty, tz = float(args.object_x), float(args.object_y), float(args.object_z)
            object_pose = gymapi.Transform()
            object_pose.p = gymapi.Vec3(tx, ty, tz)
            collision_mesh = obj_reader(asset_root + object_collision_files[target_file_idx[k]])
            collision_mesh.set_scale(object_scaling_factor[k])
            collision_mesh.add_offset(object_offset[target_file_idx[k]])
            verts, tris = collision_mesh.get_bounding_box_mesh()
            m = fcl.BVHModel()
            m.beginModel(len(verts), len(tris))
            m.addSubModel(verts, tris)
            m.endModel()
            object_handles.append(gym.create_actor(
                envs[-1], object_assets[target_file_idx[k]], object_pose,
                f"object{k}{i}", 0, 2 ** (k + 1), k + 1,
            ))
            gym.set_actor_scale(envs[-1], object_handles[-1], object_scaling_factor[k])
            object_reader_tracker.append(collision_mesh)
            object_status_list.append([collision_mesh.get_center(), collision_mesh.get_bounding_box()])
            object_collision_lib.append(m)
            GT_OBJ_POS_LIST.append([tx, ty])
            objs_manager = fcl.DynamicAABBTreeCollisionManager()
            objs_manager.registerObjects([fcl.CollisionObject(m, fcl.Transform(np.array([tx, ty, tz])))])
            objs_manager.setup()

        top_cam_handle = gym.create_camera_sensor(envs[-1], camera_props)
        gym.set_camera_location(
            top_cam_handle, envs[-1],
            gymapi.Vec3(table_pose.p.x, table_pose.p.y + 0.001, 2.2),
            gymapi.Vec3(table_pose.p.x - 0.5, table_pose.p.y, table_pose.p.z),
        )
        main_cam_handle = top_cam_handle

    if viewer is not None:
        gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(2.2, 0, 0.5), gymapi.Vec3(0, 0, 0.5))

    gym.set_light_parameters(sim, 0, gymapi.Vec3(0.3, 0.3, 0.3), gymapi.Vec3(1.0, 1.0, 1.0), gymapi.Vec3(-1.0, 0.0, 0.0))
    gym.set_light_parameters(sim, 1, gymapi.Vec3(0.3, 0.3, 0.3), gymapi.Vec3(1.0, 1.0, 1.0), gymapi.Vec3( 1.0, 0.0, 0.0))

    env = envs[-1]
    ur  = ur5e_handles[-1]
    real_position = False

    for t in range(2000):
        if not real_position:
            gym.set_dof_target_position(env, spj, 0)
            gym.set_dof_target_position(env, slj, -math.pi / 2)
            gym.set_dof_target_position(env, ej,  0)
            gym.set_dof_target_position(env, wj1, -math.pi / 2)
            gym.set_dof_target_position(env, wj2, 0)
            gym.set_dof_target_position(env, wj3, 0)
            real_position = True
        if t == 999:
            for ii, element in enumerate(object_handles):
                states = gym.get_actor_rigid_body_states(env, element, 1)
                translation = np.array(np.array(states[0][0][0]).item())
                rotation    = np.array(np.array(states[0][0][1]).item())
                object_status_list[ii][0] += translation
                tf = fcl.Transform(R.from_quat(rotation).as_matrix(), translation)
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
    rac = RC.robot_arm_configuration(
        "./assets/urdf/ur5e/meshes/collision/",
        np.array([ur5e_pose.p.x, ur5e_pose.p.y, ur5e_pose.p.z]),
        scene_info,
    )

    grasp_file = "./assets/" + "/".join(object_asset_files[target_file_idx[0]].split("/")[:-1]) + "/grasp_dict.npy"
    grasp_data = np.load(grasp_file, allow_pickle=True)
    grasp_list = np.arange(len(grasp_data))
    np.random.shuffle(grasp_list)

    num_grasp = 0
    swept_size = sys.maxsize
    init2grasp_path = None
    grasp_target_q = None
    _rrt_plan_s = 0.0
    consecutive_path_failures = 0
    MAX_FAIL = 15
    target_idx = 0

    for grasp_idx in grasp_list:
        if consecutive_path_failures >= MAX_FAIL:
            break
        target_grasp_pos  = grasp_data[grasp_idx]["target_pos"].copy()
        target_grasp_quat = grasp_data[grasp_idx]["target_quat"]
        target_grasp_pos[:2] += GT_OBJ_POS_LIST[target_idx][:2]

        init2grasp_angels_temp  = rac.grasp_verify(target_grasp_pos, target_grasp_quat)
        grasp2init_angels_temp  = rac.grasp_verify(target_grasp_pos + [0, 0, 0.01], target_grasp_quat)
        if init2grasp_angels_temp is None or grasp2init_angels_temp is None:
            continue
        if not rac.arm_collision_free(init2grasp_angels_temp, plane_obj, object_collision_models, []):
            continue
        if not rac.arm_collision_free(grasp2init_angels_temp, plane_obj, object_collision_models, []):
            continue

        t_rrt0 = time.perf_counter()
        init2grasp_path_temp = RC.get_path2grasp(
            rac, init2grasp_angels_temp, scene_info,
            target_mesh=object_mesh[target_idx], time_limit=30,
            given_static_model=object_collision_models,
        )
        rrt_planning_wall_s = float(time.perf_counter() - t_rrt0)
        if init2grasp_path_temp is None:
            consecutive_path_failures += 1
            continue

        temp_mod_bbox = rac.modify_grasp_bbox(init2grasp_angels_temp, target_mesh=object_mesh[target_idx], visualize=False)
        grasp2init_path_temp = RC.get_path2start(
            rac, grasp2init_angels_temp, temp_mod_bbox, scene_info,
            time_limit=30, given_static_model=object_collision_models,
        )
        if grasp2init_path_temp is None:
            consecutive_path_failures += 1
            continue

        consecutive_path_failures = 0
        _, swept_verts1 = rac.get_swept_volume(init2grasp_path_temp, frame_rate=60, scene_info=scene_info, animation=False, static_vi=False)
        _, swept_verts2 = rac.get_swept_volume(grasp2init_path_temp, w_target=temp_mod_bbox, frame_rate=60, scene_info=scene_info, animation=False, static_vi=False)
        _, swept_verts  = rac.get_swept_center(swept_verts1 + swept_verts2, scene_info, 0.6)

        temp_swept_size = get_swept_volume_size(swept_verts)
        if temp_swept_size < swept_size:
            swept_size       = temp_swept_size
            init2grasp_path  = init2grasp_path_temp
            grasp_target_q   = np.asarray(init2grasp_angels_temp, dtype=np.float64).reshape(6)
            _rrt_plan_s      = rrt_planning_wall_s
        num_grasp += 1
        if num_grasp == 1:
            break

    dof_snapshot  = gym.get_actor_dof_states(env, ur, gymapi.STATE_POS)
    q_start_live  = np.array(dof_snapshot["pos"][:6], dtype=np.float64)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.abspath(args.record_dir) if args.record_dir else \
        os.path.join(_PI_VLA_ROOT, "output", "trajectory_evaluation", f"benchmark_{stamp}")
    os.makedirs(session_dir, exist_ok=True)
    want_video = not args.no_video

    result = {
        "timestamp": datetime.now().isoformat(),
        "planner_playback": args.planner_playback,
        "table_dims_m": [TABLE_DIMS_X, TABLE_DIMS_Y, TABLE_DIMS_Z],
        "object_pose_world_m": [args.object_x, args.object_y, args.object_z],
        "object": "011_banana",
        "q_start_live": q_start_live.tolist(),
        "goal_configuration_grasp_verify": grasp_target_q.tolist() if grasp_target_q is not None else None,
        "ntfield_checkpoint": ckpt_abs,
        "rrtconnect": {}, "ntfield": {}, "student": {},
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

    # ----------------------------------------------------
    # RRT Execution
    # ----------------------------------------------------
    path_rrt = _path_as_6_list(init2grasp_path)
    result["rrtconnect"]["planning_wall_s_for_get_path2grasp_only"] = _rrt_plan_s
    result["rrtconnect"]["success"] = True
    result["rrtconnect"]["num_waypoints"] = len(path_rrt)
    result["rrtconnect"]["motion"] = joint_metrics(path_rrt, q_start_live, grasp_target_q)

    frames_rrt = [] if want_video else None
    result["rrtconnect"]["execution"] = execute_path_and_time(
        gym, sim, env, ur, spj, slj, ej, wj1, wj2, wj3, viewer, path_rrt, "rrt",
        main_cam_handle, camera_props, frames_rrt, args.planner_playback,
    )
    if want_video and frames_rrt:
        mp4_rrt = os.path.join(session_dir, "rrt.mp4")
        _save_mp4_rgb(frames_rrt, mp4_rrt, fps=args.video_fps)
        result["rrtconnect"]["video_path"] = mp4_rrt

    # ----------------------------------------------------
    # Teacher NTField Planning
    # ----------------------------------------------------
    reset_arm_to_q(gym, sim, env, ur, spj, slj, ej, wj1, wj2, wj3, viewer, q_start_live, n_steps=200)

    t_nt0 = time.perf_counter()
    path_nt_raw = ntfield_plan(
        nt_model, q_start_live, grasp_target_q,
        step_size=args.ntfield_step_size, max_steps=args.ntfield_max_steps,
        tol=args.ntfield_tol, device=ntfield_device_str,
    )
    nt_planning_wall_s = float(time.perf_counter() - t_nt0)
    nt_has_path = path_nt_raw is not None and len(path_nt_raw) >= 2
    nt_err_pen  = float(np.linalg.norm(np.asarray(path_nt_raw[-2], dtype=np.float64).reshape(6) - grasp_target_q)) \
        if nt_has_path else None

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
        result["ntfield"]["motion"] = joint_metrics(path_nt, q_start_live, grasp_target_q)
        frames_nt = [] if want_video else None
        result["ntfield"]["execution"] = execute_path_and_time(
            gym, sim, env, ur, spj, slj, ej, wj1, wj2, wj3, viewer, path_nt, "ntfield",
            main_cam_handle, camera_props, frames_nt, args.planner_playback,
        )
        if want_video and frames_nt:
            mp4_nt = os.path.join(session_dir, "ntfield.mp4")
            _save_mp4_rgb(frames_nt, mp4_nt, fps=args.video_fps)
            result["ntfield"]["video_path"] = mp4_nt

    # ----------------------------------------------------
    # Student NTField Planning
    # ----------------------------------------------------
    if args.student_checkpoint is not None:
        reset_arm_to_q(gym, sim, env, ur, spj, slj, ej, wj1, wj2, wj3, viewer, q_start_live, n_steps=200)
        t_stu0 = time.perf_counter()

        gym.render_all_camera_sensors(sim)
        raw_top = gym.get_camera_image(sim, env, main_cam_handle, gymapi.IMAGE_COLOR)
        rgb_top = raw_top.reshape(camera_props.height, camera_props.width, 4)[..., :3]

        z_goal_hat = torch.tensor(
            _infer_latent_on_image(rgb_top, args.student_checkpoint, ntfield_device_str),
            dtype=torch.float32, device=dev,
        ).unsqueeze(0)

        if args.student_planner == "gradient_latent":
            path_stu_raw = ntfield_latent_plan_gradient(
                teacher_network=ntfield_network,
                q_start=q_start_live,
                z_goal_hat=z_goal_hat,
                q_goal_norm=None,   # not available at inference
                step_size=args.ntfield_step_size,
                max_steps=args.ntfield_max_steps,
                tol=args.ntfield_tol,
                device=dev,
            )
        else:
            path_stu_raw = mppi_plan_latent_space(
                teacher_network=ntfield_network, q_start=q_start_live,
                z_goal_hat=z_goal_hat, scale=NTFIELD_SCALE, device=dev,
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
            result["student"]["motion"] = joint_metrics(path_stu, q_start_live, grasp_target_q)
            result["student"]["reached_destination_rad"] = path_stu[-1]
            frames_stu = [] if want_video else None
            result["student"]["execution"] = execute_path_and_time(
                gym, sim, env, ur, spj, slj, ej, wj1, wj2, wj3, viewer, path_stu, "student",
                main_cam_handle, camera_props, frames_stu, args.planner_playback,
            )
            if want_video and frames_stu:
                mp4_stu = os.path.join(session_dir, "ntfield_student.mp4")
                _save_mp4_rgb(frames_stu, mp4_stu, fps=args.video_fps)
                result["student"]["video_path"] = mp4_stu

        # ---- True latent goal sanity check ----
        with torch.no_grad():
            qs_norm = torch.tensor(q_start_live   / NTFIELD_SCALE, dtype=torch.float32, device=dev).unsqueeze(0)
            qg_norm = torch.tensor(grasp_target_q / NTFIELD_SCALE, dtype=torch.float32, device=dev).unsqueeze(0)
            coords_true = torch.cat([qs_norm, qg_norm], dim=1)
            _, z_goal_true, _, _ = ntfield_network._embed_start_goal(coords_true)

        # Compare z_goal_true (teacher embedding of true goal) vs z_goal_hat (student image latent)
        z_true_np = z_goal_true.detach().cpu().numpy()
        z_hat_np = z_goal_hat.detach().cpu().numpy()
        diff_np = z_true_np - z_hat_np
        l2 = float(np.linalg.norm(diff_np))
        latent_log_path = os.path.join(session_dir, "latent_goal_debug.txt")
        _fmt = lambda a: np.array2string(
            a, threshold=a.size, max_line_width=200, precision=8, suppress_small=False
        )
        with open(latent_log_path, "w") as lf:
            lf.write("z_goal_true (teacher latent, shape %s):\n%s\n\n" % (z_true_np.shape, _fmt(z_true_np)))
            lf.write("z_goal_hat (student latent, shape %s):\n%s\n\n" % (z_hat_np.shape, _fmt(z_hat_np)))
            lf.write("z_goal_true - z_goal_hat:\n%s\n\n" % _fmt(diff_np))
            lf.write("L2 norm ||z_true - z_hat||: %r\n" % l2)
            lf.write("L2 < 1e-4: %s\n" % (l2 < 1e-4))
        result["latent_goal_debug_txt"] = latent_log_path
        print("Wrote latent vectors and L2 summary to %s" % latent_log_path)

        reset_arm_to_q(gym, sim, env, ur, spj, slj, ej, wj1, wj2, wj3, viewer, q_start_live, n_steps=200)
        t_true0 = time.perf_counter()

        path_true_raw = ntfield_latent_plan_gradient(
            teacher_network=ntfield_network,
            q_start=q_start_live,
            z_goal_hat=z_goal_true,
            q_goal_norm=None,
            step_size=args.ntfield_step_size,
            max_steps=args.ntfield_max_steps,
            tol=args.ntfield_tol,
            device=dev,
        )

        true_planning_wall_s = float(time.perf_counter() - t_true0)
        true_has_path = path_true_raw is not None and len(path_true_raw) >= 2
        true_err = float(np.linalg.norm(
            np.asarray(path_true_raw[-2], dtype=np.float64).reshape(6) - grasp_target_q
        )) if true_has_path else None

        sanity_line = (
            f"[sanity] true-latent planner: steps={len(path_true_raw) if path_true_raw else 0}, "
            + (f"penultimate_err={true_err:.4f} rad" if true_err is not None else "no path")
        )
        with open(latent_log_path, "a") as lf:
            lf.write("\n%s\n" % sanity_line)
        print(sanity_line)

        result["student_true_latent"] = {
            "planning_wall_s": true_planning_wall_s,
            "success": true_has_path,
            "num_planner_steps": len(path_true_raw) if path_true_raw else 0,
            "penultimate_goal_error_rad": true_err,
        }
        if true_has_path:
            path_true = _path_as_6_list(path_true_raw)
            result["student_true_latent"]["motion"] = joint_metrics(path_true, q_start_live, grasp_target_q)
            result["student_true_latent"]["reached_destination_rad"] = path_true[-1]
            frames_true = [] if want_video else None
            result["student_true_latent"]["execution"] = execute_path_and_time(
                gym, sim, env, ur, spj, slj, ej, wj1, wj2, wj3, viewer, path_true, "true_latent",
                main_cam_handle, camera_props, frames_true, args.planner_playback,
            )
            if want_video and frames_true:
                mp4_true = os.path.join(session_dir, "ntfield_true_latent.mp4")
                _save_mp4_rgb(frames_true, mp4_true, fps=args.video_fps)
                result["student_true_latent"]["video_path"] = mp4_true
                        
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