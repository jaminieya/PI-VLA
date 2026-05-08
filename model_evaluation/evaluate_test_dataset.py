#!/usr/bin/env python3
"""
Offline evaluation over a preprocessed test dataset (built by preprocess_data_evaluation.py).

For each record in the test shards:
  1. Reconstruct the Isaac Gym scene using stored object IDs + locations
  2. Run the student model to predict z_goal from the stored image
  3. Run NTField gradient planner toward the predicted z_goal
  4. Record success (planner converged within tol) and latent error vs teacher z_goal

Results are written to:
  <output_dir>/
    results.jsonl        — one JSON line per record
    summary.json         — aggregate metrics (success rate, latent errors, etc.)

Example:
  python evaluate_test_dataset.py \\
    --test_dataset /home/hojinsohn/VLM-NT/PI-VLA/hanwen_grasping/output/multi_obj/test_run_pt_shards \\
    --latent_checkpoint /home/hojinsohn/VLM-NT/PI-VLA/student_model_training/best_z_goal_model_mdn_mdn_K8_bs256_lr3em4_ep90_20260505_114200.pth \\
    --ntfield_checkpoint /home/hojinsohn/VLM-NT/PI-VLA/teacher_model.pt \\
    --output_dir /home/hojinsohn/VLM-NT/PI-VLA/output/eval_results/mdn_ep90

Run from PI-VLA root.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# IMPORTANT: isaacgym must be imported before torch.
# isaacgym's gymdeps raises if torch was imported first.
from isaacgym import gymapi, gymutil
import torch
from torchvision import transforms
from tqdm import tqdm

# ── Path setup (mirrors integration script) ───────────────────────────────────
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


# ── Student model inference ───────────────────────────────────────────────────

def _tokenize_prompt(text: str) -> list:
    return re.findall(r"[a-z0-9]+", text.lower())

def _encode_prompts(prompts, token_to_id, max_len):
    pad_id = token_to_id.get("<pad>", 0)
    unk_id = token_to_id.get("<unk>", 1)
    out = torch.full((len(prompts), max_len), pad_id, dtype=torch.long)
    for i, p in enumerate(prompts):
        toks = _tokenize_prompt(p)[:max_len]
        if not toks:
            toks = ["<unk>"]
        ids = [token_to_id.get(t, unk_id) for t in toks]
        out[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return out

def load_student_model(checkpoint_path: str, device: torch.device):
    """
    Load student checkpoint. Supports MDNStudent and RegressionStudent.
    Returns (model, meta) where meta contains z_dim, model_type, token_to_id, etc.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_type   = ckpt.get("model_type", "mdn")
    z_dim        = ckpt.get("z_dim", 256)
    vocab_size   = ckpt.get("vocab_size", None)
    token_to_id  = ckpt.get("token_to_id", None)
    max_prompt_len = int(ckpt.get("max_prompt_len", 8))
    n_components = int(ckpt.get("n_components", 8))

    if model_type == "regression":
        from student_model_regression import RegressionStudent
        model = RegressionStudent(output_dim=z_dim).to(device)
    else:
        # Default: MDNStudent
        from student_model_mdn import MDNStudent
        model = MDNStudent(
            output_dim=z_dim,
            vocab_size=max(int(vocab_size or 2), 2),
            n_components=n_components,
        ).to(device)

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    meta = {
        "model_type":    model_type,
        "z_dim":         z_dim,
        "vocab_size":    vocab_size,
        "token_to_id":   token_to_id,
        "max_prompt_len": max_prompt_len,
        "n_components":  n_components,
        "checkpoint":    checkpoint_path,
    }
    return model, meta


@torch.no_grad()
def predict_z_goal(
    model,
    meta: Dict,
    image_tensor: torch.Tensor,   # (3, H, W) float32 in [0,1]
    object_name: str,
    device: torch.device,
    normalize,
    num_samples: int = 30,
) -> np.ndarray:
    """
    Run student inference on a single image tensor.
    Returns z_pred as (z_dim,) numpy float32.
    """
    x = normalize(image_tensor).unsqueeze(0).to(device)   # (1, 3, H, W)

    model_type   = meta["model_type"]
    token_to_id  = meta["token_to_id"]
    max_prompt_len = meta["max_prompt_len"]

    if model_type == "regression":
        z_pred = model(x)                                  # (1, z_dim)
        return z_pred.squeeze(0).cpu().numpy()

    # MDN: draw num_samples and pick the one closest to the ensemble mean
    if token_to_id:
        prompt = f"grasp {object_name.strip().lower()}"
        text_tokens = _encode_prompts([prompt], token_to_id, max_prompt_len).to(device)
    else:
        text_tokens = torch.zeros((1, max_prompt_len), dtype=torch.long, device=device)

    preds = model.get_multiple_latent_predictions(
        x, text_tokens, num_samples=num_samples
    )  # (S, 1, z_dim)

    mean_pred = preds.mean(dim=0)                          # (1, z_dim)
    dists     = ((preds - mean_pred) ** 2).sum(dim=-1)     # (S, 1)
    best_idx  = dists.argmin(dim=0)                        # (1,)
    best_pred = preds[best_idx, torch.arange(1)]           # (1, z_dim)
    return best_pred.squeeze(0).cpu().numpy()


# ── NTField planner (extracted from integration script) ───────────────────────

def ntfield_plan_latent(
    teacher_network,
    q_start:    np.ndarray,
    z_goal_hat: np.ndarray,
    step_size:  float = 0.02,
    max_steps:  int   = 200,
    tol:        float = 0.01,
    device:     str   = "cuda",
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Simple single-phase latent-space NTField gradient descent.
    Returns (path_raw, meta) where meta contains final_latent_dist and stopped reason.
    """
    from planning.gradient_planner_trajectory import SCALE as NTFIELD_SCALE

    def _latent_dist(qn, zg):
        with torch.no_grad():
            d, _, _ = teacher_network.out_with_goal_latent(qn.detach(), zg)
            return float(d.item())

    def _grad_step(q_t, zg, stp):
        q_t = q_t.detach().requires_grad_(True)
        dist, _, coords_out = teacher_network.out_with_goal_latent(q_t, zg)
        d_pre = float(dist.item())
        if d_pre < tol:
            return q_t.detach(), d_pre, True
        grad_out   = torch.autograd.grad(dist, coords_out)[0]
        grad_start = grad_out[:, :6]
        with torch.no_grad():
            q_next = q_t - stp * grad_start
        d_post = _latent_dist(q_next, zg)
        return q_next.detach(), d_post, d_post < tol

    q_start    = np.asarray(q_start, dtype=np.float32).reshape(-1)
    q_curr_norm = q_start / NTFIELD_SCALE
    q_curr_t   = torch.tensor(q_curr_norm, dtype=torch.float32, device=device).unsqueeze(0)

    if isinstance(z_goal_hat, np.ndarray):
        z_goal_hat = torch.tensor(
            z_goal_hat.reshape(1, -1), dtype=torch.float32, device=device
        )
    else:
        z_goal_hat = z_goal_hat.reshape(1, -1).to(device)

    path_norm  = [q_curr_norm.copy()]
    final_dist = None
    converged  = False
    stopped    = "max_steps"

    for _ in range(max_steps):
        q_curr_t, final_dist, converged = _grad_step(q_curr_t, z_goal_hat, step_size)
        path_norm.append(q_curr_t.detach().cpu().numpy()[0].copy())
        if converged:
            stopped = "latent_tol"
            break

    if len(path_norm) < 2:
        path_norm.append(q_curr_t.detach().cpu().numpy()[0].copy())

    path_raw = [p * NTFIELD_SCALE for p in path_norm]
    meta     = {"final_latent_dist": final_dist, "stopped": stopped, "path_len": len(path_raw)}
    return path_raw, meta


# ── Scene reconstruction helpers ──────────────────────────────────────────────

def build_scene(
    gym, sim, env,
    ur5e_handles,
    object_assets,
    object_collision_files,
    object_offset,
    asset_root,
    all_object_ids:       List[int],
    all_object_locations: List[List[float]],
    spj, slj, ej, wj1, wj2, wj3,
    viewer,
):
    """
    Place objects at the stored XYZ locations from the test record.
    Returns (object_handles, object_mesh, flex_collision_models, object_status_list).
    """
    from scipy.spatial.transform import Rotation as R
    import fcl
    from obj_reader import obj_reader
    from trajectory_evaluation.comparison.run_rrt_ntfield_benchmark import sim_dt

    env_h = env
    object_handles    = []
    object_mesh       = []
    flex_col_models   = []
    object_status     = []
    object_col_lib    = []

    for k, (obj_id, xyz) in enumerate(zip(all_object_ids, all_object_locations)):
        # ── Load collision mesh ───────────────────────────────────────────
        file_path = object_collision_files[obj_id]
        col_mesh  = obj_reader(asset_root + file_path)
        col_mesh.set_scale(1.0)
        col_mesh.add_offset(object_offset[obj_id])
        verts, tris  = col_mesh.get_bounding_box_mesh()
        temp_center  = col_mesh.get_center()
        temp_bbox    = col_mesh.get_bounding_box()

        m = fcl.BVHModel()
        m.beginModel(len(verts), len(tris))
        m.addSubModel(verts, tris)
        m.endModel()

        # ── Place at stored location (no random) ─────────────────────────
        tx, ty, tz = float(xyz[0]), float(xyz[1]), float(xyz[2])
        object_pose   = __import__("isaacgym").gymapi.Transform()
        object_pose.p = __import__("isaacgym").gymapi.Vec3(tx, ty, tz)

        handle = gym.create_actor(
            env_h,
            object_assets[obj_id],
            object_pose,
            f"object_{k}",
            0,
            2 ** (k + 1),
            k + 1,
        )
        gym.set_actor_scale(env_h, handle, 1.0)
        object_handles.append(handle)
        object_status.append([temp_center, temp_bbox])
        object_col_lib.append(m)

    return object_handles, object_status, object_col_lib


def warmup_sim(gym, sim, env, spj, slj, ej, wj1, wj2, wj3, viewer,
               object_handles, object_status, object_col_lib, n_steps=2000):
    """
    Run warm-up simulation and collect settled mesh states at t==999,
    exactly matching the integration script.
    """
    from scipy.spatial.transform import Rotation as R
    import fcl
    from isaacgym import gymapi

    object_mesh       = []
    flex_col_models   = []
    real_position     = False

    for t in range(n_steps):
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
                states      = gym.get_actor_rigid_body_states(env, element, 1)
                rotation    = np.array(np.array(states[0][0][1]).item())
                translation = np.array(np.array(states[0][0][0]).item())
                object_status[ii][0] += translation
                r1 = R.from_quat(rotation)
                tf = fcl.Transform(r1.as_matrix(), translation)
                flex_col_models.append([fcl.CollisionObject(object_col_lib[ii], tf), 0])
                from obj_reader import obj_reader  # already imported above
                # reuse col_lib mesh bounding box
                object_mesh.append(None)   # mesh not needed for latent-only eval
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        if viewer is not None:
            gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    # HOME_DOF settle (matches data collection)
    _HOME_DOF = [0.7, -2.0, 2.5, -0.3, 0.7, 0.0]
    for _ in range(30):
        gym.set_dof_target_position(env, spj, _HOME_DOF[0])
        gym.set_dof_target_position(env, slj, _HOME_DOF[1])
        gym.set_dof_target_position(env, ej,  _HOME_DOF[2])
        gym.set_dof_target_position(env, wj1, _HOME_DOF[3])
        gym.set_dof_target_position(env, wj2, _HOME_DOF[4])
        gym.set_dof_target_position(env, wj3, _HOME_DOF[5])
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        if viewer is not None:
            gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    return object_mesh, flex_col_models


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_test_shards(dataset_dir: str) -> List[Dict]:
    """Load all records from test_shard_*.pt files in order."""
    dataset_dir = Path(dataset_dir)
    manifest_path = dataset_dir / "manifest.pt"

    if manifest_path.exists():
        manifest   = torch.load(manifest_path, map_location="cpu")
        shard_paths = manifest.get("shards", [])
        print(f"Manifest: {manifest.get('num_samples')} samples, "
              f"{manifest.get('num_shards')} shards, z_dim={manifest.get('z_dim')}")
    else:
        shard_paths = sorted(str(p) for p in dataset_dir.glob("test_shard_*.pt"))
        print(f"No manifest found, discovered {len(shard_paths)} shard files.")

    if not shard_paths:
        raise FileNotFoundError(f"No test shards found under {dataset_dir}")

    records = []
    for sp in shard_paths:
        shard = torch.load(sp, map_location="cpu")
        records.extend(shard)
    print(f"Loaded {len(records)} test records total.")
    return records


# ── Metrics helpers ───────────────────────────────────────────────────────────

def latent_errors(z_pred: np.ndarray, z_true: np.ndarray) -> Dict[str, float]:
    diff = z_pred - z_true
    cos  = float(np.dot(z_pred, z_true) /
                 (np.linalg.norm(z_pred) * np.linalg.norm(z_true) + 1e-8))
    return {
        "l2":         float(np.linalg.norm(diff)),
        "mse":        float(np.mean(diff ** 2)),
        "mae":        float(np.mean(np.abs(diff))),
        "cos_sim":    cos,
        "cos_dist":   1.0 - cos,
        "max_abs":    float(np.max(np.abs(diff))),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    from isaacgym import gymapi, gymutil
    from trajectory_evaluation.ntfield.eval_trajectory_ntfield import (
        _ModelShim, load_network_and_function,
    )
    from planning.gradient_planner_trajectory import SCALE as NTFIELD_SCALE

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test_dataset",       required=True,
                        help="Directory containing test_shard_*.pt + manifest.pt")
    parser.add_argument("--latent_checkpoint",  required=True,
                        help="Student model checkpoint (.pth)")
    parser.add_argument("--ntfield_checkpoint", required=True,
                        help="NTField / teacher checkpoint (.pt)")
    parser.add_argument("--output_dir",         required=True,
                        help="Where to write results.jsonl + summary.json")
    parser.add_argument("--ntfield_device",     default="cuda:0")
    parser.add_argument("--latent_device",      default="auto")
    parser.add_argument("--ntfield_step_size",  type=float, default=0.02)
    parser.add_argument("--ntfield_max_steps",  type=int,   default=200)
    parser.add_argument("--ntfield_tol",        type=float, default=0.01)
    parser.add_argument(
        "--ee_success_thresh",
        type=float,
        default=0.08,
        help="EE-to-target distance threshold for EE success (metres). Default 0.08m.",
    )
    parser.add_argument(
        "--finger_mid_xy_success_thresh_m",
        type=float,
        default=0.08,
        help="Finger-midpoint -> target XY distance threshold for success (metres).",
    )
    parser.add_argument(
        "--finger_mid_z_success_thresh_m",
        type=float,
        default=0.05,
        help="|finger-midpoint Z diff| threshold for success (metres).",
    )
    parser.add_argument("--between_axis_margin_m", type=float, default=0.005)
    parser.add_argument("--between_lateral_thresh_m", type=float, default=0.02)
    parser.add_argument("--between_z_thresh_m", type=float, default=0.05)
    parser.add_argument("--max_records",        type=int,   default=-1,
                        help="Cap number of records evaluated (default: all).")
    parser.add_argument("--physx_cpu",          action="store_true")
    parser.add_argument("--use_viewer",         action="store_true")
    parser.add_argument("--no_isaac_hard_exit", action="store_true")
    args, argv_remainder = parser.parse_known_args()

    # ── Output dir ────────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    summary_path = out_dir / "summary.json"

    # ── Isaac Gym init ────────────────────────────────────────────────────────
    argv_gym = list(argv_remainder)
    if not args.use_viewer and "--headless" not in argv_gym:
        argv_gym.append("--headless")
    sys.argv = [sys.argv[0]] + argv_gym

    gym      = gymapi.acquire_gym()
    gym_args = gymutil.parse_arguments(description="eval", headless=True, custom_parameters=[])
    gym_args.headless = not args.use_viewer
    if args.physx_cpu:
        gym_args.use_gpu = False

    from trajectory_evaluation.comparison.run_rrt_ntfield_benchmark import (
        TABLE_DIMS_X, TABLE_DIMS_Y, TABLE_DIMS_Z, DRAWER_HEIGHT, sim_dt,
        reset_arm_to_q,
    )

    table_dims   = gymapi.Vec3(TABLE_DIMS_X, TABLE_DIMS_Y, TABLE_DIMS_Z)
    sim_params   = gymapi.SimParams()
    sim_params.substeps  = 2
    sim_params.dt        = sim_dt
    sim_params.up_axis   = gymapi.UP_AXIS_Z
    sim_params.gravity   = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.physx.solver_type            = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads            = gym_args.num_threads
    sim_params.physx.use_gpu                = gym_args.use_gpu
    sim_params.use_gpu_pipeline             = False

    sim = gym.create_sim(
        gym_args.compute_device_id, gym_args.graphics_device_id,
        gym_args.physics_engine, sim_params,
    )
    if sim is None:
        raise SystemExit("Failed to create sim")

    plane_params        = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)

    # ── Load NTField after sim ────────────────────────────────────────────────
    dev_nt = torch.device(
        "cpu" if args.ntfield_device == "cpu" or not torch.cuda.is_available()
        else args.ntfield_device
    )
    nt_net, ntfield_fn = load_network_and_function(
        os.path.abspath(args.ntfield_checkpoint), None, dev_nt, dim=6
    )
    ntfield_device_str = str(dev_nt)
    print(f"NTField loaded on {dev_nt}")

    # ── Load student model ────────────────────────────────────────────────────
    dev_student = torch.device(
        "cuda" if torch.cuda.is_available() and args.latent_device in ("auto", "cuda")
        else "cpu"
    )
    student_model, student_meta = load_student_model(
        os.path.abspath(args.latent_checkpoint), dev_student
    )
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    print(f"Student [{student_meta['model_type']}] loaded on {dev_student}")

    # Reuse the same geometry + success logic as evaluate_ntfield_oracle.py
    from evaluate_ntfield_oracle import (
        _execute_path_and_get_ee,
        _get_rigid_body_world_pos,
        _between_fingers_xy_z,
        OBJECT_HEIGHTS_M,
        OBJECT_HEIGHT_DEFAULT_M,
    )

    # ── Load assets (once, reused across all records) ─────────────────────────
    _cwd_prev = os.getcwd()
    os.chdir(HANWEN_GRASPING_ROOT)
    asset_root = "./assets/"

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link          = True
    asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)
    asset_options.mesh_normal_mode       = gymapi.COMPUTE_PER_VERTEX
    asset_options.use_mesh_materials     = True

    ur5e_asset  = gym.load_asset(sim, asset_root,
                                 "urdf/ur5e/ur5e_mimic_real_gripper_test.urdf", asset_options)
    table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

    object_asset_files:     List[str]        = []
    object_collision_files: List[str]        = []
    object_offset:          List[List[float]] = []
    pfx = "urdf/ycb/"
    with open(asset_root + "urdf/ycb/object_urdf_grasp.txt")      as f:
        object_asset_files     = [pfx + l.strip() for l in f if l.strip()]
    with open(asset_root + "urdf/ycb/object_collision_grasp.txt") as f:
        object_collision_files = [pfx + l.strip() for l in f if l.strip()]
    with open(asset_root + "urdf/ycb/object_offset_grasp.txt")    as f:
        object_offset          = [[float(x) for x in l.strip().split()] for l in f if l.strip()]

    asset_options.fix_base_link = False
    object_assets = [gym.load_asset(sim, asset_root, ob, asset_options)
                     for ob in object_asset_files]

    ur5e_pose   = gymapi.Transform()
    ur5e_pose.p = gymapi.Vec3(0, 0, 0)
    ur5e_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0.5 * math.pi)
    table_pose  = gymapi.Transform()
    table_pose.p = gymapi.Vec3(table_dims.x * 0.5 + 0.3, 0.0, table_dims.z * 0.5)

    gym.set_light_parameters(sim, 0, gymapi.Vec3(0.3,0.3,0.3),
                             gymapi.Vec3(1,1,1), gymapi.Vec3(-1,0,0))
    gym.set_light_parameters(sim, 1, gymapi.Vec3(0.3,0.3,0.3),
                             gymapi.Vec3(1,1,1), gymapi.Vec3(1,0,0))

    # ── Load test records ─────────────────────────────────────────────────────
    records = load_test_shards(args.test_dataset)
    if args.max_records > 0:
        records = records[:args.max_records]
    print(f"Evaluating {len(records)} records.")

    # ── Evaluation loop ───────────────────────────────────────────────────────
    all_results:    List[Dict] = []
    # Primary success (finger-midpoint XY + Z thresholded), as in evaluate_ntfield_oracle.py
    n_success       = 0
    # Diagnostic metric: EE (wrist-proxy) distance thresholded
    n_ee_success   = 0

    latent_l2_list  = []
    latent_cos_list = []

    ee_dist_list: List[float] = []
    finger_mid_to_target_xy_list: List[float] = []
    finger_mid_to_target_z_diff_list: List[float] = []
    between_fingers_success_n = 0

    with open(results_path, "w") as results_f:

        for rec_idx, record in enumerate(tqdm(records, desc="Evaluating")):
            all_object_ids       = record["all_object_ids"]
            all_object_locations = record["all_object_locations"]   # [[x,y,z], ...]
            object_name          = record["object_name"]
            z_goal_true          = record["z_goal"].float().numpy()  # (z_dim,)
            q_start              = record["q_start"].float().numpy() # (6,)
            image_tensor         = record["image"].float()           # (3, H, W)
            seed                 = record.get("seed", -1)
            source_file          = record.get("source_file", "")
            target_obj_idx       = record.get("target_obj_idx", 0)

            result: Dict[str, Any] = {
                "record_idx":    rec_idx,
                "source_file":   source_file,
                "seed":          seed,
                "object_name":   object_name,
                "target_obj_idx": target_obj_idx,

                # Primary metric (as in evaluate_ntfield_oracle.py)
                "success":       False,
                "finger_mid_success": False,

                # Diagnostic metric
                "ee_success":   False,

                "planner_stopped": None,
                "final_latent_dist": None,
                "path_len":      None,

                # Geometry metrics (sim-based, after executing the planned path)
                "ee_pos": None,
                "ee_link": None,
                "ee_dist_m": None,
                "ee_success_thresh": args.ee_success_thresh,

                "finger_mid_xy_success_thresh_m": args.finger_mid_xy_success_thresh_m,
                "finger_mid_z_success_thresh_m": args.finger_mid_z_success_thresh_m,
                "finger_midpoint_to_target_xy_distance_m": None,
                "finger_midpoint_to_target_z_diff_m": None,

                "between_fingers_success": False,
                "between_fingers_reason": None,
                "xy_axis_success": None,
                "z_height_success": None,
                "left_finger_pos": None,
                "right_finger_pos": None,
                "object_root_pos": None,
                "object_asset_id_for_height": None,
                "object_half_height_m": None,

                "finger_gap_m": None,
                "finger_axis_proj_m": None,
                "finger_axis_proj_norm": None,
                "finger_lateral_dist_xy_m": None,
                "finger_z_diff_m": None,
                "object_grasp_target_z": None,
                "finger_midpoint_z": None,

                "between_axis_margin_m": args.between_axis_margin_m,
                "between_lateral_thresh_m": args.between_lateral_thresh_m,
                "between_z_thresh_m": args.between_z_thresh_m,

                # Offline latent metrics
                "latent_error": None,
            }

            try:
                # ── Reconstruct scene ─────────────────────────────────────
                spacing   = 2
                env_lower = gymapi.Vec3(-spacing, -spacing, 0)
                env_upper = gymapi.Vec3(spacing,  spacing,  0)
                env       = gym.create_env(sim, env_lower, env_upper, 1)
                ur_handle = gym.create_actor(env, ur5e_asset, ur5e_pose, "ur5e", 0, 32767)
                gym.create_actor(env, table_asset, table_pose, "table", 0, 1)

                spj = gym.find_actor_dof_handle(env, ur_handle, "shoulder_pan_joint")
                slj = gym.find_actor_dof_handle(env, ur_handle, "shoulder_lift_joint")
                ej  = gym.find_actor_dof_handle(env, ur_handle, "elbow_joint")
                wj1 = gym.find_actor_dof_handle(env, ur_handle, "wrist_1_joint")
                wj2 = gym.find_actor_dof_handle(env, ur_handle, "wrist_2_joint")
                wj3 = gym.find_actor_dof_handle(env, ur_handle, "wrist_3_joint")

                viewer = None
                if args.use_viewer:
                    viewer = gym.create_viewer(sim, gymapi.CameraProperties())

                object_handles, object_status, object_col_lib = build_scene(
                    gym, sim, env, [ur_handle], object_assets,
                    object_collision_files, object_offset, asset_root,
                    all_object_ids, all_object_locations,
                    spj, slj, ej, wj1, wj2, wj3, viewer,
                )

                # Warm-up sim to settle physics (matches data collection)
                warmup_sim(
                    gym, sim, env, spj, slj, ej, wj1, wj2, wj3, viewer,
                    object_handles, object_status, object_col_lib,
                )

                # Get live q_start after settle
                dof_state  = gym.get_actor_dof_states(env, ur_handle, gymapi.STATE_POS)
                q_start_live = np.array(dof_state["pos"][:6], dtype=np.float64)

                # ── Student prediction ────────────────────────────────────
                z_pred = predict_z_goal(
                    student_model, student_meta,
                    image_tensor, object_name,
                    dev_student, normalize,
                )

                # ── Offline latent error (no sim needed) ──────────────────
                err = latent_errors(z_pred, z_goal_true)
                result["latent_error"] = err
                latent_l2_list.append(err["l2"])
                latent_cos_list.append(err["cos_dist"])

                # ── NTField planner ───────────────────────────────────────
                path_raw, meta_plan = ntfield_plan_latent(
                    nt_net,
                    q_start_live,
                    z_pred,
                    step_size = args.ntfield_step_size,
                    max_steps = args.ntfield_max_steps,
                    tol       = args.ntfield_tol,
                    device    = ntfield_device_str,
                )

                # Planner stats from latent-space optimization
                result["planner_stopped"]    = meta_plan.get("stopped")
                result["final_latent_dist"]  = meta_plan.get("final_latent_dist")
                result["path_len"]           = meta_plan.get("path_len")

                # ── Execute planned joint path + measure geometry ────────────
                ee_pos, ee_link = _execute_path_and_get_ee(
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
                    path_raw,
                )

                # Live target object root pose (source of truth in sim)
                mesh_idx = min(target_obj_idx, len(object_handles) - 1)
                st_obj = gym.get_actor_rigid_body_states(
                    env, object_handles[mesh_idx], gymapi.STATE_POS
                )
                T_root = gymapi.Transform.from_buffer(st_obj["pose"][0])
                obj_root_xyz = np.array(
                    [T_root.p.x, T_root.p.y, T_root.p.z],
                    dtype=np.float64,
                )

                # EE distance proxy -> success (diagnostic)
                ee_dist_m = float(
                    np.linalg.norm(np.asarray(ee_pos, dtype=np.float64) - obj_root_xyz)
                )
                ee_success = bool(ee_dist_m <= float(args.ee_success_thresh))

                # Finger positions -> finger-midpoint diffs
                left_finger_pos = _get_rigid_body_world_pos(
                    gym, env, ur_handle, "left_inner_finger"
                )
                right_finger_pos = _get_rigid_body_world_pos(
                    gym, env, ur_handle, "right_inner_finger"
                )

                finger_midpoint_to_target_xy_distance_m = None
                finger_midpoint_to_target_z_diff_m = None
                finger_mid_success = False

                between_fingers_success = False
                between_fingers_reason: Optional[str] = None
                between_meta: Dict[str, Any] = {}
                xy_ok: Optional[bool] = None
                z_ok: Optional[bool] = None

                # Height offset for between-fingers Z checks
                oid_ix = min(target_obj_idx, len(all_object_ids) - 1)
                asset_id = int(all_object_ids[oid_ix])
                half_height_m = float(
                    OBJECT_HEIGHTS_M.get(asset_id, OBJECT_HEIGHT_DEFAULT_M) * 0.5
                )

                if left_finger_pos is not None and right_finger_pos is not None:
                    finger_mid = 0.5 * (left_finger_pos + right_finger_pos)

                    dxy = finger_mid[:2] - obj_root_xyz[:2]
                    finger_midpoint_to_target_xy_distance_m = float(
                        np.hypot(float(dxy[0]), float(dxy[1]))
                    )
                    finger_midpoint_to_target_z_diff_m = float(
                        finger_mid[2] - obj_root_xyz[2]
                    )

                    if (
                        finger_midpoint_to_target_xy_distance_m is not None
                        and finger_midpoint_to_target_z_diff_m is not None
                    ):
                        finger_mid_success = (
                            finger_midpoint_to_target_xy_distance_m
                            <= float(args.finger_mid_xy_success_thresh_m)
                            and abs(finger_midpoint_to_target_z_diff_m)
                            <= float(args.finger_mid_z_success_thresh_m)
                        )

                    # Between-fingers alignment success + meta
                    between_fingers_success, between_fingers_reason, between_meta = (
                        _between_fingers_xy_z(
                            left_finger_pos,
                            right_finger_pos,
                            obj_root_xyz,
                            axis_margin_m=args.between_axis_margin_m,
                            lateral_thresh_m=args.between_lateral_thresh_m,
                            z_thresh_m=args.between_z_thresh_m,
                            obj_half_height_m=half_height_m,
                        )
                    )
                    xy_ok = bool(between_meta.get("xy_axis_success", False))
                    z_ok = bool(between_meta.get("z_height_success", False))
                else:
                    between_fingers_reason = "finger_links_missing"
                    xy_ok, z_ok = None, None

                # Update result record (same keys as evaluate_ntfield_oracle)
                result.update(
                    {
                        "success": finger_mid_success,
                        "finger_mid_success": finger_mid_success,
                        "ee_success": ee_success,
                        "ee_pos": np.asarray(ee_pos, dtype=np.float64).tolist(),
                        "ee_link": ee_link,
                        "ee_dist_m": ee_dist_m,
                        "ee_success_thresh": args.ee_success_thresh,
                        "finger_mid_xy_success_thresh_m": args.finger_mid_xy_success_thresh_m,
                        "finger_mid_z_success_thresh_m": args.finger_mid_z_success_thresh_m,
                        "finger_midpoint_to_target_xy_distance_m": finger_midpoint_to_target_xy_distance_m,
                        "finger_midpoint_to_target_z_diff_m": finger_midpoint_to_target_z_diff_m,
                        "between_fingers_success": between_fingers_success,
                        "between_fingers_reason": between_fingers_reason,
                        "xy_axis_success": xy_ok,
                        "z_height_success": z_ok,
                        "left_finger_pos": (
                            None if left_finger_pos is None else left_finger_pos.tolist()
                        ),
                        "right_finger_pos": (
                            None if right_finger_pos is None else right_finger_pos.tolist()
                        ),
                        "object_root_pos": obj_root_xyz.tolist(),
                        "object_asset_id_for_height": asset_id,
                        "object_half_height_m": half_height_m,
                        "finger_gap_m": between_meta.get("finger_gap_m"),
                        "finger_axis_proj_m": between_meta.get("finger_axis_proj_m"),
                        "finger_axis_proj_norm": between_meta.get("finger_axis_proj_norm"),
                        "finger_lateral_dist_xy_m": between_meta.get("finger_lateral_dist_xy_m"),
                        "finger_z_diff_m": between_meta.get("finger_z_diff_m"),
                        "object_grasp_target_z": between_meta.get(
                            "object_grasp_target_z"
                        ),
                        "finger_midpoint_z": between_meta.get("finger_midpoint_z"),
                        "between_axis_margin_m": args.between_axis_margin_m,
                        "between_lateral_thresh_m": args.between_lateral_thresh_m,
                        "between_z_thresh_m": args.between_z_thresh_m,
                    }
                )

                if finger_mid_success:
                    n_success += 1
                if ee_success:
                    n_ee_success += 1

                ee_dist_list.append(ee_dist_m)
                if finger_midpoint_to_target_xy_distance_m is not None:
                    finger_mid_to_target_xy_list.append(
                        finger_midpoint_to_target_xy_distance_m
                    )
                if finger_midpoint_to_target_z_diff_m is not None:
                    finger_mid_to_target_z_diff_list.append(
                        finger_midpoint_to_target_z_diff_m
                    )
                if between_fingers_success:
                    between_fingers_success_n += 1

            except Exception as e:
                result["error"] = str(e)
                tqdm.write(f"  [ERROR] record {rec_idx} ({source_file}): {e}")

            all_results.append(result)
            results_f.write(json.dumps(result) + "\n")
            results_f.flush()

            latent_l2_str = "N/A"
            if result.get("latent_error") and isinstance(result["latent_error"], dict):
                latent_l2_str = f"{float(result['latent_error'].get('l2')):.4f}"

            tqdm.write(
                f"  [{rec_idx:04d}] {object_name:20s} | "
                f"success={result['success']} | "
                f"latent_dist={result['final_latent_dist'] or 'N/A':>8} | "
                f"latent_l2={latent_l2_str}"
            )

    # ── Aggregate summary ─────────────────────────────────────────────────────
    n_total    = len(all_results)
    n_errors   = sum(1 for r in all_results if "error" in r)
    n_evaluated = n_total - n_errors

    summary = {
        "mode":                 "test_dataset_eval",
        "checkpoint":          args.latent_checkpoint,
        "model_type":          student_meta["model_type"],
        "ntfield_checkpoint":  args.ntfield_checkpoint,
        "test_dataset":        args.test_dataset,

        "n_total":             n_total,
        "n_evaluated":         n_evaluated,
        "n_errors":            n_errors,

        # Midpoint-based grasp success (primary)
        "n_success":           n_success,
        "success_rate":        n_success / max(n_evaluated, 1),

        # EE diagnostic (wrist proxy)
        "n_ee_success":        n_ee_success,
        "ee_success_rate":     n_ee_success / max(n_evaluated, 1),

        # EE-to-target distance stats (diagnostic)
        "ee_dist_mean_m":     float(np.mean(ee_dist_list))   if ee_dist_list else None,
        "ee_dist_median_m":   float(np.median(ee_dist_list)) if ee_dist_list else None,
        "ee_dist_std_m":      float(np.std(ee_dist_list))    if ee_dist_list else None,

        # Finger-midpoint-to-target stats (primary geometry)
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

        # Between-fingers alignment
        "n_between_fingers_success": between_fingers_success_n,
        "between_fingers_success_rate": (
            between_fingers_success_n / max(n_evaluated, 1)
        ),

        # Final latent dist / path length (from NTField)
        "final_latent_dist_mean": float(
            np.mean([r["final_latent_dist"] for r in all_results if isinstance(r.get("final_latent_dist"), (int, float))])
        ) if any(isinstance(r.get("final_latent_dist"), (int, float)) for r in all_results) else None,
        "path_len_mean": (
            float(
                np.mean([r["path_len"] for r in all_results if isinstance(r.get("path_len"), (int, float))])
            )
            if any(isinstance(r.get("path_len"), (int, float)) for r in all_results) else None
        ),

        # Latent-space metrics (offline)
        "latent_l2_mean":      float(np.mean(latent_l2_list))  if latent_l2_list  else None,
        "latent_l2_std":       float(np.std(latent_l2_list))   if latent_l2_list  else None,
        "latent_l2_median":    float(np.median(latent_l2_list)) if latent_l2_list else None,
        "latent_cos_dist_mean": float(np.mean(latent_cos_list)) if latent_cos_list else None,

        # Planner convergence breakdown
        "stop_reason_counts":  {},
    }

    stop_counts: Dict[str, int] = {}
    for r in all_results:
        reason = r.get("planner_stopped") or "error"
        stop_counts[reason] = stop_counts.get(reason, 0) + 1
    summary["stop_reason_counts"] = stop_counts

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60)
    print(f"Results       : {results_path}")
    print(f"Summary       : {summary_path}")
    print(f"Total records : {n_total}")
    print(f"Evaluated     : {n_evaluated}  (errors: {n_errors})")
    print(
        f"Success (fm_xy & |fm_dz|) : {n_success} / {n_evaluated}  "
        f"({100.0 * summary['success_rate']:.1f}%)"
    )
    print(
        f"EE success (wrist proxy)    : {n_ee_success} / {n_evaluated}  "
        f"({100.0 * summary['ee_success_rate']:.1f}%)"
    )
    if finger_mid_to_target_xy_list:
        print(
            "Finger-mid → target (XY)  : "
            f"mean={summary['finger_midpoint_to_target_xy_distance_mean_m']:.4f}  "
            f"median={summary['finger_midpoint_to_target_xy_distance_median_m']:.4f}  "
            f"std={summary['finger_midpoint_to_target_xy_distance_std_m']:.4f}"
        )
    if finger_mid_to_target_z_diff_list:
        print(
            "Finger-mid → target (Z)   : "
            f"mean={summary['finger_midpoint_to_target_z_diff_mean_m']:+.4f}  "
            f"median={summary['finger_midpoint_to_target_z_diff_median_m']:+.4f}  "
            f"std={summary['finger_midpoint_to_target_z_diff_std_m']:.4f}"
        )
    if latent_l2_list:
        print(f"Latent L2     : mean={summary['latent_l2_mean']:.4f}  "
              f"std={summary['latent_l2_std']:.4f}  "
              f"median={summary['latent_l2_median']:.4f}")
        print(f"Latent CosDist: mean={summary['latent_cos_dist_mean']:.4f}")
    print(f"Stop reasons  : {stop_counts}")
    print("="*60)

    os.chdir(_cwd_prev)
    if not args.no_isaac_hard_exit:
        os._exit(0)


if __name__ == "__main__":
    main()