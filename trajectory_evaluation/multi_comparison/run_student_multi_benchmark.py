from __future__ import annotations

import argparse
import json
import math
import os
import re
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
_STUDENT_DIR = os.path.join(_PI_VLA_ROOT, "student_model_training")
file_dir = os.path.join(HANWEN_GRASPING_ROOT, "collect_data")
util_dir = os.path.join(file_dir, "util")
grasp_util_dir = os.path.join(file_dir, "grasp_util")
sys.path.insert(0, HANWEN_GRASPING_ROOT)
sys.path.append(util_dir)
sys.path.append(grasp_util_dir)
sys.path.insert(0, _PI_VLA_ROOT)
sys.path.insert(0, os.path.join(_PI_VLA_ROOT, "ntrl-demo"))
sys.path.insert(0, _STUDENT_DIR)

import torch
from torchvision import transforms
import robot_arm_configuration as RC
from stl_reader import stl_reader
from obj_reader import obj_reader
from trajectory_evaluation.ntfield.eval_trajectory_ntfield import load_network_and_function
from planning.gradient_planner_trajectory import SCALE as NTFIELD_SCALE

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


def _tokenize_prompt(text: str) -> list[str]:
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
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_type = ckpt.get("model_type", "mdn")
    z_dim = ckpt.get("z_dim", 256)
    vocab_size = ckpt.get("vocab_size", None)
    token_to_id = ckpt.get("token_to_id", None)
    max_prompt_len = int(ckpt.get("max_prompt_len", 8))
    n_components = int(ckpt.get("n_components", 8))

    if model_type == "regression":
        from student_model_regression import RegressionStudent

        model = RegressionStudent(output_dim=z_dim).to(device)
    else:
        from student_model_mdn import MDNStudent

        model = MDNStudent(
            output_dim=z_dim,
            vocab_size=max(int(vocab_size or 2), 2),
            n_components=n_components,
        ).to(device)

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    meta = {
        "model_type": model_type,
        "token_to_id": token_to_id,
        "max_prompt_len": max_prompt_len,
        "latent_dim": int(ckpt.get("latent_dim", 64)),
        "z_dim": int(z_dim),
        "n_components": int(n_components),
    }
    return model, meta


@torch.no_grad()
def predict_student_latent(
    model,
    meta,
    image_uint8: np.ndarray,
    object_name: str,
    device: torch.device,
    num_samples: int = 30,
) -> np.ndarray:
    img = image_uint8
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.shape[-1] == 4:
        img = img[..., :3]
    x = transforms.ToTensor()(img)
    x = transforms.Resize((224, 224))(x)
    x = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )(x).unsqueeze(0).to(device)

    if meta["model_type"] == "regression":
        z_pred = model(x)
        return z_pred.squeeze(0).cpu().numpy()

    token_to_id = meta["token_to_id"]
    max_prompt_len = meta["max_prompt_len"]
    if token_to_id:
        prompt = f"grasp {object_name.strip().lower()}"
        text_tokens = _encode_prompts([prompt], token_to_id, max_prompt_len).to(device)
    else:
        text_tokens = torch.zeros((1, max_prompt_len), dtype=torch.long, device=device)

    # Match integrated pipeline: deterministic highest-weight MDN component.
    if hasattr(model, "predict_best"):
        best_pred = model.predict_best(x, text_tokens)
        return best_pred.squeeze(0).cpu().numpy()

    preds = model.get_multiple_latent_predictions(x, text_tokens, num_samples=num_samples)
    mean_pred = preds.mean(dim=0)
    dists = ((preds - mean_pred) ** 2).sum(dim=-1)
    best_idx = dists.argmin(dim=0)
    best_pred = preds[best_idx, torch.arange(1)]
    return best_pred.squeeze(0).cpu().numpy()


def ntfield_plan_with_goal_latent(
    teacher_network,
    q_start: np.ndarray,
    z_goal_hat: np.ndarray,
    step_size: float = 0.02,
    max_steps: int = 200,
    tol: float = 0.01,
    device: str = "cuda",
    delta_clamp_rad: float = 0.0,
    refine_max_steps: int = 0,
    refine_step_size: float = 0.01,
    refine_delta_clamp_rad: float | None = None,
    stagnate_max_steps: int = 0,
    stagnate_patience: int = 30,
    stagnate_rel_eps: float = 5e-4,
    stagnate_step_size: float = 0.005,
):
    q_start = np.asarray(q_start, dtype=np.float32).reshape(-1)
    q_curr_norm = q_start / NTFIELD_SCALE
    q_curr_t = torch.tensor(q_curr_norm, dtype=torch.float32, device=device).unsqueeze(0)

    if isinstance(z_goal_hat, np.ndarray):
        z_goal_hat = torch.tensor(z_goal_hat.reshape(1, -1), dtype=torch.float32, device=device)
    else:
        z_goal_hat = z_goal_hat.reshape(1, -1).to(device)

    def _latent_dist(qn, zg):
        with torch.no_grad():
            d, _, _ = teacher_network.out_with_goal_latent(qn.detach(), zg)
            return float(d.item())

    def _grad_step(q_t, zg, stp, clamp_rad):
        q_t = q_t.detach().requires_grad_(True)
        dist, _, coords_out = teacher_network.out_with_goal_latent(q_t, zg)
        d_pre = float(dist.item())
        if d_pre < tol:
            return q_t.detach(), d_pre, True
        grad_out = torch.autograd.grad(dist, coords_out)[0]
        grad_start = grad_out[:, :6]
        with torch.no_grad():
            delta = -stp * grad_start
            if clamp_rad > 0.0:
                cap_norm = float(clamp_rad) / float(NTFIELD_SCALE)
                peak = torch.amax(torch.abs(delta))
                if float(peak.item()) > cap_norm + 1e-12:
                    delta = delta * (cap_norm / (peak + 1e-12))
            q_next = q_t + delta
        d_post = _latent_dist(q_next, zg)
        return q_next.detach(), d_post, d_post < tol

    path_norm = [q_curr_norm.copy()]
    final_dist = None
    converged = False
    stopped = "max_main_steps"
    main_clamp = float(delta_clamp_rad) if delta_clamp_rad > 0.0 else 0.0

    for _ in range(max_steps):
        q_curr_t, final_dist, converged = _grad_step(q_curr_t, z_goal_hat, step_size, main_clamp)
        path_norm.append(q_curr_t.detach().cpu().numpy()[0].copy())
        if converged:
            stopped = "latent_tol_main"
            break

    refine_clamp = (
        float(refine_delta_clamp_rad) if refine_delta_clamp_rad and refine_delta_clamp_rad > 0.0 else 0.0
    )
    if refine_max_steps > 0 and not converged:
        for _ in range(refine_max_steps):
            q_curr_t, final_dist, converged = _grad_step(q_curr_t, z_goal_hat, refine_step_size, refine_clamp)
            path_norm.append(q_curr_t.detach().cpu().numpy()[0].copy())
            if converged:
                stopped = "latent_tol_refine"
                break
        else:
            if not converged:
                stopped = "max_refine_steps"

    if stagnate_max_steps > 0 and not converged:
        best_d = final_dist if final_dist is not None else _latent_dist(q_curr_t, z_goal_hat)
        stall = 0
        for _ in range(stagnate_max_steps):
            q_curr_t, final_dist, converged = _grad_step(q_curr_t, z_goal_hat, stagnate_step_size, 0.0)
            path_norm.append(q_curr_t.detach().cpu().numpy()[0].copy())
            if converged:
                stopped = "latent_tol_stagnate"
                break
            imp = (best_d - final_dist) / max(best_d, 1e-8)
            if imp > stagnate_rel_eps:
                best_d = min(best_d, final_dist)
                stall = 0
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
    return [p * NTFIELD_SCALE for p in path_norm], {"final_latent_dist": final_dist, "stopped": stopped}


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
    parser.add_argument("--student_checkpoint", type=str, required=True, help="Student latent model checkpoint (.pth)")
    parser.add_argument("--latent_device", type=str, default="auto", help="'auto', 'cpu', or 'cuda'.")
    parser.add_argument("--student_num_samples", type=int, default=30, help="MDN latent sampling count.")
    parser.add_argument("--object_name", type=str, default=None, help="Optional prompt object name for student model.")
    parser.add_argument("--ntfield_experiment_dir", type=str, default=None)
    parser.add_argument("--use_viewer", action="store_true")
    parser.add_argument("--ntfield_device", type=str, default="cuda:0")
    parser.add_argument("--ntfield_step_size", type=float, default=0.02)
    parser.add_argument("--ntfield_max_steps", type=int, default=200)
    parser.add_argument("--ntfield_tol", type=float, default=0.01)
    parser.add_argument("--ntfield_delta_clamp_rad", type=float, default=0.0)
    parser.add_argument("--ntfield_refine_max_steps", type=int, default=-1)
    parser.add_argument("--ntfield_refine_step_size", type=float, default=None)
    parser.add_argument("--ntfield_refine_step_size_factor", type=float, default=None)
    parser.add_argument("--ntfield_refine_delta_clamp_rad", type=float, default=None)
    parser.add_argument("--ntfield_stagnate_max_steps", type=int, default=400)
    parser.add_argument("--ntfield_stagnate_patience", type=int, default=30)
    parser.add_argument("--ntfield_stagnate_rel_eps", type=float, default=5e-4)
    parser.add_argument("--ntfield_stagnate_step_size", type=float, default=None)
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
    ntfield_network, _ = load_network_and_function(ckpt_abs, args.ntfield_experiment_dir, dev, dim=6)
    ntfield_device_str = str(dev) if dev.type == "cuda" else "cpu"
    student_device = torch.device(
        "cuda" if torch.cuda.is_available() and args.latent_device in ("auto", "cuda") else "cpu"
    )
    student_ckpt_abs = _resolve_pi_vla_checkpoint(args.student_checkpoint)
    if not os.path.isfile(student_ckpt_abs):
        print(f"Student checkpoint not found: {student_ckpt_abs}")
        sys.exit(1)
    student_model, student_meta = load_student_model(student_ckpt_abs, student_device)

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
    top_cam_handle = None
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
        # Exact top-view camera placement from integrated pipeline.
        top_cam_handle = gym.create_camera_sensor(envs[-1], camera_props)
        top_cam_pos = gymapi.Vec3(table_pose.p.x, table_pose.p.y + 0.001, 2.2)
        top_cam_target = gymapi.Vec3(table_pose.p.x - 0.5, table_pose.p.y, table_pose.p.z)
        gym.set_camera_location(top_cam_handle, envs[-1], top_cam_pos, top_cam_target)

    if viewer is not None:
        cam_pos = gymapi.Vec3(2.2, 0, 0.5)
        cam_target = gymapi.Vec3(0, 0, 0.5)
        gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    # Match lighting from final_integrate/run_integrated_pipeline_latent_multi_obj_mdn.py.
    gym.set_light_parameters(
        sim,
        0,
        gymapi.Vec3(0.3, 0.3, 0.3),
        gymapi.Vec3(1.0, 1.0, 1.0),
        gymapi.Vec3(-1.0, 0.0, 0.0),
    )
    gym.set_light_parameters(
        sim,
        1,
        gymapi.Vec3(0.3, 0.3, 0.3),
        gymapi.Vec3(1.0, 1.0, 1.0),
        gymapi.Vec3(1.0, 0.0, 0.0),
    )

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

    # Match integrated student pipeline: move to HOME_DOF and settle
    # before taking the camera image and capturing q_start.
    _HOME_DOF = [0.7, -2.0, 2.5, -0.3, 0.7, 0.0]
    _START_SETTLE = 30
    for _ in range(_START_SETTLE):
        gym.set_dof_target_position(env, spj, _HOME_DOF[0])
        gym.set_dof_target_position(env, slj, _HOME_DOF[1])
        gym.set_dof_target_position(env, ej, _HOME_DOF[2])
        gym.set_dof_target_position(env, wj1, _HOME_DOF[3])
        gym.set_dof_target_position(env, wj2, _HOME_DOF[4])
        gym.set_dof_target_position(env, wj3, _HOME_DOF[5])
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        if viewer is not None:
            gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

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
        "goal_configuration_grasp_verify": None,
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
        "student_ntfield": {},
        "ntfield_checkpoint": ckpt_abs,
        "student_checkpoint": student_ckpt_abs,
    }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.abspath(args.record_dir) if args.record_dir else os.path.join(_PI_VLA_ROOT, "output", "trajectory_evaluation", f"multi_benchmark_{stamp}")
    os.makedirs(session_dir, exist_ok=True)
    mp4_stu = os.path.join(session_dir, "student_ntfield.mp4")
    result["video_session_dir"] = session_dir
    want_video = not args.no_video

    object_name = args.object_name
    if not object_name:
        asset_dir_name = os.path.basename(os.path.dirname(object_asset_files[int(target_file_idx[target_idx])]))
        object_name = re.sub(r"^\d+_", "", asset_dir_name).replace("_", " ")

    gym.render_all_camera_sensors(sim)
    assert top_cam_handle is not None
    raw_top = gym.get_camera_image(sim, env, top_cam_handle, gymapi.IMAGE_COLOR)
    rgba_top = raw_top.reshape(camera_props.height, camera_props.width, 4)
    rgb_main = rgba_top[..., :3].copy()
    top_image_path = os.path.join(session_dir, "top_view.png")
    plt.imsave(top_image_path, rgb_main)
    result["top_view_image_path"] = top_image_path

    z_goal_hat = predict_student_latent(
        student_model,
        student_meta,
        rgb_main,
        object_name=object_name,
        device=student_device,
        num_samples=args.student_num_samples,
    )
    refine_max = args.ntfield_refine_max_steps
    if refine_max < 0:
        refine_max = 100 if args.ntfield_delta_clamp_rad > 0.0 else 0
    if args.ntfield_refine_step_size is not None:
        refine_step = float(args.ntfield_refine_step_size)
    elif args.ntfield_refine_step_size_factor is not None:
        refine_step = float(args.ntfield_step_size) * float(args.ntfield_refine_step_size_factor)
    else:
        refine_step = float(args.ntfield_step_size) * 0.5
    refine_delta_clamp_rad = None
    if args.ntfield_refine_delta_clamp_rad is not None and float(args.ntfield_refine_delta_clamp_rad) > 0.0:
        refine_delta_clamp_rad = float(args.ntfield_refine_delta_clamp_rad)
    stagnate_step = (
        float(args.ntfield_stagnate_step_size)
        if args.ntfield_stagnate_step_size is not None
        else float(args.ntfield_step_size) * 0.25
    )

    t_stu0 = time.perf_counter()
    path_stu_raw, meta_stu = ntfield_plan_with_goal_latent(
        ntfield_network,
        q_start_live,
        z_goal_hat,
        step_size=args.ntfield_step_size,
        max_steps=args.ntfield_max_steps,
        tol=args.ntfield_tol,
        device=ntfield_device_str,
        delta_clamp_rad=args.ntfield_delta_clamp_rad,
        refine_max_steps=refine_max,
        refine_step_size=refine_step,
        refine_delta_clamp_rad=refine_delta_clamp_rad,
        stagnate_max_steps=int(args.ntfield_stagnate_max_steps),
        stagnate_patience=int(args.ntfield_stagnate_patience),
        stagnate_rel_eps=float(args.ntfield_stagnate_rel_eps),
        stagnate_step_size=stagnate_step,
    )
    student_planning_wall_s = float(time.perf_counter() - t_stu0)

    path_stu = _path_as_6_list(path_stu_raw) if path_stu_raw else []
    student_goal_q = np.asarray(path_stu[-1], dtype=np.float64).reshape(6) if path_stu else None
    result["goal_configuration_grasp_verify"] = student_goal_q.tolist() if student_goal_q is not None else None

    selected_grasp_check = _evaluate_ee_proxy_against_scene(
        rac,
        student_goal_q if student_goal_q is not None else q_start_live,
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

    student_has_path = bool(path_stu and len(path_stu) >= 2)
    if args.ntfield_waypoint_mode == "two_point" and len(path_stu) >= 2:
        path_stu = [path_stu[0], path_stu[-1]]
    elif args.ntfield_fixed_waypoints > 0 and len(path_stu) >= 2:
        path_stu = _resample_path_fixed_waypoints(path_stu, args.ntfield_fixed_waypoints)

    stu_terminal_check = planner_terminal_ee_check(
        rac,
        path_stu,
        flex_collision_models,
        target_idx=target_idx,
        spawned_object_asset_indices=target_file_idx.tolist(),
        center_mode=args.ee_proxy_center_mode,
        max_radius_m=args.ee_proxy_max_radius_m,
    )
    result["student_ntfield"] = {
        "planning_wall_s": student_planning_wall_s,
        "success": student_has_path,
        "final_latent_dist": meta_stu.get("final_latent_dist"),
        "planner_stopped": meta_stu.get("stopped"),
        "num_waypoints_after_postprocess": len(path_stu),
        "trajectory_waypoints_rad": path_stu if student_has_path else None,
        "ee_object_pair_collision_terminal_waypoint": stu_terminal_check,
    }
    if student_has_path:
        result["student_ntfield"]["motion"] = joint_metrics(path_stu, q_start_live, path_stu[-1])
    if args.require_ee_object_collision and not stu_terminal_check["success"]:
        result["student_ntfield"]["success"] = False
        result["student_ntfield"]["ee_object_postcheck_failed"] = True

    frames_stu = [] if want_video else None
    exec_stu = execute_path_and_time(
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
        path_stu,
        "student_ntfield",
        main_cam_handle=main_cam_handle,
        camera_props=camera_props,
        record_rgb=frames_stu,
        planner_playback=args.planner_playback,
    )
    result["student_ntfield"]["execution"] = exec_stu
    if want_video and frames_stu:
        _save_mp4_rgb(frames_stu, mp4_stu, fps=args.video_fps)
        result["student_ntfield"]["video_path"] = mp4_stu
    elif want_video:
        result["student_ntfield"]["video_path"] = None

    if args.save_final_geometric_debug and student_goal_q is not None and len(object_mesh) > 0:
        debug_png = (
            os.path.abspath(args.final_geometric_debug_path)
            if args.final_geometric_debug_path
            else os.path.join(session_dir, "final_geometric_debug.png")
        )
        ee_center_for_viz = result["ee_object_pair_collision_selected_grasp"].get("ee_center_world_m")
        ee_radius_for_viz = result["ee_object_pair_collision_proxy"].get("radius_m")
        if ee_center_for_viz is None:
            ee_center_for_viz = list(
                np.asarray(rac.calculate_transform_from_angles(student_goal_q)[8][0], dtype=np.float64).reshape(3)
            )
        save_final_geometric_debug_image_multi(
            out_path=debug_png,
            rac=rac,
            dof_result=student_goal_q,
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
