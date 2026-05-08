#!/usr/bin/env python3
"""
End-to-end PI-VLA integration (Isaac Gym + NTField + latent goal only).

Supports both MDNStudent and CVAEStudent checkpoints — model type is detected
automatically from the checkpoint's "model_type" key:
  - "mdn"  → MDNStudent.predict_best() for deterministic inference
  - "cvae" → CVAEStudent forward with z_goal=None (prior mean)
  - absent → falls back to CVAEStudent for backward compatibility

Run from PI-VLA root::
  python final_integrate/run_integrated_pipeline_latent_multi_obj_mdn.py \
    --ntfield_checkpoint teacher_model.pt \
    --latent_checkpoint /home/hojinsohn/VLM-NT/PI-VLA/student_model_training/best_z_goal_model_mdn_mdn_K8_bs256_lr3em4_ep60_20260505_110118.pth
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import sys
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_PI_VLA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HANWEN_GRASPING_ROOT = os.path.join(_PI_VLA_ROOT, "hanwen_grasping")
_COLLECT_DATA_DIR = os.path.join(HANWEN_GRASPING_ROOT, "collect_data")
_UTIL_DIR = os.path.join(_COLLECT_DATA_DIR, "util")
_HANWEN_UTIL_DIR = os.path.join(HANWEN_GRASPING_ROOT, "util")
_GRASP_UTIL_DIR = os.path.join(_COLLECT_DATA_DIR, "grasp_util")
_NTRL_DEMO = os.path.join(_PI_VLA_ROOT, "ntrl-demo")
_STUDENT_TRAINING_DIR = os.path.join(_PI_VLA_ROOT, "student_model_training")

for _p in (
    HANWEN_GRASPING_ROOT,
    _HANWEN_UTIL_DIR,
    _UTIL_DIR,
    _GRASP_UTIL_DIR,
    _PI_VLA_ROOT,
    _NTRL_DEMO,
    _STUDENT_TRAINING_DIR,
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TABLE_DIMS_X = 0.8
TABLE_DIMS_Y = 1.0
TABLE_DIMS_Z = 0.10
_SIM_DT = 1.0 / 60.0


def _resolve_under_root(path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(_PI_VLA_ROOT, path))


def _args_namespace_to_jsonable(ns: argparse.Namespace) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in vars(ns).items():
        try:
            json.dumps(v)
            out[k] = v
        except TypeError:
            out[k] = repr(v)
    return out


def _write_batch_summary(
    session_dir: str,
    trial_rows: List[Dict[str, Any]],
    args_ns: argparse.Namespace,
) -> Dict[str, Any]:
    n = len(trial_rows)
    ee_dists = [
        float(r["ee_to_target_distance_m"])
        for r in trial_rows
        if isinstance(r.get("ee_to_target_distance_m"), (int, float))
        and r["ee_to_target_distance_m"] == r["ee_to_target_distance_m"]
    ]
    finger_mid_xy_dists = [
        float(r["finger_midpoint_to_target_xy_distance_m"])
        for r in trial_rows
        if isinstance(r.get("finger_midpoint_to_target_xy_distance_m"), (int, float))
        and r["finger_midpoint_to_target_xy_distance_m"]
        == r["finger_midpoint_to_target_xy_distance_m"]
    ]
    finger_mid_z_diffs = [
        float(r["finger_midpoint_to_target_z_diff_m"])
        for r in trial_rows
        if isinstance(r.get("finger_midpoint_to_target_z_diff_m"), (int, float))
        and r["finger_midpoint_to_target_z_diff_m"]
        == r["finger_midpoint_to_target_z_diff_m"]
    ]
    err_n = sum(1 for r in trial_rows if r.get("error") is not None)
    aggregate: Dict[str, Any] = {
        "batch_session_dir": session_dir,
        "num_trials": n,
        "error_count": err_n,
        "ee_to_target_distance_m_summary": {
            "min": min(ee_dists) if ee_dists else None,
            "max": max(ee_dists) if ee_dists else None,
            "mean": float(sum(ee_dists) / len(ee_dists)) if ee_dists else None,
        },
        "finger_midpoint_to_target_xy_distance_m_summary": {
            "min": min(finger_mid_xy_dists) if finger_mid_xy_dists else None,
            "max": max(finger_mid_xy_dists) if finger_mid_xy_dists else None,
            "mean": float(sum(finger_mid_xy_dists) / len(finger_mid_xy_dists))
            if finger_mid_xy_dists else None,
        },
        "finger_midpoint_to_target_z_diff_m_summary": {
            "min": min(finger_mid_z_diffs) if finger_mid_z_diffs else None,
            "max": max(finger_mid_z_diffs) if finger_mid_z_diffs else None,
            "mean": float(sum(finger_mid_z_diffs) / len(finger_mid_z_diffs))
            if finger_mid_z_diffs else None,
        },
        "trials": trial_rows,
        "cli_args": _args_namespace_to_jsonable(args_ns),
    }
    out_path = os.path.join(session_dir, "batch_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)
    return aggregate


def _process_img(img: np.ndarray, img_size: int):
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


def _tokenize_prompt(text: str) -> list:
    return re.findall(r"[a-z0-9]+", text.lower())


def _encode_prompts(prompts: list, token_to_id: dict, max_len: int):
    import torch
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


# ---------------------------------------------------------------------------
# Unified inference — detects MDN vs CVAE from checkpoint metadata
# ---------------------------------------------------------------------------

def _infer_latent_on_image(
    image: np.ndarray,
    checkpoint_path: str,
    device: str,
    object_name: str = "object",
) -> np.ndarray:
    """
    Run image → latent inference. Automatically selects MDNStudent or
    CVAEStudent based on the "model_type" key in the checkpoint:
      - "mdn"        → MDNStudent.predict_best()  (deterministic, highest-weight component)
      - "cvae" / absent → CVAEStudent forward with z_goal=None (prior mean at inference)

    Parameters
    ----------
    image       : RGB image (H×W×3, uint8)
    checkpoint_path : path to .pth checkpoint saved by training script
    device      : "auto", "cpu", or "cuda:N"
    object_name : used to build the text prompt "grasp <object_name>"
    """
    import torch
    from torchvision import transforms

    dev = torch.device(
        "cuda" if torch.cuda.is_available() and device == "auto" else device
    )
    ckpt = torch.load(os.path.abspath(checkpoint_path), map_location=dev)

    model_type: str   = str(ckpt.get("model_type", "cvae")).lower().strip()
    z_dim: int        = int(ckpt.get("z_dim", 256))
    vocab_size        = ckpt.get("vocab_size", None)
    token_to_id       = ckpt.get("token_to_id", None)
    max_prompt_len    = int(ckpt.get("max_prompt_len", 8))

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    x = _process_img(image, 224).unsqueeze(0)
    x = normalize(x).to(dev)

    if token_to_id:
        prompt      = f"grasp {object_name.strip().lower()}"
        text_tokens = _encode_prompts([prompt], token_to_id, max_prompt_len).to(dev)
    else:
        text_tokens = torch.zeros((1, max_prompt_len), dtype=torch.long, device=dev)

    # ── MDN path ─────────────────────────────────────────────────────────────
    if model_type == "mdn":
        from student_model_mdn import MDNStudent

        n_components = int(ckpt.get("n_components", 8))
        model = MDNStudent(
            output_dim=z_dim,
            vocab_size=max(int(vocab_size or 0), 2),
            n_components=n_components,
        ).to(dev)
        try:
            model.load_state_dict(ckpt["model_state_dict"], strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                "Strict checkpoint load failed for MDNStudent. "
                "Architecture mismatch — check n_components and output_dim."
            ) from exc
        model.eval()
        print(
            f"[latent] MDNStudent loaded: z_dim={z_dim} n_components={n_components}",
            flush=True,
        )
        with torch.no_grad():
            # predict_best returns the mean of the highest-weight component —
            # deterministic and stable for NTField gradient planning.
            pred = model.predict_best(x, text_tokens)
        return pred.squeeze(0).cpu().numpy()

    # ── CVAE path (default / backward compat) ────────────────────────────────
    from student_model_cvae import CVAEStudent

    latent_dim = int(ckpt.get("latent_dim", 64))
    model = CVAEStudent(
        output_dim=z_dim,
        vocab_size=max(int(vocab_size or 0), 2),
        latent_dim=latent_dim,
    ).to(dev)
    try:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Strict checkpoint load failed for CVAEStudent. "
            "Architecture mismatch detected."
        ) from exc
    model.eval()
    print(
        f"[latent] CVAEStudent loaded: z_dim={z_dim} latent_dim={latent_dim}",
        flush=True,
    )
    with torch.no_grad():
        pred, _ = model(x, text_tokens, z_goal=None)
    return pred.squeeze(0).cpu().numpy()


def main() -> None:
    from scipy.spatial.transform import Rotation as R
    from isaacgym import gymapi
    from isaacgym import gymutil

    import fcl
    import torch
    import cv2
    from stl_reader import stl_reader
    from obj_reader import obj_reader

    from trajectory_evaluation.comparison.run_rrt_ntfield_benchmark import (
        execute_path_and_time,
        reset_arm_to_q,
        _save_mp4_rgb,
        _path_as_6_list,
    )
    from trajectory_evaluation.ntfield.eval_trajectory_ntfield import load_network_and_function
    from planning.gradient_planner_trajectory import SCALE as NTFIELD_SCALE

    def _ee_to_primary_object_distance_m(
        gym, sim, env, ur, object_handles, viewer,
        spj, slj, ej, wj1, wj2, wj3, settle_steps: int,
    ) -> Tuple[
        Optional[float],
        Optional[List[float]],
        Optional[List[float]],
        Optional[str],
        Optional[str],
        Optional[List[float]],
        Optional[List[float]],
        Optional[List[float]],
    ]:
        """
        Returns
        -------
        distance_m : EE link to primary object root, or None if unavailable
        ee_xyz_m   : end-effector (wrist-proxy) position [x,y,z] in metres
        obj_xyz_m  : primary object root position [x,y,z] in metres
        ee_link    : rigid body name used for EE distance
        body_label : primary object root body label
        finger_midpoint_xyz_m : midpoint of inner fingers when both links exist,
            else None (same convention as evaluate_ntfield_oracle)
        left_finger_xyz_m, right_finger_xyz_m : world positions when the
            corresponding `left_inner_finger` / `right_inner_finger` exists
        """
        if not object_handles:
            return None, None, None, None, None, None, None, None
        try:
            for _ in range(max(0, int(settle_steps))):
                dof = gym.get_actor_dof_states(env, ur, gymapi.STATE_POS)["pos"][:6]
                gym.set_dof_target_position(env, spj, float(dof[0]))
                gym.set_dof_target_position(env, slj, float(dof[1]))
                gym.set_dof_target_position(env, ej,  float(dof[2]))
                gym.set_dof_target_position(env, wj1, float(dof[3]))
                gym.set_dof_target_position(env, wj2, float(dof[4]))
                gym.set_dof_target_position(env, wj3, float(dof[5]))
                gym.simulate(sim)
                gym.fetch_results(sim, True)
                gym.step_graphics(sim)
                if viewer is not None:
                    gym.draw_viewer(viewer, sim, True)
                gym.sync_frame_time(sim)

            rb_ur   = gym.get_actor_rigid_body_dict(env, ur)
            ee_name = None
            for cand in ("wrist_3_link", "tool0", "ee_link", "robotiq_arg2f_base_link"):
                if cand in rb_ur:
                    ee_name = cand
                    break
            if ee_name is None:
                return None, None, None, None, None, None, None, None

            st_ur = gym.get_actor_rigid_body_states(env, ur, gymapi.STATE_POS)
            i_ee  = int(rb_ur[ee_name])
            T_ee  = gymapi.Transform.from_buffer(st_ur["pose"][i_ee])
            p_ee  = np.array([T_ee.p.x, T_ee.p.y, T_ee.p.z], dtype=np.float64)

            finger_mid_xyz: Optional[List[float]] = None
            left_finger_xyz: Optional[List[float]] = None
            right_finger_xyz: Optional[List[float]] = None
            li = rb_ur.get("left_inner_finger")
            ri = rb_ur.get("right_inner_finger")
            if li is not None:
                pl = gymapi.Transform.from_buffer(st_ur["pose"][int(li)]).p
                left_finger_xyz = [float(pl.x), float(pl.y), float(pl.z)]
            if ri is not None:
                pr = gymapi.Transform.from_buffer(st_ur["pose"][int(ri)]).p
                right_finger_xyz = [float(pr.x), float(pr.y), float(pr.z)]
            if left_finger_xyz is not None and right_finger_xyz is not None:
                finger_mid_xyz = [
                    0.5 * (left_finger_xyz[0] + right_finger_xyz[0]),
                    0.5 * (left_finger_xyz[1] + right_finger_xyz[1]),
                    0.5 * (left_finger_xyz[2] + right_finger_xyz[2]),
                ]

            obj       = object_handles[0]
            rb_o      = gym.get_actor_rigid_body_dict(env, obj)
            st_o      = gym.get_actor_rigid_body_states(env, obj, gymapi.STATE_POS)
            inv       = {int(v): k for k, v in rb_o.items()}
            body_label = inv.get(0, "0")
            T_obj     = gymapi.Transform.from_buffer(st_o["pose"][0])
            p_obj     = np.array([T_obj.p.x, T_obj.p.y, T_obj.p.z], dtype=np.float64)

            return (
                float(np.linalg.norm(p_ee - p_obj)),
                p_ee.reshape(-1).tolist(),
                p_obj.reshape(-1).tolist(),
                ee_name,
                str(body_label),
                finger_mid_xyz,
                left_finger_xyz,
                right_finger_xyz,
            )
        except Exception:
            return None, None, None, None, None, None, None, None

    parser = argparse.ArgumentParser(description="PI-VLA integration — latent goal (MDN or CVAE)")
    parser.add_argument("--ntfield_checkpoint",     type=str, required=True)
    parser.add_argument("--ntfield_experiment_dir", type=str, default=None)
    parser.add_argument("--output_dir",             type=str, default=None)
    parser.add_argument("--object_z",               type=float, default=0.18)
    parser.add_argument("--target_obj_indices",     type=str, default="1,3,5")
    parser.add_argument("--num_objects",            type=int, default=3)
    parser.add_argument("--ox_min",  type=float, default=0.42)
    parser.add_argument("--ox_max",  type=float, default=0.98)
    parser.add_argument("--oy_min",  type=float, default=-0.38)
    parser.add_argument("--oy_max",  type=float, default=0.38)
    parser.add_argument("--seed",    type=int,   default=None)
    parser.add_argument("--use_viewer", action="store_true")
    parser.add_argument("--ntfield_device",    type=str, default="cuda:0")
    parser.add_argument("--physx_cpu",         action="store_true")
    parser.add_argument("--latent_checkpoint", type=str, required=True)
    parser.add_argument("--latent_device",     type=str, default="auto")
    parser.add_argument("--object_name",       type=str, default=None)
    parser.add_argument("--ntfield_step_size", type=float, default=0.02)
    parser.add_argument("--ntfield_max_steps", type=int,   default=200)
    parser.add_argument("--ntfield_tol",       type=float, default=0.01)
    parser.add_argument("--ntfield_delta_clamp_rad",        type=float, default=0.0)
    parser.add_argument("--ntfield_refine_max_steps",       type=int,   default=-1)
    parser.add_argument("--ntfield_refine_step_size",       type=float, default=None)
    parser.add_argument("--ntfield_refine_step_size_factor",type=float, default=None)
    parser.add_argument("--ntfield_refine_delta_clamp_rad", type=float, default=None)
    parser.add_argument("--ntfield_stagnate_max_steps",     type=int,   default=400)
    parser.add_argument("--ntfield_stagnate_patience",      type=int,   default=30)
    parser.add_argument("--ntfield_stagnate_rel_eps",       type=float, default=5e-4)
    parser.add_argument("--ntfield_stagnate_step_size",     type=float, default=None)
    parser.add_argument("--video_fps",          type=float, default=60.0)
    parser.add_argument("--planner_playback",   type=str,
                        choices=("direct", "settle"), default="direct")
    parser.add_argument("--no_isaac_hard_exit", action="store_true")
    parser.add_argument("--num_trials",         type=int, default=1)
    parser.add_argument("--stop_on_error",      action="store_true")
    parser.add_argument("--ee_settle_steps",    type=int,   default=10)
    parser.add_argument("--ee_success_thresh_m",type=float, default=None)
    args, argv_remainder = parser.parse_known_args()

    argv_gym = list(argv_remainder)
    if not args.use_viewer and "--headless" not in argv_gym:
        argv_gym.append("--headless")
    sys.argv = [sys.argv[0]] + argv_gym

    target_obj_index: List[int] = [
        int(x.strip()) for x in str(args.target_obj_indices).split(",") if x.strip()
    ]
    num_objects = int(args.num_objects)
    if num_objects <= 0:
        raise SystemExit("--num_objects must be >= 1")
    if len(target_obj_index) < num_objects:
        raise SystemExit(f"Need at least {num_objects} indices, got {len(target_obj_index)}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = (
        os.path.join(os.path.abspath(args.output_dir), stamp)
        if args.output_dir
        else os.path.join(_PI_VLA_ROOT, "output", "final_integrate", stamp)
    )
    os.makedirs(session_dir, exist_ok=True)

    num_trials = int(args.num_trials)
    if num_trials < 1:
        raise SystemExit("--num_trials must be >= 1")
    batch_log_path    = os.path.join(session_dir, "batch.log")
    batch_trials_jsonl = os.path.join(session_dir, "batch_trials.jsonl")

    oz      = float(args.object_z)
    ckpt_abs = _resolve_under_root(args.ntfield_checkpoint)
    if not os.path.isfile(ckpt_abs):
        raise SystemExit(f"NTField checkpoint not found: {ckpt_abs}")

    _cwd_prev = os.getcwd()
    os.chdir(HANWEN_GRASPING_ROOT)

    gym      = gymapi.acquire_gym()
    gym_args = gymutil.parse_arguments(description="final_integrate",
                                       headless=True, custom_parameters=[])
    gym_args.headless = not args.use_viewer
    if args.physx_cpu:
        gym_args.use_gpu = False

    dev_nt = torch.device(
        "cpu" if args.ntfield_device == "cpu" or not torch.cuda.is_available()
        else args.ntfield_device
    )
    ntfield_device_str = str(dev_nt) if dev_nt.type == "cuda" else "cpu"
    nt_net     = None
    ntfield_fn = None

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
    object_asset_files:      List[str]        = []
    object_collision_files:  List[str]        = []
    object_offset:           List[List[float]] = []
    object_common_prefix = "urdf/ycb/"
    with open(asset_root + "urdf/ycb/object_urdf_grasp.txt")      as f:
        for line in f: object_asset_files.append(object_common_prefix + line[:-1])
    with open(asset_root + "urdf/ycb/object_collision_grasp.txt") as f:
        for line in f: object_collision_files.append(object_common_prefix + line[:-1])
    with open(asset_root + "urdf/ycb/object_offset_grasp.txt")    as f:
        for line in f: object_offset.append([float(x) for x in line[:-1].split(" ")])

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
        [0,0,0],[0,0,0],[0,-0.138,0],[0,-0.007,0],[0,0.127,0],[0,0,0],[0,0,0],
    ]
    for idx, parts_path in enumerate(ur5e_collision_parts):
        cm = stl_reader(asset_root + parts_path)
        m  = fcl.BVHModel()
        cm.transform(ur5e_rotations[idx], ur5e_translations[idx])
        verts, tris = cm.get_vertices(), cm.get_faces()
        m.beginModel(len(verts), len(tris))
        m.addSubModel(verts, tris)
        m.endModel()
        ur5e_collision_models.append(m)

    trial_rows: List[Dict[str, Any]] = []
    with open(batch_log_path, "w", encoding="utf-8") as _blog:
        _blog.write(f"batch_session_dir={session_dir} num_trials={num_trials}\n")

    for trial_idx in range(num_trials):
        trial_dir = (session_dir if num_trials == 1
                     else os.path.join(session_dir, f"trial_{trial_idx:02d}"))
        if num_trials > 1:
            os.makedirs(trial_dir, exist_ok=True)
        if args.seed is not None:
            np.random.seed(int(args.seed) + int(trial_idx))

        with open(batch_log_path, "a", encoding="utf-8") as _blog:
            _blog.write(f"\n=== trial {trial_idx}/{num_trials} ===\n")

        sim    = None
        viewer = None
        try:
            table_dims  = gymapi.Vec3(TABLE_DIMS_X, TABLE_DIMS_Y, TABLE_DIMS_Z)
            sim_params  = gymapi.SimParams()
            sim_params.substeps  = 2
            sim_params.dt        = _SIM_DT
            sim_params.up_axis   = gymapi.UP_AXIS_Z
            sim_params.gravity   = gymapi.Vec3(0.0, 0.0, -9.81)
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
                raise RuntimeError("Failed to create sim")

            plane_params        = gymapi.PlaneParams()
            plane_params.normal = gymapi.Vec3(0, 0, 1)
            gym.add_ground(sim, plane_params)

            if nt_net is None:
                nt_net, ntfield_fn = load_network_and_function(
                    ckpt_abs, args.ntfield_experiment_dir, dev_nt, dim=6
                )

            viewer = None
            if not gym_args.headless:
                viewer = gym.create_viewer(sim, gymapi.CameraProperties())
                if viewer is None:
                    gym_args.headless = True

            spacing   = 2
            env_lower = gymapi.Vec3(-spacing, -spacing, 0)
            env_upper = gymapi.Vec3(spacing,  spacing,  0)

            asset_options = gymapi.AssetOptions()
            asset_options.fix_base_link            = True
            asset_options.default_dof_drive_mode   = int(gymapi.DOF_MODE_POS)
            asset_options.mesh_normal_mode         = gymapi.COMPUTE_PER_VERTEX
            asset_options.use_mesh_materials       = True
            ur5e_asset  = gym.load_asset(sim, asset_root, ur5e_asset_file, asset_options)
            table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)
            upper_cover_dims = gymapi.Vec3(table_dims.x, table_dims.y, 0.03)
            gym.create_box(sim, upper_cover_dims.x, upper_cover_dims.y, upper_cover_dims.z, asset_options)

            asset_options.fix_base_link = False
            object_assets = [gym.load_asset(sim, asset_root, ob, asset_options)
                             for ob in object_asset_files]

            ur5e_pose   = gymapi.Transform()
            ur5e_pose.p = gymapi.Vec3(0, 0, 0)
            ur5e_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0.5 * math.pi)
            table_pose   = gymapi.Transform()
            table_pose.p = gymapi.Vec3(table_dims.x * 0.5 + 0.3, 0.0, table_dims.z * 0.5)

            camera_focus = gymapi.Vec3(0, 0, 0)
            camera_props = gymapi.CameraProperties()
            camera_props.horizontal_fov = 70.25
            camera_props.width  = 1280
            camera_props.height = 720

            table_x_min = table_pose.p.x - table_dims.x * 0.5 + 0.05
            table_x_max = table_pose.p.x + table_dims.x * 0.5 - 0.10
            table_y_min = table_pose.p.y - table_dims.y * 0.5 + 0.10
            table_y_max = table_pose.p.y - table_dims.y * 0.5 - 0.20 + table_dims.y

            envs:                   List[Any] = []
            ur5e_handles:           List[Any] = []
            object_handles:         List[Any] = []
            placed_object_locations:List[List[float]] = []
            spj = slj = ej = wj1 = wj2 = wj3 = None

            target_file_idx = np.random.choice(target_obj_index, num_objects, replace=False)
            main_cam_handle = None
            top_cam_handle  = None

            for i in range(1):
                envs.append(gym.create_env(sim, env_lower, env_upper, 1))
                ur5e_handles.append(
                    gym.create_actor(envs[-1], ur5e_asset, ur5e_pose, "ur5e"+str(i), 0, 32767)
                )
                spj = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "shoulder_pan_joint")
                slj = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "shoulder_lift_joint")
                ej  = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "elbow_joint")
                wj1 = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "wrist_1_joint")
                wj2 = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "wrist_2_joint")
                wj3 = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "wrist_3_joint")
                gym.create_actor(envs[-1], table_asset, table_pose, "table"+str(i), 0, 1)

                objs_manager = fcl.DynamicAABBTreeCollisionManager()
                objs_manager.setup()
                obstacle_objs:    List[Any]        = []
                gt_obj_pos_list:  List[List[float]] = []
                gt_target_pos = [
                    float(np.random.uniform(max(table_x_min, 0.20 + table_dims.x / 2), table_x_max)),
                    float(np.random.uniform(table_y_min, table_y_max)),
                    float(oz),
                ]
                object_scaling_factor = np.ones(num_objects, dtype=np.float64)

                for k in range(num_objects):
                    object_pose = gymapi.Transform()
                    file_path   = object_collision_files[target_file_idx[k]]
                    cm          = obj_reader(asset_root + file_path)
                    cm.set_scale(object_scaling_factor[k])
                    cm.add_offset(object_offset[target_file_idx[k]])
                    verts, tris = cm.get_bounding_box_mesh()
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
                            if (np.sqrt((tx-gt_target_pos[0])**2+(ty-gt_target_pos[1])**2) <= 0.2):
                                is_collision = True
                                continue
                            for obj_xy in gt_obj_pos_list:
                                if np.sqrt((tx-obj_xy[0])**2+(ty-obj_xy[1])**2) <= 0.16:
                                    is_collision = True
                                    break
                    object_pose.p = gymapi.Vec3(tx, ty, oz)
                    gt_obj_pos_list.append([tx, ty])
                    placed_object_locations.append([tx, ty, float(oz)])
                    object_handles.append(
                        gym.create_actor(
                            envs[-1], object_assets[target_file_idx[k]],
                            object_pose, "object"+str(k)+str(i),
                            0, 2**(k+1), k+1,
                        )
                    )
                    gym.set_actor_scale(envs[-1], object_handles[-1], object_scaling_factor[k])
                    obstacle_objs.append(fcl.CollisionObject(m, t))
                    objs_manager.registerObjects(obstacle_objs)
                    objs_manager.setup()

                main_cam_handle = gym.create_camera_sensor(envs[-1], camera_props)
                gym.set_camera_location(main_cam_handle, envs[-1],
                                        gymapi.Vec3(3, 0, 0.3), camera_focus)
                top_cam_handle  = gym.create_camera_sensor(envs[-1], camera_props)
                top_cam_pos     = gymapi.Vec3(table_pose.p.x, table_pose.p.y + 0.001, 2.2)
                top_cam_target  = gymapi.Vec3(table_pose.p.x - 0.5, table_pose.p.y, table_pose.p.z)
                gym.set_camera_location(top_cam_handle, envs[-1], top_cam_pos, top_cam_target)

            if viewer is not None:
                gym.viewer_camera_look_at(viewer, None,
                                          gymapi.Vec3(2.2, 0, 0.5),
                                          gymapi.Vec3(0, 0, 0.5))

            gym.set_light_parameters(sim, 0,
                gymapi.Vec3(0.3,0.3,0.3), gymapi.Vec3(1,1,1), gymapi.Vec3(-1,0,0))
            gym.set_light_parameters(sim, 1,
                gymapi.Vec3(0.3,0.3,0.3), gymapi.Vec3(1,1,1), gymapi.Vec3(1,0,0))

            env = envs[-1]
            ur  = ur5e_handles[-1]
            real_position = False

            for t in range(2000):
                if not real_position:
                    for jh, val in zip([spj,slj,ej,wj1,wj2,wj3],
                                       [0, -math.pi/2, 0, -math.pi/2, 0, 0]):
                        gym.set_dof_target_position(env, jh, val)
                    real_position = True
                gym.simulate(sim); gym.fetch_results(sim, True)
                gym.step_graphics(sim)
                if viewer: gym.draw_viewer(viewer, sim, True)
                gym.sync_frame_time(sim)

            _HOME_DOF        = [0.7, -2.0, 2.5, -0.3, 0.7, 0.0]
            _START_SETTLE    = 30
            for _ in range(_START_SETTLE):
                for jh, val in zip([spj,slj,ej,wj1,wj2,wj3], _HOME_DOF):
                    gym.set_dof_target_position(env, jh, val)
                gym.simulate(sim); gym.fetch_results(sim, True)
                gym.step_graphics(sim)
                if viewer: gym.draw_viewer(viewer, sim, True)
                gym.sync_frame_time(sim)

            assert top_cam_handle is not None
            gym.render_all_camera_sensors(sim)
            raw_top  = gym.get_camera_image(sim, env, top_cam_handle, gymapi.IMAGE_COLOR)
            rgba_top = raw_top.reshape(camera_props.height, camera_props.width, 4)
            rgb_top  = rgba_top[..., :3].copy()
            cv2.imwrite(os.path.join(trial_dir, "top_view.png"),
                        cv2.cvtColor(rgb_top, cv2.COLOR_RGB2BGR))

            if args.object_name:
                infer_object_name = args.object_name.strip()
            else:
                try:
                    raw_fname = object_asset_files[target_file_idx[0]]
                    stem      = raw_fname.split("/")[-2] if "/" in raw_fname else raw_fname
                    stem      = re.sub(r"^\d+_", "", stem)
                    infer_object_name = stem.replace("_", " ")
                except Exception:
                    infer_object_name = "object"
            print(f"[latent] Using object name for text prompt: '{infer_object_name}'", flush=True)

            dof_snapshot = gym.get_actor_dof_states(env, ur, gymapi.STATE_POS)
            q_start_live = np.array(dof_snapshot["pos"][:6], dtype=np.float64)

            # ── Inference (MDN or CVAE auto-detected) ────────────────────────
            latent_goal_pred = _infer_latent_on_image(
                rgb_top,
                _resolve_under_root(args.latent_checkpoint),
                args.latent_device,
                object_name=infer_object_name,
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
            refine_delta_clamp_rad: Optional[float] = None
            if (args.ntfield_refine_delta_clamp_rad is not None and
                    float(args.ntfield_refine_delta_clamp_rad) > 0.0):
                refine_delta_clamp_rad = float(args.ntfield_refine_delta_clamp_rad)
            stagnate_step = (
                float(args.ntfield_stagnate_step_size)
                if args.ntfield_stagnate_step_size is not None
                else float(args.ntfield_step_size) * 0.25
            )

            prompt_text = f"grasp {infer_object_name.strip().lower()}"

            def _sim_object_root_xyz_m(actor_handle):
                st_o = gym.get_actor_rigid_body_states(env, actor_handle, gymapi.STATE_POS)
                T_o = gymapi.Transform.from_buffer(st_o["pose"][0])
                return [float(T_o.p.x), float(T_o.p.y), float(T_o.p.z)]

            def ntfield_plan_gradient_with_goal_latent(
                teacher_network, q_start, z_goal_hat,
                step_size=0.02, max_steps=200, tol=0.01, device="cuda",
                delta_clamp_rad=0.0, refine_max_steps=0, refine_step_size=0.01,
                refine_delta_clamp_rad=None, stagnate_max_steps=0,
                stagnate_patience=30, stagnate_rel_eps=5e-4, stagnate_step_size=0.005,
            ):
                import torch

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

                q_start    = np.asarray(q_start, dtype=np.float32).reshape(-1)
                q_curr_norm = q_start / NTFIELD_SCALE
                q_curr_t   = torch.tensor(q_curr_norm, dtype=torch.float32,
                                          device=device).unsqueeze(0)
                if isinstance(z_goal_hat, np.ndarray):
                    z_goal_hat = torch.tensor(z_goal_hat.reshape(1,-1),
                                              dtype=torch.float32, device=device)
                else:
                    z_goal_hat = z_goal_hat.reshape(1,-1).to(device)

                path_norm  = [q_curr_norm.copy()]
                final_dist = None
                converged  = False
                stopped    = "max_main_steps"
                main_clamp = float(delta_clamp_rad) if delta_clamp_rad > 0.0 else 0.0

                for _ in range(max_steps):
                    q_curr_t, final_dist, converged = _grad_step(
                        q_curr_t, z_goal_hat, step_size, main_clamp)
                    path_norm.append(q_curr_t.detach().cpu().numpy()[0].copy())
                    if converged:
                        stopped = "latent_tol_main"; break

                refine_clamp = (float(refine_delta_clamp_rad)
                                if refine_delta_clamp_rad and refine_delta_clamp_rad > 0 else 0.0)
                if refine_max_steps > 0 and not converged:
                    for _ in range(refine_max_steps):
                        q_curr_t, final_dist, converged = _grad_step(
                            q_curr_t, z_goal_hat, refine_step_size, refine_clamp)
                        path_norm.append(q_curr_t.detach().cpu().numpy()[0].copy())
                        if converged:
                            stopped = "latent_tol_refine"; break
                    else:
                        if not converged: stopped = "max_refine_steps"

                if stagnate_max_steps > 0 and not converged:
                    best_d = final_dist if final_dist is not None else _latent_dist(q_curr_t, z_goal_hat)
                    stall  = 0
                    for _ in range(stagnate_max_steps):
                        q_curr_t, final_dist, converged = _grad_step(
                            q_curr_t, z_goal_hat, stagnate_step_size, 0.0)
                        path_norm.append(q_curr_t.detach().cpu().numpy()[0].copy())
                        if converged:
                            stopped = "latent_tol_stagnate"; break
                        imp = (best_d - final_dist) / max(best_d, 1e-8)
                        if imp > stagnate_rel_eps:
                            best_d = min(best_d, final_dist); stall = 0
                        else:
                            stall += 1
                            if stall >= stagnate_patience:
                                stopped = "latent_dist_stagnated"; break
                    else:
                        if not converged and stopped != "latent_tol_stagnate":
                            stopped = "max_stagnate_steps"

                if len(path_norm) < 2:
                    path_norm.append(q_curr_t.detach().cpu().numpy()[0].copy())
                return [p * NTFIELD_SCALE for p in path_norm], \
                       {"final_latent_dist": final_dist, "stopped": stopped}

            def _record_ntfield_path(path_raw, label, mp4_path):
                if path_raw and len(path_raw) >= 2:
                    reset_arm_to_q(gym, sim, env, ur, spj, slj, ej,
                                   wj1, wj2, wj3, viewer, q_start_live, n_steps=200)
                    frames: List[np.ndarray] = []
                    execute_path_and_time(
                        gym, sim, env, ur, spj, slj, ej, wj1, wj2, wj3, viewer,
                        _path_as_6_list(path_raw), label,
                        main_cam_handle=main_cam_handle,
                        camera_props=camera_props,
                        record_rgb=frames,
                        planner_playback=args.planner_playback,
                    )
                    _save_mp4_rgb(frames, mp4_path, fps=args.video_fps)
                else:
                    print(f"[warn] NTField planner returned an empty path ({label}).")

            path_pred, _ = ntfield_plan_gradient_with_goal_latent(
                nt_net, q_start_live.reshape(1,-1), latent_goal_pred.reshape(1,-1),
                step_size=args.ntfield_step_size, max_steps=args.ntfield_max_steps,
                tol=args.ntfield_tol, device=ntfield_device_str,
                delta_clamp_rad=args.ntfield_delta_clamp_rad,
                refine_max_steps=refine_max, refine_step_size=refine_step,
                refine_delta_clamp_rad=refine_delta_clamp_rad,
                stagnate_max_steps=int(args.ntfield_stagnate_max_steps),
                stagnate_patience=int(args.ntfield_stagnate_patience),
                stagnate_rel_eps=float(args.ntfield_stagnate_rel_eps),
                stagnate_step_size=stagnate_step,
            )
            mp4_path = os.path.join(trial_dir, "ntfield_trajectory_predicted_latent_goal.mp4")
            _record_ntfield_path(path_pred, "predicted_latent_goal", mp4_path)

            (
                ee_m,
                ee_xyz,
                tgt_xyz,
                _ee_ln,
                _bd,
                finger_mid_xyz,
                left_finger_xyz,
                right_finger_xyz,
            ) = _ee_to_primary_object_distance_m(
                gym, sim, env, ur, object_handles, viewer,
                spj, slj, ej, wj1, wj2, wj3, int(args.ee_settle_steps),
            )

            tgt_loc = tgt_xyz
            other_locs: List[List[float]] = []
            if len(object_handles) > 1:
                for ah in object_handles[1:]:
                    other_locs.append(_sim_object_root_xyz_m(ah))
            elif placed_object_locations and len(placed_object_locations) > 1:
                other_locs = [list(map(float, p)) for p in placed_object_locations[1:]]

            if tgt_loc is None and placed_object_locations:
                tgt_loc = [float(x) for x in placed_object_locations[0]]

            finger_midpoint_to_target_xy_distance_m = None
            finger_midpoint_to_target_z_diff_m = None
            if finger_mid_xyz is not None and tgt_loc is not None:
                fm = np.asarray(finger_mid_xyz, dtype=np.float64).reshape(3)
                tg = np.asarray(tgt_loc, dtype=np.float64).reshape(3)
                finger_midpoint_to_target_xy_distance_m = float(
                    np.hypot(float(fm[0] - tg[0]), float(fm[1] - tg[1]))
                )
                # Signed: fingertip midpoint Z minus target reference Z (object root).
                finger_midpoint_to_target_z_diff_m = float(fm[2] - tg[2])

            summary = {
                "trial_index": trial_idx,
                "prompt": prompt_text,
                "target_object_location_xyz_m": tgt_loc,
                "other_object_locations_xyz_m": other_locs,
                "end_effector_location_xyz_m": ee_xyz,
                "left_finger_location_xyz_m": left_finger_xyz,
                "right_finger_location_xyz_m": right_finger_xyz,
                "finger_midpoint_location_xyz_m": finger_mid_xyz,
                "finger_midpoint_to_target_xy_distance_m": finger_midpoint_to_target_xy_distance_m,
                "finger_midpoint_to_target_z_diff_m": finger_midpoint_to_target_z_diff_m,
                "ee_to_target_distance_m": ee_m,
            }

            with open(os.path.join(trial_dir, "pipeline_summary.json"), "w") as f:
                json.dump(summary, f, indent=2)
            trial_rows.append(dict(summary))
            with open(batch_trials_jsonl, "a") as jf:
                jf.write(json.dumps(summary, default=str) + "\n")

        except Exception as exc:
            err_row: Dict[str, Any] = {
                "trial_index": trial_idx,
                "prompt": None,
                "target_object_location_xyz_m": None,
                "other_object_locations_xyz_m": None,
                "end_effector_location_xyz_m": None,
                "left_finger_location_xyz_m": None,
                "right_finger_location_xyz_m": None,
                "finger_midpoint_location_xyz_m": None,
                "finger_midpoint_to_target_xy_distance_m": None,
                "finger_midpoint_to_target_z_diff_m": None,
                "ee_to_target_distance_m": None,
                "error": str(exc),
            }
            trial_rows.append(err_row)
            tb = traceback.format_exc()
            with open(batch_log_path, "a") as lf:
                lf.write(tb + "\n")
            with open(batch_trials_jsonl, "a") as jf:
                jf.write(json.dumps(err_row, default=str) + "\n")
            print(f"[batch] trial {trial_idx} failed: {exc}", flush=True)
            if args.stop_on_error:
                raise
        finally:
            if viewer is not None: gym.destroy_viewer(viewer)
            if sim    is not None: gym.destroy_sim(sim)
            gc.collect()
            if dev_nt.type == "cuda":
                torch.cuda.synchronize()

    os.chdir(_cwd_prev)
    aggregate = _write_batch_summary(session_dir, trial_rows, args)
    print(json.dumps(aggregate, indent=2))
    if not args.no_isaac_hard_exit:
        os._exit(0)


if __name__ == "__main__":
    main()