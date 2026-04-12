#!/usr/bin/env python3
"""
End-to-end PI-VLA integration (Isaac Gym + multi-obj-loc + TracIK + NTField).

Matches collect_multi_obj_layout_only.py exactly:
  - 3 YCB objects (TARGET_OBJ_INDEX = [1, 3, 5]) placed with collision-free
    random placement on the table
  - Same camera, lighting, table dims, settle steps, and HOME_DOF before top RGB
  - Object location predicted per-object using GraspHead (ResNet-50 + CLIP text + FiLM)
    trained by train_fast.py / extract_features.py pipeline

Run from PI-VLA root:
python final_integrate/run_integrated_pipeline_multi.py \
  --ntfield_checkpoint ntrl-demo/Experiments/UR5_trajectory_no_wall_accuracy_check/trajectory_03_25_20_28/Model_Epoch_05000_ValLoss_7.820605e-01.pt \
  --objloc_checkpoint  /home/hojinsohn/VLM-NT/PI-VLA/output/runs/exp_fast/best.pt \
  --target_object_idx  0 \
  --headless

Outputs under output/final_integrate_multi/<timestamp>/
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import textwrap
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

for _p in (HANWEN_GRASPING_ROOT, _UTIL_DIR, _GRASP_UTIL_DIR, _PI_VLA_ROOT, _NTRL_DEMO, _IMG2OBJ):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Isaac Gym must be imported before torch (see isaacgym/gymdeps.py).
from isaacgym import gymapi, gymutil  # noqa: E402
import torch  # noqa: E402


# ---------------------------------------------------------------------------
# Table / scene constants — must match collect_multi_obj_layout_only.py exactly
# ---------------------------------------------------------------------------
TABLE_DIMS_X   = 0.8
TABLE_DIMS_Y   = 1.0
TABLE_DIMS_Z   = 0.10
DRAWER_HEIGHT  = 0.40
NUM_OF_OBJECTS = 3
TARGET_OBJ_INDEX = [1, 3, 5]   # indices into object_urdf_grasp.txt

# Before top-view RGB (matches collect_multi_obj_layout_only.py HOME_DOF + START_SETTLE_STEPS)
HOME_DOF             = [0.7, -2.0, 2.5, -0.3, 0.7, 0.0]
START_SETTLE_STEPS   = 30

# YCB display names — same mapping as collect_multi_obj_layout_only.py
_OBJECT_DISPLAY_NAMES = {
    "002_master_chef_can": "master chef can",
    "004_sugar_box":       "sugar box",
    "005_tomato_soup_can": "tomato soup can",
    "006_mustard_bottle":  "mustard bottle",
    "036_wood_block":      "wood block",
    "011_banana":          "banana",
}


def _grasp_asset_path_to_display_name(rel_path: str) -> str:
    rel    = rel_path.replace("\\", "/")
    folder = rel.split("/")[-2] if "/" in rel else ""
    stem   = os.path.splitext(os.path.basename(rel))[0]
    for key in (folder, stem):
        if key and key in _OBJECT_DISPLAY_NAMES:
            return _OBJECT_DISPLAY_NAMES[key]
    label = stem or folder
    if len(label) > 4 and label[:3].isdigit() and label[3] == "_":
        label = label[4:]
    return label.replace("_", " ").strip()


def _resolve_under_root(path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(_PI_VLA_ROOT, path))


# ---------------------------------------------------------------------------
# Image preprocessing — matches convert_to_pt.py process_img exactly
# ---------------------------------------------------------------------------

def _process_img(img: np.ndarray, img_size: int = 224):
    import torch
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    h, w = img.shape[:2]
    if h != img_size or w != img_size:
        ys = np.linspace(0, h - 1, img_size, dtype=np.float32).astype(np.int32)
        xs = np.linspace(0, w - 1, img_size, dtype=np.float32).astype(np.int32)
        img = img[np.ix_(ys, xs)]
    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


# ---------------------------------------------------------------------------
# GraspHead model — mirrors train_fast.py exactly
# ---------------------------------------------------------------------------

def _build_grasp_head(img_feat_dim: int = 2048, txt_feat_dim: int = 512, embed_dim: int = 512):
    import torch.nn as nn

    class FiLM(nn.Module):
        def __init__(self, cond_dim, feat_dim):
            super().__init__()
            self.gamma_proj = nn.Linear(cond_dim, feat_dim)
            self.beta_proj  = nn.Linear(cond_dim, feat_dim)
            nn.init.ones_(self.gamma_proj.weight);  nn.init.zeros_(self.gamma_proj.bias)
            nn.init.zeros_(self.beta_proj.weight);  nn.init.zeros_(self.beta_proj.bias)
        def forward(self, x, cond):
            return self.gamma_proj(cond) * x + self.beta_proj(cond)

    class GraspHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.img_proj = nn.Linear(img_feat_dim, embed_dim)
            self.film     = FiLM(txt_feat_dim, embed_dim)
            self.head     = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, 256), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(256, 128),       nn.GELU(), nn.Dropout(0.1),
                nn.Linear(128, 3),
            )
        def forward(self, img_feat, txt_feat):
            x = self.img_proj(img_feat)
            x = x / x.norm(dim=-1, keepdim=True)
            x = self.film(x, txt_feat)
            return self.head(x)

    return GraspHead()


# ---------------------------------------------------------------------------
# Object location inference — one call per object
# ---------------------------------------------------------------------------

class ObjLocInferencer:
    """
    Loads ResNet-50, CLIP text encoder, and GraspHead once.
    Call predict(image, prompt) for each object.
    """

    def __init__(self, checkpoint_path: str, device: str = "auto"):
        import torch
        import torchvision.models as tvm
        import open_clip

        self.dev = torch.device(
            "cuda" if torch.cuda.is_available() and device in ("auto", "cuda") else "cpu"
        )

        # ── ResNet-50 image encoder ──────────────────────────────────────
        resnet = tvm.resnet50(pretrained=True)
        self.image_encoder = torch.nn.Sequential(*list(resnet.children())[:-1]).to(self.dev).eval()

        # ── CLIP text encoder ────────────────────────────────────────────
        clip, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        self.tokenizer            = open_clip.get_tokenizer("ViT-B-32")
        self.text_transformer     = clip.transformer.to(self.dev).eval()
        self.token_embedding      = clip.token_embedding.to(self.dev).eval()
        self.positional_embedding = clip.positional_embedding.to(self.dev)
        self.ln_final             = clip.ln_final.to(self.dev).eval()
        self.text_projection      = clip.text_projection.to(self.dev)
        self.attn_mask            = clip.attn_mask.to(self.dev) if clip.attn_mask is not None else None

        # ── GraspHead (trained weights) ──────────────────────────────────
        self.head = _build_grasp_head().to(self.dev)
        ckpt = torch.load(_resolve_under_root(checkpoint_path), map_location="cpu")
        self.head.load_state_dict(ckpt["model"])
        self.head.eval()

        # Freeze everything
        for m in (self.image_encoder, self.text_transformer,
                  self.token_embedding, self.ln_final, self.head):
            for p in m.parameters():
                p.requires_grad_(False)

        # Cache for text embeddings (prompts repeat across objects)
        self._text_cache: dict = {}

        print(f"ObjLocInferencer ready on {self.dev}")

    @torch.no_grad()
    def _encode_image(self, img_np: np.ndarray):
        import torch
        x = _process_img(img_np, 224).unsqueeze(0).to(self.dev)
        feat = self.image_encoder(x).squeeze(-1).squeeze(-1)   # (1, 2048)
        return feat

    @torch.no_grad()
    def _encode_text(self, prompt: str):
        import torch
        if prompt not in self._text_cache:
            tokens = self.tokenizer([prompt]).to(self.dev)
            x = self.token_embedding(tokens) + self.positional_embedding
            x = x.permute(1, 0, 2)
            x = self.text_transformer(x, attn_mask=self.attn_mask)
            x = x.permute(1, 0, 2)
            x = self.ln_final(x)
            x = x[torch.arange(x.shape[0]), tokens.argmax(dim=-1)]
            x = x @ self.text_projection
            x = x / x.norm(dim=-1, keepdim=True)
            self._text_cache[prompt] = x
        return self._text_cache[prompt]

    @torch.no_grad()
    def predict(self, img_np: np.ndarray, prompt: str) -> Tuple[float, float, float]:
        """Returns predicted (x, y, z) in world metres."""
        img_feat = self._encode_image(img_np)          # (1, 2048)
        txt_feat = self._encode_text(prompt)            # (1, 512)
        xyz      = self.head(img_feat, txt_feat)        # (1, 3)
        xyz      = xyz.squeeze(0).cpu().numpy()
        return float(xyz[0]), float(xyz[1]), float(xyz[2])


# ---------------------------------------------------------------------------
# Grasp goal finding (unchanged from original)
# ---------------------------------------------------------------------------

def find_grasp_q_goal(
    rac, RC_mod, scene_info, grasp_data, grasp_list,
    obj_world_xy, target_idx, object_mesh, object_collision_models,
    plane_obj, get_swept_volume_size_fn,
):
    GT_OBJ_POS_LIST = [obj_world_xy.tolist()]
    num_grasp = 0
    swept_size = sys.maxsize
    init2grasp_path = None
    grasp_target_q  = None
    rrt_plan_s      = 0.0
    consecutive_path_failures = 0
    MAX_FAIL = 15

    for grasp_idx in grasp_list:
        if consecutive_path_failures >= MAX_FAIL:
            break
        target_grasp_pos  = grasp_data[grasp_idx]["target_pos"].copy()
        target_grasp_quat = grasp_data[grasp_idx]["target_quat"]
        target_grasp_pos[:2] = target_grasp_pos[:2] + GT_OBJ_POS_LIST[target_idx][:2]
        init2grasp_angels_temp  = rac.grasp_verify(target_grasp_pos, target_grasp_quat)
        grasp2init_angels_temp  = rac.grasp_verify(target_grasp_pos + [0, 0, 0.01], target_grasp_quat)
        if init2grasp_angels_temp is None or grasp2init_angels_temp is None:
            continue
        if not rac.arm_collision_free(init2grasp_angels_temp, plane_obj, object_collision_models, []):
            continue
        if not rac.arm_collision_free(grasp2init_angels_temp, plane_obj, object_collision_models, []):
            continue

        t0 = time.perf_counter()
        init2grasp_path_temp = RC_mod.get_path2grasp(
            rac, init2grasp_angels_temp, scene_info,
            target_mesh=object_mesh[target_idx], time_limit=30,
            given_static_model=object_collision_models,
        )
        rrt_wall_s = float(time.perf_counter() - t0)
        if init2grasp_path_temp is None:
            consecutive_path_failures += 1
            continue

        temp_mod_bbox = rac.modify_grasp_bbox(
            init2grasp_angels_temp, target_mesh=object_mesh[target_idx], visualize=False
        )
        grasp2init_path_temp = RC_mod.get_path2start(
            rac, grasp2init_angels_temp, temp_mod_bbox, scene_info,
            time_limit=30, given_static_model=object_collision_models,
        )
        if grasp2init_path_temp is None:
            consecutive_path_failures += 1
            continue
        consecutive_path_failures = 0

        sv1, sv1v = rac.get_swept_volume(init2grasp_path_temp,  frame_rate=60, scene_info=scene_info, animation=False, static_vi=False)
        sv2, sv2v = rac.get_swept_volume(grasp2init_path_temp,  w_target=temp_mod_bbox, frame_rate=60, scene_info=scene_info, animation=False, static_vi=False)
        num_grasp += 1
        _, swept_verts = rac.get_swept_center(sv1v + sv2v, scene_info, 0.6)
        temp_size = get_swept_volume_size_fn(swept_verts)
        if temp_size < swept_size:
            swept_size       = temp_size
            init2grasp_path  = init2grasp_path_temp
            grasp_target_q   = np.asarray(init2grasp_angels_temp, dtype=np.float64).reshape(6)
            rrt_plan_s       = rrt_wall_s
        if num_grasp == 1:
            break

    return grasp_target_q, init2grasp_path, rrt_plan_s


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    from scipy.spatial.transform import Rotation as R
    import fcl
    import cv2
    import robot_arm_configuration as RC
    from stl_reader import stl_reader
    from obj_reader import obj_reader

    from trajectory_evaluation.comparison.run_rrt_ntfield_benchmark import (
        TABLE_DIMS_X as _TDX, TABLE_DIMS_Y as _TDY, TABLE_DIMS_Z as _TDZ,
        DRAWER_HEIGHT as _DH,
        execute_path_and_time, get_swept_volume_size,
        reset_arm_to_q, _save_mp4_rgb, _path_as_6_list, sim_dt,
    )
    from trajectory_evaluation.ntfield.eval_trajectory_ntfield import _ModelShim, load_network_and_function
    from planning.gradient_planner_trajectory import SCALE as NTFIELD_SCALE
    from planning.gradient_planner_trajectory import plan as ntfield_plan

    parser = argparse.ArgumentParser(description="PI-VLA multi-object integrated pipeline")
    parser.add_argument("--ntfield_checkpoint",     type=str, required=True)
    parser.add_argument("--ntfield_experiment_dir", type=str, default=None)
    parser.add_argument("--objloc_checkpoint",      type=str, required=True,
                        help="Path to best.pt from train_fast.py (GraspHead weights)")
    parser.add_argument("--output_dir",             type=str, default=None)
    parser.add_argument("--target_object_idx",      type=int, default=0,
                        help="Which of the 3 objects to grasp (0, 1, or 2)")
    parser.add_argument("--seed",                   type=int, default=None)
    parser.add_argument("--use_viewer",             action="store_true")
    parser.add_argument("--objloc_device",          type=str, default="auto")
    parser.add_argument("--ntfield_device",         type=str, default="cuda:0")
    parser.add_argument("--ntfield_step_size",      type=float, default=0.02)
    parser.add_argument("--ntfield_max_steps",      type=int,   default=200)
    parser.add_argument("--ntfield_tol",            type=float, default=0.01)
    parser.add_argument("--ntfield_goal_eps_rad",   type=float, default=None)
    parser.add_argument("--video_fps",              type=float, default=60.0)
    parser.add_argument("--planner_playback",       type=str,
                        choices=("direct", "settle"), default="direct")
    args, argv_remainder = parser.parse_known_args()

    argv_gym = list(argv_remainder)
    if not args.use_viewer and "--headless" not in argv_gym:
        argv_gym.append("--headless")
    sys.argv = [sys.argv[0]] + argv_gym

    if args.seed is not None:
        np.random.seed(args.seed)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.abspath(args.output_dir) if args.output_dir else \
        os.path.join(_PI_VLA_ROOT, "output", "final_integrate", stamp)
    os.makedirs(session_dir, exist_ok=True)

    # ── NTField ───────────────────────────────────────────────────────────
    ckpt_abs = _resolve_under_root(args.ntfield_checkpoint)
    if not os.path.isfile(ckpt_abs):
        raise SystemExit(f"NTField checkpoint not found: {ckpt_abs}")
    dev_nt = torch.device(
        "cpu" if args.ntfield_device == "cpu" or not torch.cuda.is_available()
        else args.ntfield_device
    )
    _, ntfield_fn = load_network_and_function(ckpt_abs, args.ntfield_experiment_dir, dev_nt, dim=6)
    nt_model = _ModelShim(ntfield_fn)
    ntfield_device_str = str(dev_nt) if dev_nt.type == "cuda" else "cpu"
    goal_eps = (float(args.ntfield_goal_eps_rad) if args.ntfield_goal_eps_rad is not None
                else float(args.ntfield_tol * NTFIELD_SCALE))

    # ── Object localisation model (loaded once, reused per object) ────────
    print("Loading ObjLoc model (ResNet-50 + CLIP text + FiLM)...")
    inferencer = ObjLocInferencer(args.objloc_checkpoint, device=args.objloc_device)

    # ── Isaac Gym setup ───────────────────────────────────────────────────
    _cwd_prev = os.getcwd()
    os.chdir(HANWEN_GRASPING_ROOT)

    gym      = gymapi.acquire_gym()
    gym_args = gymutil.parse_arguments(description="final_integrate", headless=True, custom_parameters=[])
    gym_args.headless = not args.use_viewer

    table_dims   = gymapi.Vec3(TABLE_DIMS_X, TABLE_DIMS_Y, TABLE_DIMS_Z)
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
    sim_params.physx.use_gpu    = gym_args.use_gpu
    sim_params.use_gpu_pipeline = False

    sim = gym.create_sim(gym_args.compute_device_id, gym_args.graphics_device_id,
                         gym_args.physics_engine, sim_params)
    if sim is None:
        raise SystemExit("Failed to create sim")

    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)

    asset_root = "./assets/"

    # ── Load URDF lists (same as collect script) ──────────────────────────
    object_asset_files     = []
    object_collision_files = []
    object_offset          = []
    object_common_prefix   = "urdf/ycb/"
    with open(asset_root + "urdf/ycb/object_urdf_grasp.txt") as f:
        for line in f:
            object_asset_files.append(object_common_prefix + line.rstrip())
    with open(asset_root + "urdf/ycb/object_collision_grasp.txt") as f:
        for line in f:
            object_collision_files.append(object_common_prefix + line.rstrip())
    with open(asset_root + "urdf/ycb/object_offset_grasp.txt") as f:
        for line in f:
            object_offset.append([float(x) for x in line.rstrip().split()])

    # ── UR5e collision meshes ─────────────────────────────────────────────
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
        R.from_euler("x",  [90],        degrees=True),
        R.from_euler("xy", [90,  180],  degrees=True),
        R.from_euler("xy", [180, 180],  degrees=True),
        R.from_euler("z",  [-180],      degrees=True),
        R.from_euler("x",  [-180],      degrees=True),
        R.from_euler("x",  [90],        degrees=True),
        R.from_euler("z",  [-90],       degrees=True),
    ]
    ur5e_translations = [[0,0,0],[0,0,0],[0,-0.138,0],[0,-0.007,0],[0,0.127,0],[0,0,0],[0,0,0]]
    ur5e_collision_models = []
    for idx, parts_path in enumerate(ur5e_collision_parts):
        mesh = stl_reader(asset_root + parts_path)
        mesh.transform(ur5e_rotations[idx], ur5e_translations[idx])
        verts, tris = mesh.get_vertices(), mesh.get_faces()
        m = fcl.BVHModel()
        m.beginModel(len(verts), len(tris)); m.addSubModel(verts, tris); m.endModel()
        ur5e_collision_models.append(m)

    # ── Scene assets ──────────────────────────────────────────────────────
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)
    asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
    asset_options.use_mesh_materials = True

    ur5e_asset_file = "urdf/ur5e/ur5e_mimic_real_gripper_test.urdf"
    ur5e_asset  = gym.load_asset(sim, asset_root, ur5e_asset_file, asset_options)
    table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)
    upper_cover_dims  = gymapi.Vec3(table_dims.x, table_dims.y, 0.03)
    upper_cover_asset = gym.create_box(sim, upper_cover_dims.x, upper_cover_dims.y, upper_cover_dims.z, asset_options)

    asset_options.fix_base_link = False
    object_assets = [gym.load_asset(sim, asset_root, ob, asset_options) for ob in object_asset_files]

    # ── Poses ─────────────────────────────────────────────────────────────
    ur5e_pose = gymapi.Transform()
    ur5e_pose.p = gymapi.Vec3(0, 0, 0)
    ur5e_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0.5 * math.pi)

    table_pose = gymapi.Transform()
    table_pose.p = gymapi.Vec3(table_dims.x * 0.5 + 0.3, 0.0, table_dims.z * 0.5)

    upper_cover_pose = gymapi.Transform()
    upper_cover_pose.p = gymapi.Vec3(table_pose.p.x, 0.0, table_dims.z + drawer_height + 0.015)

    # Table placement bounds — same as collect script
    table_x_min = table_pose.p.x - table_dims.x * 0.5 + 0.05
    table_x_max = table_pose.p.x + table_dims.x * 0.5 - 0.10
    table_y_min = table_pose.p.y - table_dims.y * 0.5 + 0.10
    table_y_max = table_pose.p.y + table_dims.y * 0.5 - 0.20

    camera_focus = gymapi.Vec3(0, 0, 0)
    camera_props = gymapi.CameraProperties()
    camera_props.horizontal_fov = 70.25
    camera_props.width  = 1280
    camera_props.height = 720

    # FCL plane + table
    col_plane = fcl.Plane(np.array([0., 0., 1.]), 0)
    plane_obj = fcl.CollisionObject(col_plane, fcl.Transform())
    col_table  = fcl.Box(table_dims.x, table_dims.y, table_dims.z)
    trans_table = fcl.Transform(np.array([table_dims.x * 0.5 + 0.3, 0.0, table_dims.z * 0.5]))
    table_obj  = fcl.CollisionObject(col_table, trans_table)
    object_collision_models = [table_obj]

    # ── Create env ────────────────────────────────────────────────────────
    spacing   = 2
    env_lower = gymapi.Vec3(-spacing, -spacing, 0)
    env_upper = gymapi.Vec3(spacing,  spacing,  0)
    env = gym.create_env(sim, env_lower, env_upper, 1)
    ur  = gym.create_actor(env, ur5e_asset, ur5e_pose, "ur5e", 0, 32767)

    spj = gym.find_actor_dof_handle(env, ur, "shoulder_pan_joint")
    slj = gym.find_actor_dof_handle(env, ur, "shoulder_lift_joint")
    ej  = gym.find_actor_dof_handle(env, ur, "elbow_joint")
    wj1 = gym.find_actor_dof_handle(env, ur, "wrist_1_joint")
    wj2 = gym.find_actor_dof_handle(env, ur, "wrist_2_joint")
    wj3 = gym.find_actor_dof_handle(env, ur, "wrist_3_joint")

    gym.create_actor(env, table_asset, table_pose, "table", 0, 1)

    # ── Randomly place 3 objects with collision-free check (matches collect) ──
    target_file_idx   = np.random.choice(TARGET_OBJ_INDEX, NUM_OF_OBJECTS, replace=False)
    object_slot_names = [
        _grasp_asset_path_to_display_name(object_asset_files[int(target_file_idx[k])])
        for k in range(NUM_OF_OBJECTS)
    ]
    print(f"Objects: {object_slot_names}")
    print(f"Target for grasping: [{args.target_object_idx}] {object_slot_names[args.target_object_idx]}")

    object_scaling_factor = np.ones(NUM_OF_OBJECTS)
    object_handles        = []
    object_collision_lib  = []
    object_status_list    = []
    object_reader_tracker = []
    object_mesh           = []
    flex_collision_models = []

    objs_manager  = fcl.DynamicAABBTreeCollisionManager()
    obstacle_objs = []
    GT_OBJ_POS_LIST = []

    for k in range(NUM_OF_OBJECTS):
        is_collision = True
        tx = ty = tz = 0.0
        while is_collision:
            tx = np.random.uniform(table_x_min, table_x_max)
            ty = np.random.uniform(table_y_min, table_y_max)
            tz = TABLE_DIMS_Z + 0.08

            file_path      = object_collision_files[target_file_idx[k]]
            collision_mesh = obj_reader(asset_root + file_path)
            collision_mesh.set_scale(object_scaling_factor[k])
            collision_mesh.add_offset(object_offset[target_file_idx[k]])
            verts, tris = collision_mesh.get_bounding_box_mesh()
            temp_center = collision_mesh.get_center()
            temp_bbox   = collision_mesh.get_bounding_box()

            m = fcl.BVHModel()
            m.beginModel(len(verts), len(tris)); m.addSubModel(verts, tris); m.endModel()
            t_tf = fcl.Transform(np.array([tx, ty, tz]))
            temp_co = fcl.CollisionObject(m, t_tf)

            req   = fcl.CollisionRequest()
            rdata = fcl.CollisionData(request=req)
            objs_manager.collide(temp_co, rdata, fcl.defaultCollisionCallback)
            is_collision = rdata.result.is_collision

            if not is_collision:
                for obj_pos in GT_OBJ_POS_LIST:
                    if np.sqrt((tx - obj_pos[0])**2 + (ty - obj_pos[1])**2) <= 0.16:
                        is_collision = True
                        break

        object_pose   = gymapi.Transform()
        object_pose.p = gymapi.Vec3(tx, ty, tz)

        handle = gym.create_actor(env, object_assets[target_file_idx[k]], object_pose,
                                  f"object{k}", 0, 2**(k+1), k+1)
        gym.set_actor_scale(env, handle, object_scaling_factor[k])
        object_handles.append(handle)
        object_collision_lib.append(m)
        object_status_list.append([temp_center, temp_bbox])
        object_reader_tracker.append(collision_mesh)
        obstacle_objs.append(temp_co)
        GT_OBJ_POS_LIST.append([tx, ty])
        objs_manager.registerObjects(obstacle_objs)
        objs_manager.setup()

    # ── Cameras ────────────────────────────────────────────────────────────
    top_cam_handle  = gym.create_camera_sensor(env, camera_props)
    top_cam_pos     = gymapi.Vec3(table_pose.p.x, table_pose.p.y + 0.001, 2.0)
    top_cam_target  = gymapi.Vec3(table_pose.p.x - 0.5, table_pose.p.y, table_pose.p.z)
    gym.set_camera_location(top_cam_handle, env, top_cam_pos, top_cam_target)

    main_cam_handle = gym.create_camera_sensor(env, camera_props)
    main_cam_pos    = gymapi.Vec3(3, 0, 0.3)
    gym.set_camera_location(main_cam_handle, env, main_cam_pos, camera_focus)

    viewer = None
    if not gym_args.headless:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())

    # Lighting — matches collect script exactly
    gym.set_light_parameters(sim, 0, gymapi.Vec3(0.3,0.3,0.3), gymapi.Vec3(1,1,1), gymapi.Vec3(-1,0,0))
    gym.set_light_parameters(sim, 1, gymapi.Vec3(0.3,0.3,0.3), gymapi.Vec3(1,1,1), gymapi.Vec3( 1,0,0))

    # ── Settle simulation (same 2000 steps as collect, capture mesh at step 999) ──
    original_centers = [s[0].copy() for s in object_status_list]
    real_position    = False

    for t in range(2000):
        if not real_position:
            gym.set_dof_target_position(env, spj,  0)
            gym.set_dof_target_position(env, slj, -math.pi / 2)
            gym.set_dof_target_position(env, ej,   0)
            gym.set_dof_target_position(env, wj1, -math.pi / 2)
            gym.set_dof_target_position(env, wj2,  0)
            gym.set_dof_target_position(env, wj3,  0)
            real_position = True

        if t == 999:
            for ii, element in enumerate(object_handles):
                states      = gym.get_actor_rigid_body_states(env, element, 1)
                rotation    = np.array(np.array(states[0][0][1]).item())
                translation = np.array(np.array(states[0][0][0]).item())
                object_status_list[ii][0] = original_centers[ii] + translation
                r1  = R.from_quat(rotation)
                tf  = fcl.Transform(r1.as_matrix(), translation)
                flex_collision_models.append([fcl.CollisionObject(object_collision_lib[ii], tf), 0])
                tmp = object_reader_tracker[ii]
                tmp.set_offset(translation)
                verts, faces = tmp.get_bounding_box_mesh()
                object_mesh.append([verts, faces])

        gym.simulate(sim); gym.fetch_results(sim, True); gym.step_graphics(sim)
        if viewer is not None:
            gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    # ── Arm home pose before top camera (same as collect_multi_obj_layout_only.py) ──
    for _ in range(START_SETTLE_STEPS):
        gym.set_dof_target_position(env, spj,  HOME_DOF[0])
        gym.set_dof_target_position(env, slj,  HOME_DOF[1])
        gym.set_dof_target_position(env, ej,   HOME_DOF[2])
        gym.set_dof_target_position(env, wj1,  HOME_DOF[3])
        gym.set_dof_target_position(env, wj2,  HOME_DOF[4])
        gym.set_dof_target_position(env, wj3,  HOME_DOF[5])
        gym.simulate(sim); gym.fetch_results(sim, True); gym.step_graphics(sim)
        if viewer is not None:
            gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    # ── Capture top-view image ─────────────────────────────────────────────
    gym.render_all_camera_sensors(sim)
    raw_top  = gym.get_camera_image(sim, env, top_cam_handle, gymapi.IMAGE_COLOR)
    rgba_top = raw_top.reshape(camera_props.height, camera_props.width, 4)
    rgb_top  = rgba_top[..., :3].copy()

    top_view_path = os.path.join(session_dir, "top_view.png")
    cv2.imwrite(top_view_path, cv2.cvtColor(rgb_top, cv2.COLOR_RGB2BGR))
    print(f"Top view saved: {top_view_path}")

    # ── Predict object locations — one prediction per object ──────────────
    true_locations      = []
    predicted_locations = []
    for k in range(NUM_OF_OBJECTS):
        true_xyz = object_status_list[k][0].tolist()
        true_locations.append(true_xyz)

        prompt = f"grasp {object_slot_names[k]}"
        px, py, pz = inferencer.predict(rgb_top, prompt)
        predicted_locations.append([px, py, pz])
        l2 = np.sqrt((px - true_xyz[0])**2 + (py - true_xyz[1])**2 + (pz - true_xyz[2])**2)
        print(f"  [{k}] {object_slot_names[k]:<20s}  true={true_xyz}  pred=[{px:.3f},{py:.3f},{pz:.3f}]  L2={l2:.4f}m")

    with open(os.path.join(session_dir, "object_locations_true.json"), "w") as f:
        json.dump({"objects": object_slot_names, "xyz_m": true_locations}, f, indent=2)
    with open(os.path.join(session_dir, "object_locations_predicted.json"), "w") as f:
        json.dump({"objects": object_slot_names, "xyz_m": predicted_locations}, f, indent=2)

    # ── Grasp planning for target object ──────────────────────────────────
    scene_info = [table_dims.x, table_dims.y, table_dims.z, drawer_height]
    file_path_rac = "./assets/urdf/ur5e/meshes/collision/"
    rac = RC.robot_arm_configuration(
        file_path_rac,
        np.array([ur5e_pose.p.x, ur5e_pose.p.y, ur5e_pose.p.z]),
        scene_info,
    )

    target_idx = args.target_object_idx
    grasp_file = ("./assets/" +
                  "/".join(object_asset_files[target_file_idx[target_idx]].split("/")[:-1]) +
                  "/grasp_dict.npy")
    grasp_data = np.load(grasp_file, allow_pickle=True)
    grasp_list = np.arange(len(grasp_data))
    np.random.shuffle(grasp_list)

    true_xy = np.array(true_locations[target_idx][:2],      dtype=np.float64)
    pred_xy = np.array(predicted_locations[target_idx][:2], dtype=np.float64)

    print("Finding grasp goal with TRUE location...")
    q_goal_true, _, _ = find_grasp_q_goal(
        rac, RC, scene_info, grasp_data, grasp_list,
        true_xy, 0, object_mesh, object_collision_models, plane_obj, get_swept_volume_size,
    )
    print("Finding grasp goal with PREDICTED location...")
    q_goal_pred, _, _ = find_grasp_q_goal(
        rac, RC, scene_info, grasp_data, grasp_list,
        pred_xy, 0, object_mesh, object_collision_models, plane_obj, get_swept_volume_size,
    )

    with open(os.path.join(session_dir, "q_goal_true.json"),      "w") as f:
        json.dump({"joint_rad": None if q_goal_true is None else q_goal_true.tolist()}, f, indent=2)
    with open(os.path.join(session_dir, "q_goal_predicted.json"), "w") as f:
        json.dump({"joint_rad": None if q_goal_pred is None else q_goal_pred.tolist()}, f, indent=2)

    dof_snap     = gym.get_actor_dof_states(env, ur, gymapi.STATE_POS)
    q_start_live = np.array(dof_snap["pos"][:6], dtype=np.float64)

    summary: Dict[str, Any] = {
        "session_dir":            session_dir,
        "objects":                object_slot_names,
        "target_object":          object_slot_names[target_idx],
        "true_locations":         true_locations,
        "predicted_locations":    predicted_locations,
        "q_start_live":           q_start_live.tolist(),
        "q_goal_true_found":      q_goal_true is not None,
        "q_goal_predicted_found": q_goal_pred is not None,
        "videos": {},
    }

    grasp_prompt_target = f"grasp {object_slot_names[target_idx]}"
    summary["objloc_prompt_target"] = grasp_prompt_target

    def _annotate_ntfield_frames(frames_list: List[np.ndarray], playback_label: str) -> None:
        """Draw obj-loc prompt + playback label on each RGB frame before encoding MP4."""
        if not frames_list:
            return
        h, w = frames_list[0].shape[:2]
        wrap_w = max(28, min(96, w // 14))
        line_blocks = [
            textwrap.wrap(f"Prompt: {grasp_prompt_target}", width=wrap_w)
            or [f"Prompt: {grasp_prompt_target}"],
            textwrap.wrap(f"Playback: {playback_label}", width=wrap_w) or [playback_label],
        ]
        lines: List[str] = []
        for block in line_blocks:
            lines.extend(block)

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = float(min(0.9, max(0.5, w / 1400.0)))
        thick = max(1, int(round(scale * 2)))
        line_gap = int(26 * scale + 8)
        y0 = int(22 * scale + 12)

        for fi, fr in enumerate(frames_list):
            bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
            y = y0
            for line in lines:
                for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2)):
                    cv2.putText(
                        bgr,
                        line,
                        (10 + dx, y + dy),
                        font,
                        scale,
                        (0, 0, 0),
                        thick + 2,
                        cv2.LINE_AA,
                    )
                cv2.putText(
                    bgr, line, (10, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA
                )
                y += line_gap
            frames_list[fi] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # ── NTField trajectory videos ──────────────────────────────────────────
    def _run_ntfield_video(q_goal, out_mp4, label):
        if q_goal is None:
            summary["videos"][label] = None
            print(f"  [{label}] No valid grasp goal found, skipping video.")
            return
        reset_arm_to_q(gym, sim, env, ur, spj, slj, ej, wj1, wj2, wj3, viewer, q_start_live, n_steps=200)
        path_raw = ntfield_plan(
            nt_model, q_start_live, q_goal,
            step_size=args.ntfield_step_size,
            max_steps=args.ntfield_max_steps,
            tol=args.ntfield_tol,
            device=ntfield_device_str,
        )
        if not path_raw or len(path_raw) < 2:
            summary["videos"][label] = None
            return
        path    = _path_as_6_list(path_raw)
        frames  = []
        execute_path_and_time(
            gym, sim, env, ur, spj, slj, ej, wj1, wj2, wj3, viewer,
            path, label,
            main_cam_handle=main_cam_handle,
            camera_props=camera_props,
            record_rgb=frames,
            planner_playback=args.planner_playback,
        )
        _annotate_ntfield_frames(frames, label)
        _save_mp4_rgb(frames, out_mp4, fps=args.video_fps)
        summary["videos"][label] = out_mp4
        print(f"  [{label}] Video saved: {out_mp4}")

    _run_ntfield_video(q_goal_pred, os.path.join(session_dir, "ntfield_predicted_goal.mp4"), "predicted_goal")
    _run_ntfield_video(q_goal_true, os.path.join(session_dir, "ntfield_true_goal.mp4"),      "true_goal")

    with open(os.path.join(session_dir, "pipeline_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    gym.destroy_sim(sim)
    if viewer is not None:
        gym.destroy_viewer(viewer)
    os.chdir(_cwd_prev)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()