#!/usr/bin/env python3
"""
End-to-end PI-VLA integration (Isaac Gym + NTField + latent goal only).

python "final_integrate/run_integrated_pipeline_latent_multi_obj_check.py" \
  --ntfield_checkpoint "/home/hojinsohn/VLM-NT/PI-VLA/teacher_model.pt" \
  --latent_checkpoint "/home/hojinsohn/VLM-NT/PI-VLA/student_model_training/best_z_goal_model_mdn_mdn_K8_bs256_lr3em4_ep90_20260505_114200.pth" \
  --output_dir "output/manual_check_run" \
  --seed 1002 \
  --ntfield_step_size 0.02 \
  --ntfield_max_steps 200 \
  --ntfield_tol 0.01


Run from PI-VLA root::

  python final_integrate/run_integrated_pipeline_latent_only.py \\
    --ntfield_checkpoint ntrl-demo/Experiments/.../Model_Epoch_*.pt \\
    --latent_checkpoint models/best_z_goal_model_wonorm_mse_cos.pth

If ``--output_dir`` is set, outputs go under ``<output_dir>/<timestamp>/``.
Otherwise under ``output/final_integrate/<timestamp>/``.

After the image→latent run, the script (by default) solves the same grasp
``q_goal`` as ``run_integrated_pipeline_latent.py`` and records two extra
trajectories using the **same** ``--ntfield_delta_clamp_rad`` / refine /
stagnation settings as the predicted-latent planner:

- ``ntfield_trajectory_original_goal.mp4`` — joint-space NTField toward
  ``q_goal`` (same L∞ per-step cap as latent).
- ``ntfield_trajectory_original_goal_latent.mp4`` — latent NTField toward the
  teacher latent ``z(q_start, q_goal)``.

Pass ``--no_ground_truth_ntfield_compare`` to skip grasp + these videos.

On some Linux setups, Isaac Gym + CUDA can segfault during normal interpreter
shutdown after ``destroy_sim``. By default this script calls ``os._exit(0)``
after successful saves to avoid that. Pass ``--no_isaac_hard_exit`` to skip
that (useful for debugging under a debugger).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
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
# Text prompt helpers  (must match train_config_wonorm.py exactly)
# ---------------------------------------------------------------------------

def _tokenize_prompt(text: str) -> list:
    return re.findall(r"[a-z0-9]+", text.lower())


def _encode_prompts(
    prompts: list,
    token_to_id: dict,
    max_len: int,
):
    """Encode a list of prompt strings to a (N, max_len) LongTensor."""
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
# Model definition  — must mirror StudentModelWonorm in train_config_wonorm.py
# ---------------------------------------------------------------------------

def _build_resnet18_backbone():
    """Build ResNet18 backbone across torchvision versions."""
    import torchvision.models as models
    import torch.nn as nn

    try:
        backbone = models.resnet18(weights=None)
    except TypeError:
        backbone = models.resnet18(pretrained=False)
    in_features = backbone.fc.in_features   # 512 for ResNet18
    backbone.fc = nn.Identity()
    return backbone, in_features


def _get_latent_model(output_dim: int, vocab_size: int, text_embed_dim: int = 32):
    """
    Builds StudentModelWonorm: frozen ResNet18 backbone + lightweight adapter
    + text prompt encoder fused into a two-layer MLP head.
    Matches the updated architecture in train_config_wonorm.py.

    Falls back gracefully when vocab_size is None/0 so that legacy
    checkpoints (without a text encoder) still load correctly.
    """
    import torch
    import torch.nn as nn

    backbone, in_features = _build_resnet18_backbone()

    # Freeze entire backbone — matches freeze_backbone=True, unfreeze_layers=()
    for param in backbone.parameters():
        param.requires_grad = False

    class _TextPromptEncoder(nn.Module):
        def __init__(self, vocab_size: int, embed_dim: int):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        def forward(self, token_ids):
            emb = self.embed(token_ids)                         # (B, T, D)
            mask = (token_ids != 0).unsqueeze(-1).float()
            denom = mask.sum(dim=1).clamp_min(1.0)
            return (emb * mask).sum(dim=1) / denom              # (B, D)

    class _StudentHeadWonorm(nn.Module):
        def __init__(self, in_features: int, output_dim: int, dropout_p: float = 0.2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(256, output_dim),
            )

        def forward(self, x):
            return self.net(x)

    class _StudentModelWonorm(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone

            # Adapter: projects frozen backbone output → 256-dim
            self.adapter = nn.Sequential(
                nn.Linear(in_features, 256),    # 512 → 256
                nn.ReLU(),
                nn.Dropout(0.2),
            )

            self.text_encoder = _TextPromptEncoder(
                vocab_size=max(int(vocab_size), 2),
                embed_dim=text_embed_dim,       # 32
            )

            # Matches full_train_multi.py:
            #   self.head = StudentHeadWonorm(...)
            # so checkpoint keys are under "head.net.*".
            self.head = _StudentHeadWonorm(
                in_features=256 + text_embed_dim,
                output_dim=output_dim,
                dropout_p=0.2,
            )
        def forward(self, x, text_tokens):
            with torch.no_grad():               # backbone always frozen at inference
                image_feat = self.backbone(x)   # (B, 512)
            adapted = self.adapter(image_feat)  # (B, 256)
            text_feat = self.text_encoder(text_tokens)  # (B, 32)
            fused = torch.cat([adapted, text_feat], dim=1)  # (B, 288)
            return self.head(fused)             # (B, output_dim)

    return _StudentModelWonorm()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _infer_latent_on_image(
    image: np.ndarray,
    checkpoint_path: str,
    device: str,
    object_name: str = "object",
    latent_goal_true: np.ndarray = None,
) -> np.ndarray:
    """
    Run image→latent inference using the checkpoint.

    Uses the same MDNStudent architecture as training and enforces strict
    checkpoint loading. Any architecture drift fails fast with an error.

    Parameters
    ----------
    image : np.ndarray
        RGB image (H×W×3, uint8).
    checkpoint_path : str
        Path to the ``.pth`` checkpoint.
    device : str
        ``"auto"``, ``"cpu"``, or a CUDA device string such as ``"cuda:0"``.
    object_name : str
        Object name used to build the text prompt (e.g. ``"bleach cleanser"``).
        Defaults to ``"object"`` which matches the fallback used during
        training for dict-shard samples without an ``object_name`` key.
    latent_goal_true : np.ndarray
        True latent goal (256,). If provided, use it instead of the predicted latent goal.
    """
    import torch
    from torchvision import transforms

    dev = torch.device(
        "cuda" if torch.cuda.is_available() and device == "auto" else device
    )
    ckpt = torch.load(os.path.abspath(checkpoint_path), map_location=dev)

    z_dim = ckpt.get("z_dim", 256)
    vocab_size: Optional[int] = ckpt.get("vocab_size", None)
    token_to_id: Optional[dict] = ckpt.get("token_to_id", None)
    max_prompt_len: int = int(ckpt.get("max_prompt_len", 8))

    from student_model_mdn import MDNStudent

    n_components = int(ckpt.get("n_components", 8))
    model = MDNStudent(
        output_dim=z_dim,
        vocab_size=max(int(vocab_size or 0), 2),
        n_components=n_components,
    ).to(dev)
    state_dict = ckpt["model_state_dict"]
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Strict checkpoint load failed for MDNStudent. "
            "Training/integration model architecture mismatch detected."
        ) from exc
    model.eval()

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    x = _process_img(image, 224).unsqueeze(0)
    x = normalize(x).to(dev)

    with torch.no_grad():
        if token_to_id:
            prompt = f"grasp {object_name.strip().lower()}"
            text_tokens = _encode_prompts(
                [prompt], token_to_id, max_prompt_len
            ).to(dev)
        else:
            # Compatibility path: text-conditioned checkpoints that do not
            # include tokenizer metadata. Use an all-padding prompt.
            text_tokens = torch.zeros((1, max_prompt_len), dtype=torch.long, device=dev)
        # pred, _ = model(x, text_tokens, z_goal=None)
        preds = model.get_multiple_latent_predictions(x, text_tokens, num_samples=30)
        # preds: (30, 1, 256)
        if latent_goal_true is not None:
            true_latent_t = torch.as_tensor(
                latent_goal_true, dtype=preds.dtype, device=preds.device
            ).reshape(1, -1)
            dists = ((preds - true_latent_t) ** 2).sum(dim=-1)         # (30, 1)
            best_idx = dists.argmin(dim=0)                         # (1,)
            best_pred = preds[best_idx, torch.arange(preds.size(1))]  # (1, 256)
            pred = best_pred.squeeze(0).cpu().numpy()
        else:
            # Option A: pick the one closest to the mean prediction
            mean_pred = preds.mean(dim=0)                          # (1, 256)
            dists = ((preds - mean_pred) ** 2).sum(dim=-1)         # (30, 1)
            best_idx = dists.argmin(dim=0)                         # (1,)
            best_pred = preds[best_idx, torch.arange(preds.size(1))]  # (1, 256)
            pred = best_pred.squeeze(0).cpu().numpy()

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
        execute_path_and_time,
        get_swept_volume_size,
        reset_arm_to_q,
        _save_mp4_rgb,
        _path_as_6_list,
        sim_dt,
    )
    from trajectory_evaluation.ntfield.eval_trajectory_ntfield import (
        _ModelShim,
        load_network_and_function,
    )
    from final_integrate.run_integrated_pipeline_latent import (
        _compute_z_goal,
        find_grasp_q_goal,
    )
    from planning.gradient_planner_trajectory import SCALE as NTFIELD_SCALE

    parser = argparse.ArgumentParser(description="PI-VLA integration — latent goal only")
    parser.add_argument("--ntfield_checkpoint", type=str, required=True)
    parser.add_argument("--ntfield_experiment_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Base dir; session is <output_dir>/TIMESTAMP (default: output/final_integrate/TIMESTAMP)")
    parser.add_argument("--object_z", type=float, default=0.18)
    parser.add_argument(
        "--target_obj_indices",
        type=str,
        default="1,3,5",
        help=(
            "Comma-separated object indices from object_urdf_grasp.txt. "
            "Default matches multi-object collector: 1,3,5."
        ),
    )
    parser.add_argument(
        "--num_objects",
        type=int,
        default=3,
        help="How many distinct objects to place (default: 3).",
    )
    parser.add_argument("--ox_min", type=float, default=0.42)
    parser.add_argument("--ox_max", type=float, default=0.98)
    parser.add_argument("--oy_min", type=float, default=-0.38)
    parser.add_argument("--oy_max", type=float, default=0.38)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use_viewer", action="store_true")
    parser.add_argument("--ntfield_device", type=str, default="cuda:0")
    parser.add_argument(
        "--physx_cpu",
        action="store_true",
        help="Force PhysX on CPU (avoids some GPU PhysX + PyTorch interaction crashes).",
    )
    parser.add_argument("--latent_checkpoint", type=str, required=True,
                        help="Path to image→latent goal checkpoint (MDNStudent: ResNet18 + text encoder + mixture head, output_dim=z_dim)")
    parser.add_argument("--latent_device", type=str, default="auto")
    # NEW: optional override for the object name used to build the text prompt.
    # When omitted the default "object" prompt is used, which matches the
    # fallback behaviour during training and is always safe to use.
    parser.add_argument(
        "--object_name",
        type=str,
        default=None,
        help='Object name for the text prompt, e.g. "bleach cleanser". '
             'Defaults to "object" (same fallback used during training).',
    )
    parser.add_argument("--ntfield_step_size", type=float, default=0.02)
    parser.add_argument("--ntfield_max_steps", type=int, default=200)
    parser.add_argument("--ntfield_tol", type=float, default=0.01)
    parser.add_argument(
        "--ntfield_delta_clamp_rad",
        type=float,
        default=0.0,
        help="If >0, cap max per-joint step (rad) per iteration; step direction is preserved "
        "(L∞ scaling). Elementwise clamp skews the gradient and often stalls short of the goal.",
    )
    parser.add_argument(
        "--ntfield_refine_max_steps",
        type=int,
        default=-1,
        help="Unclamped refinement iterations after the main loop. Default -1: use 100 if "
        "delta_clamp_rad>0 else 0. Set 0 to disable.",
    )
    parser.add_argument(
        "--ntfield_refine_step_size",
        type=float,
        default=None,
        help="Absolute step size during refinement. Overrides --ntfield_refine_step_size_factor.",
    )
    parser.add_argument(
        "--ntfield_refine_step_size_factor",
        type=float,
        default=None,
        help="If set and --ntfield_refine_step_size is omitted, refine step = "
        "ntfield_step_size * this factor (e.g. 1.0 matches main, 1.5 larger). "
        "Default when both omitted: 0.5 * ntfield_step_size.",
    )
    parser.add_argument(
        "--ntfield_refine_delta_clamp_rad",
        type=float,
        default=None,
        help="If set (>0), L∞ per-joint step cap (rad) during refine only; main phase still uses "
        "--ntfield_delta_clamp_rad. Omit or 0: refine is unclamped (previous behavior). "
        "Use a value larger than main to allow bigger steps only after the clamped phase.",
    )
    parser.add_argument(
        "--ntfield_stagnate_max_steps",
        type=int,
        default=400,
        help="After clamp+refine, extra unclamped steps until latent dist stagnates (0=off).",
    )
    parser.add_argument(
        "--ntfield_stagnate_patience",
        type=int,
        default=30,
        help="Stop stagnation phase if relative improvement in latent dist is below threshold "
        "for this many consecutive steps.",
    )
    parser.add_argument(
        "--ntfield_stagnate_rel_eps",
        type=float,
        default=5e-4,
        help="Relative improvement threshold for stagnation (fraction of best-so-far dist).",
    )
    parser.add_argument(
        "--ntfield_stagnate_step_size",
        type=float,
        default=None,
        help="Step size in stagnation phase (default: 0.25 * ntfield_step_size).",
    )
    parser.add_argument("--video_fps", type=float, default=60.0)
    parser.add_argument(
        "--planner_playback",
        type=str,
        choices=("direct", "settle"),
        default="direct",
    )
    parser.add_argument(
        "--no_isaac_hard_exit",
        action="store_true",
        help="Do not os._exit(0) after success (may segfault on Isaac Gym teardown).",
    )
    parser.add_argument(
        "--no_ground_truth_ntfield_compare",
        action="store_true",
        help="Skip TracIK/RRT grasp solve and the original_goal / original_goal_latent NTField videos.",
    )
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
            f"Need at least {num_objects} indices in --target_obj_indices, got {len(target_obj_index)}"
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        session_dir = os.path.join(os.path.abspath(args.output_dir), stamp)
    else:
        session_dir = os.path.join(_PI_VLA_ROOT, "output", "final_integrate", stamp)
    os.makedirs(session_dir, exist_ok=True)

    oz = float(args.object_z)

    ckpt_abs = _resolve_under_root(args.ntfield_checkpoint)
    if not os.path.isfile(ckpt_abs):
        raise SystemExit(f"NTField checkpoint not found: {ckpt_abs}")

    _cwd_prev = os.getcwd()
    os.chdir(HANWEN_GRASPING_ROOT)

    # ── Isaac Gym setup ──────────────────────────────────────────────────────
    gym = gymapi.acquire_gym()
    gym_args = gymutil.parse_arguments(description="final_integrate", headless=True, custom_parameters=[])
    gym_args.headless = not args.use_viewer
    if args.physx_cpu:
        gym_args.use_gpu = False

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

    # Load NTField after PhysX/GPU sim exists. Initializing a large CUDA model
    # before create_sim can segfault or OOM on some Linux + GPU PhysX setups.
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
    table_x_min = table_pose.p.x - table_dims.x * 0.5 + 0.05
    table_x_max = table_pose.p.x + table_dims.x * 0.5 - 0.10
    table_y_min = table_pose.p.y - table_dims.y * 0.5 + 0.10
    table_y_max = table_pose.p.y + table_dims.y * 0.5 - 0.20
    plane_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    col_plane = fcl.Plane(plane_normal, 0.0)
    plane_obj = fcl.CollisionObject(col_plane, fcl.Transform())

    drawer_height = DRAWER_HEIGHT
    object_mesh: List[Any] = []
    flex_collision_models: List[Any] = []

    envs: List[Any] = []
    ur5e_handles: List[Any] = []
    object_handles: List[Any] = []
    object_status_list: List[Any] = []
    object_reader_tracker: List[Any] = []
    object_collision_lib: List[Any] = []
    placed_object_locations: List[List[float]] = []
    spj = slj = ej = wj1 = wj2 = wj3 = None
    if len(target_obj_index) < num_objects:
        raise RuntimeError(
            f"Need at least {num_objects} entries in target object list, got {len(target_obj_index)}"
        )
    target_file_idx = np.random.choice(target_obj_index, num_objects, replace=False)
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
        gt_obj_pos_list: List[List[float]] = []
        gt_target_pos = [
            float(np.random.uniform(max(table_x_min, 0.20 + table_dims.x / 2), table_x_max)),
            float(np.random.uniform(table_y_min, table_y_max)),
            float(oz),
        ]

        object_scaling_factor = np.ones(num_objects, dtype=np.float64)

        for k in range(num_objects):
            object_pose = gymapi.Transform()
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
            is_collision = True
            tx = ty = 0.0
            while is_collision:
                tx = float(np.random.uniform(table_x_min, table_x_max))
                ty = float(np.random.uniform(table_y_min, table_y_max))
                t = fcl.Transform(np.array([tx, ty, oz]))

                req = fcl.CollisionRequest()
                rdata = fcl.CollisionData(request=req)
                objs_manager.collide(
                    fcl.CollisionObject(m, t), rdata, fcl.defaultCollisionCallback
                )
                is_collision = rdata.result.is_collision

                if not is_collision:
                    dist_target = float(
                        np.sqrt((tx - gt_target_pos[0]) ** 2 + (ty - gt_target_pos[1]) ** 2)
                    )
                    if dist_target <= 0.2:
                        is_collision = True
                        continue
                    for obj_xy in gt_obj_pos_list:
                        dist_obj = float(np.sqrt((tx - obj_xy[0]) ** 2 + (ty - obj_xy[1]) ** 2))
                        if dist_obj <= 0.16:
                            is_collision = True
                            break

            object_pose.p = gymapi.Vec3(tx, ty, oz)
            gt_obj_pos_list.append([tx, ty])
            placed_object_locations.append([tx, ty, float(oz)])
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

    # ── Warm-up simulation (match latent pipeline: mesh state at t==999 for grasp/RRT) ──
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

    # ── Match collect_multi_obj_data_for_student.py: HOME_DOF settle before
    # start_image — same targets and step count so q_start matches training.
    _HOME_DOF = [0.7, -2.0, 2.5, -0.3, 0.7, 0.0]
    _START_SETTLE_STEPS = 30
    for _ in range(_START_SETTLE_STEPS):
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

    # ── Capture top-view image ───────────────────────────────────────────────
    assert top_cam_handle is not None
    gym.render_all_camera_sensors(sim)
    raw_top = gym.get_camera_image(sim, env, top_cam_handle, gymapi.IMAGE_COLOR)
    rgba_top = raw_top.reshape(camera_props.height, camera_props.width, 4)
    rgb_top = rgba_top[..., :3].copy()

    top_view_path = os.path.join(session_dir, "top_view.png")
    cv2.imwrite(top_view_path, cv2.cvtColor(rgb_top, cv2.COLOR_RGB2BGR))

    object_location = placed_object_locations[0] if placed_object_locations else [0.0, 0.0, float(oz)]
    with open(os.path.join(session_dir, "object_location.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "xyz_m": object_location,
                "xyz_m_all": placed_object_locations,
            },
            f,
            indent=2,
        )

    # ── Determine the object name for the text prompt ────────────────────────
    # Use --object_name if provided; otherwise derive a best-effort name from
    # the YCB asset path for the target object (e.g. "003_cracker_box" →
    # "cracker box").  Falls back to the neutral "object" string which matches
    # the training-time default for unlabelled dict-shard samples.
    if args.object_name:
        infer_object_name = args.object_name.strip()
    else:
        try:
            raw_fname = object_asset_files[target_file_idx[0]]
            # YCB names look like "003_cracker_box/..." — strip the numeric prefix
            stem = raw_fname.split("/")[-2] if "/" in raw_fname else raw_fname
            stem = re.sub(r"^\d+_", "", stem)          # remove leading digits+underscore
            infer_object_name = stem.replace("_", " ")
        except Exception:
            infer_object_name = "object"
    print(f"[latent] Using object name for text prompt: '{infer_object_name}'", flush=True)

    # ── Get live q_start ─────────────────────────────────────────────────────
    dof_snapshot = gym.get_actor_dof_states(env, ur, gymapi.STATE_POS)
    q_start_live = np.array(dof_snapshot["pos"][:6], dtype=np.float64)

    # ── Grasp joint goal (same as run_integrated_pipeline_latent.py) ─────────
    q_goal_true: Optional[np.ndarray] = None
    rrt_path_true: Optional[List[np.ndarray]] = None
    rrt_plan_s: Optional[float] = None
    if not args.no_ground_truth_ntfield_compare:
        if not object_mesh:
            print(
                "[warn] object_mesh empty; skip ground-truth grasp and "
                "original_goal / original_goal_latent NTField videos."
            )
        else:
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
            target_idx = 0
            q_goal_true, rrt_path_true, rrt_plan_s = find_grasp_q_goal(
                rac,
                RC,
                scene_info,
                grasp_data,
                grasp_list,
                true_xy,
                target_idx,
                object_mesh,
                object_collision_models,
                plane_obj,
                get_swept_volume_size,
            )
            with open(
                os.path.join(session_dir, "q_goal_original.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    {"joint_rad": None if q_goal_true is None else q_goal_true.tolist()},
                    f,
                    indent=2,
                )

    print(f"[latent] q_goal_true: {q_goal_true}", flush=True)

    latent_goal_true: Optional[np.ndarray] = None
    if q_goal_true is not None:
        latent_goal_true = _compute_z_goal(
            nt_net,
            q_start_live.reshape(1, -1),
            q_goal_true.reshape(1, -1),
            dev_nt,
        )

    # ── Infer latent goal from image + teacher latent at true grasp (if any) ─
    latent_goal_pred = _infer_latent_on_image(
        rgb_top,
        _resolve_under_root(args.latent_checkpoint),
        args.latent_device,
        object_name=infer_object_name,    # <-- new: pass object name
        latent_goal_true=latent_goal_true,
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
    if args.ntfield_refine_delta_clamp_rad is not None and float(args.ntfield_refine_delta_clamp_rad) > 0.0:
        refine_delta_clamp_rad = float(args.ntfield_refine_delta_clamp_rad)
    stagnate_step = (
        float(args.ntfield_stagnate_step_size)
        if args.ntfield_stagnate_step_size is not None
        else float(args.ntfield_step_size) * 0.25
    )

    true_latent_error: Optional[Dict[str, float]] = None
    latent_diff_vec: Optional[np.ndarray] = None
    if latent_goal_true is not None:
        latent_diff = latent_goal_pred - latent_goal_true
        latent_diff_vec = latent_diff
        true_latent_error = {
            "mse": float(np.mean(np.square(latent_diff))),
            "l2": float(np.linalg.norm(latent_diff)),
            "max_abs": float(np.max(np.abs(latent_diff))),
        }
    latent_goal_pred_path = os.path.join(session_dir, "latent_goal_predicted.json")
    with open(latent_goal_pred_path, "w", encoding="utf-8") as f:
        json.dump({"latent_goal_predicted": np.asarray(latent_goal_pred).reshape(-1).tolist()}, f, indent=2)
    latent_goal_true_path: Optional[str] = None
    if latent_goal_true is not None:
        latent_goal_true_path = os.path.join(session_dir, "latent_goal_true.json")
        with open(latent_goal_true_path, "w", encoding="utf-8") as f:
            json.dump({"latent_goal_true": np.asarray(latent_goal_true).reshape(-1).tolist()}, f, indent=2)
    latent_goal_diff_path: Optional[str] = None
    if latent_diff_vec is not None:
        latent_goal_diff_path = os.path.join(session_dir, "latent_goal_diff.json")
        with open(latent_goal_diff_path, "w", encoding="utf-8") as f:
            json.dump(
                {"latent_goal_diff": np.asarray(latent_diff_vec).reshape(-1).tolist()},
                f,
                indent=2,
            )
    prompt_text = f"grasp {infer_object_name.strip().lower()}"
    summary: Dict[str, Any] = {
        "session_dir": session_dir,
        "prompt": prompt_text,
        "predicted_video": None,
        "rrt_video": None,
        "rrt_plan_s": rrt_plan_s,
        "latent_goal_predicted_path": latent_goal_pred_path,
        "latent_goal_true_path": latent_goal_true_path,
        "latent_goal_diff_path": latent_goal_diff_path,
        "true_latent_error": true_latent_error,
        "status": "Failure",
        "planner_stop_reason": None,
        "ntfield_final_latent_dist": None,
    }

    # ── Gradient planner using predicted latent goal ─────────────────────────
    def ntfield_plan_gradient_with_goal_latent(
        teacher_network, q_start: np.ndarray, z_goal_hat: np.ndarray,
        step_size: float = 0.02, max_steps: int = 200,
        tol: float = 0.01, device: str = "cuda",
        delta_clamp_rad: float = 0.0,
        refine_max_steps: int = 0,
        refine_step_size: float = 0.01,
        refine_delta_clamp_rad: Optional[float] = None,
        stagnate_max_steps: int = 0,
        stagnate_patience: int = 30,
        stagnate_rel_eps: float = 5e-4,
        stagnate_step_size: float = 0.005,
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        import torch

        def _latent_dist(qn: torch.Tensor, zg: torch.Tensor) -> float:
            with torch.no_grad():
                d, _, _ = teacher_network.out_with_goal_latent(qn.detach(), zg)
                return float(d.item())

        def _grad_step(
            q_t: torch.Tensor, zg: torch.Tensor, stp: float, clamp_rad: float
        ) -> tuple:
            """Returns (q_out, dist_after, converged). clamp_rad 0 = no clamp."""
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
            converged = d_post < tol
            return q_next.detach(), d_post, converged

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
        final_dist: Optional[float] = None
        converged = False
        stopped = "max_main_steps"

        main_clamp = float(delta_clamp_rad) if delta_clamp_rad > 0.0 else 0.0
        for _ in range(max_steps):
            q_curr_t, final_dist, converged = _grad_step(
                q_curr_t,
                z_goal_hat,
                step_size,
                main_clamp,
            )
            if converged:
                path_norm.append(q_curr_t.detach().cpu().numpy()[0].copy())
                stopped = "latent_tol_main"
                break
            path_norm.append(q_curr_t.detach().cpu().numpy()[0].copy())

        refine_clamp = (
            float(refine_delta_clamp_rad)
            if refine_delta_clamp_rad is not None and refine_delta_clamp_rad > 0.0
            else 0.0
        )
        if refine_max_steps > 0 and not converged:
            for _ in range(refine_max_steps):
                q_curr_t, final_dist, converged = _grad_step(
                    q_curr_t, z_goal_hat, refine_step_size, refine_clamp
                )
                if converged:
                    path_norm.append(q_curr_t.detach().cpu().numpy()[0].copy())
                    stopped = "latent_tol_refine"
                    break
                path_norm.append(q_curr_t.detach().cpu().numpy()[0].copy())
            else:
                if not converged:
                    stopped = "max_refine_steps"

        if stagnate_max_steps > 0 and not converged:
            best_d = final_dist if final_dist is not None else _latent_dist(q_curr_t, z_goal_hat)
            stall = 0
            for _ in range(stagnate_max_steps):
                q_curr_t, final_dist, converged = _grad_step(
                    q_curr_t,
                    z_goal_hat,
                    stagnate_step_size,
                    0.0,
                )
                if converged:
                    path_norm.append(q_curr_t.detach().cpu().numpy()[0].copy())
                    stopped = "latent_tol_stagnate"
                    break
                path_norm.append(q_curr_t.detach().cpu().numpy()[0].copy())
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

        meta = {"final_latent_dist": final_dist, "stopped": stopped}
        return [p * NTFIELD_SCALE for p in path_norm], meta

    def ntfield_plan_joint_gradient(
        model,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        step_size: float = 0.02,
        max_steps: int = 200,
        tol: float = 0.01,
        device: str = "cuda",
        delta_clamp_rad: float = 0.0,
        refine_max_steps: int = 0,
        refine_step_size: float = 0.01,
        refine_delta_clamp_rad: Optional[float] = None,
        stagnate_max_steps: int = 0,
        stagnate_patience: int = 30,
        stagnate_rel_eps: float = 5e-4,
        stagnate_step_size: float = 0.005,
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """Joint-space NTField gradient steps with the same L∞ clamp schedule as latent."""
        import torch

        dim = 6
        q_start = np.asarray(q_start, dtype=np.float64).reshape(dim)
        q_goal = np.asarray(q_goal, dtype=np.float64).reshape(dim)
        q_start_norm = (q_start / NTFIELD_SCALE).astype(np.float32)
        q_goal_norm = (q_goal / NTFIELD_SCALE).astype(np.float32)
        q_goal_t = torch.tensor(q_goal_norm, dtype=torch.float32, device=device).view(1, dim)

        def joint_dist_norm(q_batch: torch.Tensor) -> float:
            return float(torch.norm(q_batch - q_goal_t).item())

        def joint_grad_step(
            q_batch: torch.Tensor, stp: float, clamp_rad: float
        ) -> Tuple[torch.Tensor, float, bool]:
            XP = torch.cat([q_batch, q_goal_t], dim=1)
            XP = XP.detach().requires_grad_(True)
            d_pre = joint_dist_norm(XP[:, :dim].detach())
            if d_pre < tol:
                return XP[:, :dim].detach(), d_pre, True
            grad = model.function.Gradient(XP)
            with torch.no_grad():
                delta = stp * grad[:, :dim].detach()
                if clamp_rad > 0.0:
                    cap_norm = float(clamp_rad) / float(NTFIELD_SCALE)
                    peak = torch.amax(torch.abs(delta))
                    if float(peak.item()) > cap_norm + 1e-12:
                        delta = delta * (cap_norm / (peak + 1e-12))
                q_next = XP[:, :dim].detach() + delta
            d_post = joint_dist_norm(q_next)
            return q_next, d_post, d_post < tol

        q_curr = torch.tensor(
            q_start_norm.reshape(1, dim), dtype=torch.float32, device=device
        )
        path_norm = [q_start_norm.copy()]
        final_dist = joint_dist_norm(q_curr)
        converged = False
        stopped = "max_main_steps"

        main_clamp_j = float(delta_clamp_rad) if delta_clamp_rad > 0.0 else 0.0
        for _ in range(max_steps):
            q_curr, final_dist, converged = joint_grad_step(
                q_curr, step_size, main_clamp_j
            )
            if converged:
                path_norm.append(q_curr.detach().cpu().numpy()[0].copy())
                stopped = "joint_tol_main"
                break
            path_norm.append(q_curr.detach().cpu().numpy()[0].copy())

        refine_clamp_j = (
            float(refine_delta_clamp_rad)
            if refine_delta_clamp_rad is not None and refine_delta_clamp_rad > 0.0
            else 0.0
        )
        if refine_max_steps > 0 and not converged:
            for _ in range(refine_max_steps):
                q_curr, final_dist, converged = joint_grad_step(
                    q_curr, refine_step_size, refine_clamp_j
                )
                if converged:
                    path_norm.append(q_curr.detach().cpu().numpy()[0].copy())
                    stopped = "joint_tol_refine"
                    break
                path_norm.append(q_curr.detach().cpu().numpy()[0].copy())
            else:
                if not converged:
                    stopped = "max_refine_steps"

        if stagnate_max_steps > 0 and not converged:
            best_d = final_dist
            stall = 0
            for _ in range(stagnate_max_steps):
                q_curr, final_dist, converged = joint_grad_step(
                    q_curr, stagnate_step_size, 0.0
                )
                if converged:
                    path_norm.append(q_curr.detach().cpu().numpy()[0].copy())
                    stopped = "joint_tol_stagnate"
                    break
                path_norm.append(q_curr.detach().cpu().numpy()[0].copy())
                imp = (best_d - final_dist) / max(best_d, 1e-8)
                if imp > stagnate_rel_eps:
                    best_d = min(best_d, final_dist)
                    stall = 0
                else:
                    stall += 1
                    if stall >= stagnate_patience:
                        stopped = "joint_dist_stagnated"
                        break
            else:
                if not converged and stopped != "joint_tol_stagnate":
                    stopped = "max_stagnate_steps"

        if len(path_norm) < 2:
            path_norm.append(q_curr.detach().cpu().numpy()[0].copy())

        meta = {
            "final_joint_err_norm": final_dist,
            "final_joint_err_rad": float(final_dist * NTFIELD_SCALE),
            "stopped": stopped,
        }
        return [p * NTFIELD_SCALE for p in path_norm], meta

    def _path_end_joint_err_vs_goal(
        path_raw: List[np.ndarray], q_goal: np.ndarray
    ) -> Dict[str, float]:
        """Planner endpoint vs grasp joints (latent descent does not optimize this)."""
        q_end = np.asarray(path_raw[-1], dtype=np.float64).reshape(6)
        qg = np.asarray(q_goal, dtype=np.float64).reshape(6)
        d = q_end - qg
        # Shortest revolute error per joint (raw diff can be >> pi and inflate L2).
        d_wrapped = np.arctan2(np.sin(d), np.cos(d))
        return {
            "joint_err_l2_rad": float(np.linalg.norm(d)),
            "joint_err_linf_rad": float(np.max(np.abs(d))),
            "joint_err_l2_wrapped_rad": float(np.linalg.norm(d_wrapped)),
            "joint_err_linf_wrapped_rad": float(np.max(np.abs(d_wrapped))),
        }

    def _record_ntfield_path(
        path_raw: List[np.ndarray],
        label: str,
        mp4_path: str,
        summary_label: str,
    ) -> None:
        if path_raw and len(path_raw) >= 2:
            reset_arm_to_q(
                gym, sim, env, ur, spj, slj, ej, wj1, wj2, wj3, viewer, q_start_live, n_steps=200
            )
            path_nt = _path_as_6_list(path_raw)
            frames_nt: List[np.ndarray] = []
            execute_path_and_time(
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
                path_nt,
                label,
                main_cam_handle=main_cam_handle,
                camera_props=camera_props,
                record_rgb=frames_nt,
                planner_playback=args.planner_playback,
            )
            _save_mp4_rgb(frames_nt, mp4_path, fps=args.video_fps)
            if summary_label == "predicted_latent_goal":
                summary["predicted_video"] = mp4_path
        else:
            print(f"[warn] NTField planner returned an empty path ({summary_label}).")
            if summary_label == "predicted_latent_goal":
                summary["predicted_video"] = None

    def _record_rrt_path(
        path_raw: Optional[List[np.ndarray]],
        label: str,
        mp4_path: str,
    ) -> None:
        if path_raw and len(path_raw) >= 2:
            reset_arm_to_q(
                gym, sim, env, ur, spj, slj, ej, wj1, wj2, wj3, viewer, q_start_live, n_steps=200
            )
            path_rrt = _path_as_6_list(path_raw)
            frames_rrt: List[np.ndarray] = []
            execute_path_and_time(
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
                path_rrt,
                label,
                main_cam_handle=main_cam_handle,
                camera_props=camera_props,
                record_rgb=frames_rrt,
                planner_playback=args.planner_playback,
            )
            _save_mp4_rgb(frames_rrt, mp4_path, fps=args.video_fps)
            summary["rrt_video"] = mp4_path
        else:
            print("[warn] RRT planner returned an empty path (original_goal_rrt).")
            summary["rrt_video"] = None

    # ── Predicted latent goal (image) ────────────────────────────────────────
    path_pred, meta_pred = ntfield_plan_gradient_with_goal_latent(
        nt_net,
        q_start_live.reshape(1, -1),
        latent_goal_pred.reshape(1, -1),
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
    summary["ntfield_final_latent_dist"] = meta_pred["final_latent_dist"]
    summary["planner_stop_reason"] = meta_pred["stopped"]
    if isinstance(meta_pred["stopped"], str) and meta_pred["stopped"].startswith("latent_tol"):
        summary["status"] = "Success"

    mp4_path = os.path.join(session_dir, "ntfield_trajectory_predicted_latent_goal.mp4")
    _record_ntfield_path(path_pred, "predicted_latent_goal", mp4_path, "predicted_latent_goal")

    # ── RRT trajectory to the true grasp goal (if available) ─────────────────
    if q_goal_true is not None:
        rrt_mp4_path = os.path.join(session_dir, "rrt_trajectory_original_goal.mp4")
        _record_rrt_path(rrt_path_true, "original_goal_rrt", rrt_mp4_path)

    # ── True latent goal (teacher z(q_start, q_goal_true)) ───────────────────
    if latent_goal_true is not None:
        latent_goal_true_noisy = latent_goal_true + np.random.normal(0, 0.0001, latent_goal_true.shape)
        l2_dist_noisy = np.linalg.norm(latent_goal_true_noisy - latent_goal_true)
        linf_dist_noisy = np.max(np.abs(latent_goal_true_noisy - latent_goal_true))
        print(f"L2 distance between noisy and true latent goal: {l2_dist_noisy}")
        print(f"Linf distance between noisy and true latent goal: {linf_dist_noisy}")
        path_true_latent, meta_true_latent = ntfield_plan_gradient_with_goal_latent(
            nt_net,
            q_start_live.reshape(1, -1),
            latent_goal_true_noisy.reshape(1, -1),
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
        mp4_path = os.path.join(session_dir, "ntfield_trajectory_true_latent_goal.mp4")
        _record_ntfield_path(path_true_latent, "true_latent_goal", mp4_path, "true_latent_goal")

    # ── Save outputs ─────────────────────────────────────────────────────────
    with open(os.path.join(session_dir, "pipeline_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Tear down graphics before sim (matches hanwen_grasping collect scripts).
    if viewer is not None:
        gym.destroy_viewer(viewer)
        viewer = None
    gym.destroy_sim(sim)
    sim = None

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