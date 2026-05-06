#!/usr/bin/env python3
"""
Preprocess multi-object grasp H5 dataset into .pt shard files.

Each H5 file contains N objects with a shared start_image.
Each data point = (start_image, target_object_name, z_goal).

z_goal is computed from the teacher encoder:
    coords = [q_start | q_goal]  (12-dim, normalized)
    _, z_goal = teacher.encode_pair_latents(coords)

q_start is the robot home position [0, -pi/2, 0, -pi/2, 0, 0],
matching the Isaac Gym warm-up pose used during data collection
and NTField planning at inference time.

Output shards are saved as:
    output_dir/shard_0000.pt
    output_dir/shard_0001.pt
    ...

Each shard is a list of dicts:
    {
        "image":       torch.Tensor  [3, H, W]  float32, normalized [0,1]
        "object_name": str
        "z_goal":      torch.Tensor  [z_dim]    float32
        # optional metadata:
        "object_location":  torch.Tensor [3]    float32
        "object_id_folder": str
        "source_file":      str
    }

Manifest saved to output_dir/manifest.pt:
    {
        "num_samples": int,
        "num_shards":  int,
        "shards":      list[str],
        "img_size":    int,
        "z_dim":       int,
        "q_start":     list[float],   # recorded for traceability
    }

python student_model_training/preprocess_dataset/preprocess_multi_configs.py \
  /home/hojinsohn/VLM-NT/multi_obj_dataset \
  --teacher-checkpoint teacher_model.pt \
  --output-dir student_model_training/data/pt_shards_multi
"""

import os
import glob
import argparse
import math
import h5py
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter
from torch import nn
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config (override via CLI flags or edit here)
# ---------------------------------------------------------------------------

IMG_SIZE   = 224
SHARD_SIZE = 256
SCALE      = float(np.pi / 0.5)   # NTField coord normalization (matches planning code)

# FIX: home position must match the Isaac Gym warm-up pose used during
# data collection AND the q_start used by the NTField planner at inference.
# Previously this was torch.zeros, which is [0,0,0,0,0,0] — wrong.
HOME_Q = torch.tensor(
    [0.0, -math.pi / 2, 0.0, -math.pi / 2, 0.0, 0.0],
    dtype=torch.float32,
)

# ---------------------------------------------------------------------------
# Teacher loader
# ---------------------------------------------------------------------------

def load_teacher(
    checkpoint: str,
    device: torch.device,
    data_path: Optional[str] = None,
) -> Tuple[nn.Module, int]:
    from models.metric_arm import model_test_metric as md

    model_path = os.path.dirname(os.path.abspath(checkpoint))
    if data_path is None:
        data_path = os.path.join("datasets", "arm", "UR5_trajectory")

    model = md.Model(model_path, data_path, dim=6, source=[0.0] * 6, device=str(device))
    model.load(checkpoint)
    model.network.eval()
    for p in model.network.parameters():
        p.requires_grad_(False)

    z_dim = 256
    if hasattr(model.network, "encoder") and len(model.network.encoder) > 0:
        lin = model.network.encoder[-1]
        if hasattr(lin, "out_features"):
            z_dim = int(lin.out_features)

    return model.network, z_dim


# ---------------------------------------------------------------------------
# z_goal computation
# ---------------------------------------------------------------------------

def build_coords_batch(
    q_start: torch.Tensor,   # (B, 6)
    q_goal:  torch.Tensor,   # (B, 6)
    normalize: bool,
) -> torch.Tensor:
    """Concatenate start/goal configs into (B, 12) teacher input."""
    if normalize:
        q_start = q_start / SCALE
        q_goal  = q_goal  / SCALE
    return torch.cat([q_start, q_goal], dim=1)


@torch.no_grad()
def compute_z_goals(
    teacher:    nn.Module,
    q_starts:   torch.Tensor,   # (B, 6) CPU
    q_goals:    torch.Tensor,   # (B, 6) CPU
    device:     torch.device,
    normalize:  bool,
) -> torch.Tensor:
    """
    Encode a batch of (q_start, q_goal) pairs through the teacher.
    Returns z_goals: (B, z_dim) on CPU.
    """
    coords = build_coords_batch(q_starts, q_goals, normalize).to(device)
    _, z_goals = teacher.encode_pair_latents(coords)
    return z_goals.cpu()


# ---------------------------------------------------------------------------
# Image helper
# ---------------------------------------------------------------------------

def process_img(img: np.ndarray, img_size: int) -> torch.Tensor:
    """Resize (nearest-neighbor) and convert HWC uint8 -> CHW float32 [0,1]."""
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
    return torch.from_numpy(img).permute(2, 0, 1).float().div(255.0)


# ---------------------------------------------------------------------------
# H5 reader
# ---------------------------------------------------------------------------

def load_h5_datapoints(
    h5_path:          str,
    teacher:          nn.Module,
    device:           torch.device,
    normalize_coords: bool,
    img_size:         int,
    include_metadata: bool,
    q_start_home:     torch.Tensor,   # (6,) — the correct home position
) -> List[Dict]:
    """
    Load one H5 file and return a list of per-object data-point dicts.

    For each of the N objects:
      - q_start = HOME_Q  (robot home position, matching inference)
      - q_goal  = goal_joint_configs[i]
      - z_goal  = teacher.encode_pair_latents(q_start | q_goal)[1]
    """
    with h5py.File(h5_path, "r") as f:
        img_np     = f["start_image"][:]
        img_tensor = process_img(img_np, img_size)

        object_names      = [s.decode() if isinstance(s, bytes) else s
                             for s in f["object_names"][:]]
        goal_configs_np   = f["goal_joint_configs"][:].astype(np.float32)  # (N, 6)
        object_locations  = f["object_locations"][:].astype(np.float32)    # (N, 3)
        object_id_folders = [s.decode() if isinstance(s, bytes) else s
                             for s in f["object_id_folders"][:]]

    num_objects = len(object_names)
    assert goal_configs_np.shape[0] == num_objects, (
        f"Mismatch: {num_objects} names but {goal_configs_np.shape[0]} configs "
        f"in {h5_path}"
    )

    q_goals  = torch.from_numpy(goal_configs_np)               # (N, 6)
    # FIX: broadcast the correct home position across the batch
    # Previously: q_starts = torch.zeros_like(q_goals)  ← was [0,0,0,0,0,0]
    q_starts = q_start_home.unsqueeze(0).expand(num_objects, -1).clone()  # (N, 6)

    z_goals = compute_z_goals(
        teacher, q_starts, q_goals, device, normalize_coords
    )  # (N, z_dim)

    source_file = os.path.basename(h5_path)
    datapoints  = []

    for i in range(num_objects):
        dp = {
            "image":       img_tensor,
            "object_name": object_names[i],
            "z_goal":      z_goals[i],
            "q_goal":      q_goals[i],    # (6,) float32 — raw radians, un-normalized
        }
        if include_metadata:
            dp["object_location"]  = torch.from_numpy(object_locations[i])
            dp["object_id_folder"] = object_id_folders[i]
            dp["source_file"]      = source_file
        datapoints.append(dp)

    return datapoints


# ---------------------------------------------------------------------------
# Shard writer helper
# ---------------------------------------------------------------------------

def chunk(lst: List, size: int):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert multi-object grasp H5 files to .pt shard files with z_goal."
    )
    parser.add_argument(
        "input",
        help="Path to a single .h5 file OR a directory of .h5 files.",
    )
    parser.add_argument(
        "--teacher-checkpoint", "-t",
        required=True,
        help="Path to the teacher model checkpoint (.pt).",
    )
    parser.add_argument(
        "--teacher-data-path",
        default=None,
        help="Optional data path for teacher model init.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="pt_shards",
        help="Directory where shard .pt files will be written (default: pt_shards/).",
    )
    parser.add_argument(
        "--shard-size", "-s",
        type=int,
        default=SHARD_SIZE,
        help=f"Number of data points per shard file (default: {SHARD_SIZE}).",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=IMG_SIZE,
        help=f"Resize images to this square size (default: {IMG_SIZE}).",
    )
    parser.add_argument(
        "--no-normalize-coords",
        action="store_true",
        help="Skip NTField-style coordinate normalization (divide by pi/0.5).",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Exclude object_location, object_id_folder, source_file from output.",
    )
    parser.add_argument(
        "--glob", "-g",
        default="*.h5",
        help="Glob pattern when input is a directory (default: '*.h5').",
    )
    # Allow overriding home position from CLI for flexibility,
    # but the default is the correct Isaac Gym warm-up pose.
    parser.add_argument(
        "--home-q",
        type=float,
        nargs=6,
        default=None,
        metavar=("J0", "J1", "J2", "J3", "J4", "J5"),
        help=(
            "6 joint angles (rad) for the robot home / q_start pose. "
            "Default: [0, -pi/2, 0, -pi/2, 0, 0] (Isaac Gym warm-up pose)."
        ),
    )
    args = parser.parse_args()

    # Resolve q_start home position
    if args.home_q is not None:
        q_start_home = torch.tensor(args.home_q, dtype=torch.float32)
        print(f"Using custom home_q from CLI: {q_start_home.tolist()}")
    else:
        q_start_home = HOME_Q.clone()
        print(f"Using default home_q: {q_start_home.tolist()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading teacher from {args.teacher_checkpoint} ...")
    teacher, z_dim = load_teacher(
        args.teacher_checkpoint, device, args.teacher_data_path
    )
    teacher = teacher.to(device)
    print(f"Teacher loaded. z_goal dim: {z_dim}")

    normalize_coords = not args.no_normalize_coords

    input_path = Path(args.input)
    if input_path.is_file():
        h5_files = [str(input_path)]
    elif input_path.is_dir():
        h5_files = sorted(glob.glob(str(input_path / args.glob)))
        if not h5_files:
            raise FileNotFoundError(
                f"No files matching '{args.glob}' found in {input_path}"
            )
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")

    print(f"Found {len(h5_files)} H5 file(s).")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_datapoints: List[Dict] = []
    skipped = 0

    for idx, h5_path in enumerate(tqdm(h5_files, desc="Files")):
        try:
            dps = load_h5_datapoints(
                h5_path,
                teacher=teacher,
                device=device,
                normalize_coords=normalize_coords,
                img_size=args.img_size,
                include_metadata=not args.no_metadata,
                q_start_home=q_start_home,
            )
            all_datapoints.extend(dps)
            tqdm.write(
                f"  [{idx+1:>4}/{len(h5_files)}] {os.path.basename(h5_path)}"
                f"  → {len(dps)} datapoints  (total: {len(all_datapoints)})"
            )
        except Exception as e:
            skipped += 1
            tqdm.write(f"  [WARN] Skipping {h5_path}: {e}")

    print(f"\nTotal datapoints: {len(all_datapoints)}  |  Skipped files: {skipped}")

    if not all_datapoints:
        raise RuntimeError("No valid datapoints found. Aborting.")

    shards     = list(chunk(all_datapoints, args.shard_size))
    num_shards = len(shards)
    pad        = len(str(num_shards - 1))
    shard_paths: List[str] = []

    for shard_idx, shard_data in enumerate(shards):
        shard_path = output_dir / f"shard_{shard_idx:0{pad}d}.pt"
        torch.save(shard_data, shard_path)
        shard_paths.append(str(shard_path))
        print(f"  Wrote {shard_path}  ({len(shard_data)} datapoints)")

    manifest_path = output_dir / "manifest.pt"
    torch.save(
        {
            "num_samples": len(all_datapoints),
            "num_shards":  num_shards,
            "shards":      shard_paths,
            "img_size":    args.img_size,
            "z_dim":       z_dim,
            "q_start":     q_start_home.tolist(),   # recorded for traceability
        },
        manifest_path,
    )
    print(f"\nManifest saved to {manifest_path}")
    print(f"\nDone. {num_shards} shard(s) in '{output_dir}/'.")
    print("\n--- Dataset summary ---")
    name_counts = Counter(dp["object_name"] for dp in all_datapoints)
    for name, count in sorted(name_counts.items()):
        print(f"  {name:30s}: {count:>5} samples")


if __name__ == "__main__":
    main()