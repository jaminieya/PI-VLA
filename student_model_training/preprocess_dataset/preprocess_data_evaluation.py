#!/usr/bin/env python3
"""
Convert grasp_multi3_*.h5 to a fixed test dataset for model evaluation.

Each shard is a list[dict] containing:
  - "image"                (3, H, W) float32 in [0, 1]  — top-view RGB
  - "object_name"          str                           — target object name
  - "object_location"      [x, y, z] float              — target object XYZ
  - "all_object_locations" [[x,y,z], ...]               — all 3 objects' XYZ
  - "all_object_names"     [str, ...]                   — all 3 object names
  - "all_object_ids"       [int, ...]                   — YCB asset indices
  - "z_goal"               (z_dim,) float32             — teacher latent goal
  - "q_goal"               (6,) float32                 — goal joints (radians)
  - "q_start"              (6,) float32                 — start joints (radians)
  - "seed"                 int                          — scene seed
  - "source_file"          str                          — source H5 filename

The output is intentionally NOT shuffled and NOT split into train/val —
it is a fixed held-out test set. Feed it directly to evaluate_test_dataset.py.

Example:
  python preprocess_data_evaluation.py \
    --input /home/hojinsohn/VLM-NT/PI-VLA/hanwen_grasping/output/multi_obj/test_run \
    --teacher-checkpoint /home/hojinsohn/VLM-NT/PI-VLA/teacher_model.pt \
    --output-dir /home/hojinsohn/VLM-NT/PI-VLA/hanwen_grasping/output/multi_obj/test_run_pt_shards \
    --glob 'grasp_multi3_*.h5'
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_ST_ROOT    = _SCRIPT_DIR.parent
_PI_VLA_ROOT = _ST_ROOT.parent
for _p in (_SCRIPT_DIR, _ST_ROOT, _PI_VLA_ROOT, _PI_VLA_ROOT / "ntrl-demo"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from preprocess_multi_configs import (   # noqa: E402
    HOME_Q,
    IMG_SIZE,
    SHARD_SIZE,
    load_teacher,
)


# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_INPUT  = "/home/hojinsohn/VLM-NT/PI-VLA/hanwen_grasping/output/multi_obj/test_scenes"
DEFAULT_OUTPUT = "/home/hojinsohn/VLM-NT/PI-VLA/student_model_training/data/test_dataset"


# ── H5 reading ────────────────────────────────────────────────────────────────

def _read_h5_scene(h5_path: str) -> Optional[Dict[str, Any]]:
    """
    Read one H5 file written by collect_multi_obj_data_for_student.py.

    Actual H5 layout:
      start_image          uint8 (H, W, 3)
      object_names         (3,)  bytes/str  — all 3 object names
      object_locations     (3, 3) float     — all 3 object XYZ
      goal_joint_configs   (3, 6) float     — one q_goal per object
      seed                 int scalar       — optional
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required: pip install h5py")

    try:
        with h5py.File(h5_path, "r") as f:
            keys = set(f.keys())

            # ── Debug: print keys on first call ───────────────────────────
            # Uncomment if issues persist:
            # print(f"  [debug] keys in {os.path.basename(h5_path)}: {sorted(keys)}")

            # ── Image ─────────────────────────────────────────────────────
            for img_key in ("start_image", "top_view_image", "image_top", "image"):
                if img_key in keys:
                    img_raw = f[img_key][()]   # (H, W, 3) uint8
                    break
            else:
                tqdm.write(f"  [SKIP] {os.path.basename(h5_path)}: no image key found in {sorted(keys)}")
                return None

            # ── goal_joint_configs: (3, 6) — one q_goal per object ────────
            for qg_key in ("goal_joint_configs", "q_goal", "q_goals"):
                if qg_key in keys:
                    goal_configs = f[qg_key][()].astype(np.float32)  # (3, 6) or (6,)
                    break
            else:
                tqdm.write(f"  [SKIP] {os.path.basename(h5_path)}: no q_goal key found")
                return None

            # Normalise to (N, 6) regardless of whether it was (6,) or (N, 6)
            if goal_configs.ndim == 1:
                goal_configs = goal_configs.reshape(1, 6)   # legacy single-object

            # ── object_locations: (3, 3) ──────────────────────────────────
            for loc_key in ("object_locations", "all_object_locations", "object_location"):
                if loc_key in keys:
                    all_locs = f[loc_key][()].astype(np.float64)  # (3, 3) or (3,)
                    break
            else:
                all_locs = np.zeros((goal_configs.shape[0], 3), dtype=np.float64)

            if all_locs.ndim == 1:
                all_locs = all_locs.reshape(1, 3)

            # ── object_names: (3,) bytes ──────────────────────────────────
            for nm_key in ("object_names", "all_object_names", "object_name"):
                if nm_key in keys:
                    raw_names = f[nm_key][()]
                    if isinstance(raw_names, (bytes, str, np.bytes_)):
                        # scalar — single object
                        name_str = raw_names.decode("utf-8") if isinstance(raw_names, (bytes, np.bytes_)) else str(raw_names)
                        all_object_names = [name_str]
                    else:
                        all_object_names = [
                            n.decode("utf-8") if isinstance(n, (bytes, np.bytes_)) else str(n)
                            for n in raw_names
                        ]
                    break
            else:
                all_object_names = ["object"] * goal_configs.shape[0]

            # ── object IDs ────────────────────────────────────────────────
            for id_key in ("all_object_ids", "object_ids", "object_id"):
                if id_key in keys:
                    all_object_ids = f[id_key][()].astype(int).tolist()
                    if not isinstance(all_object_ids, list):
                        all_object_ids = [all_object_ids]
                    break
            else:
                all_object_ids = list(range(goal_configs.shape[0]))

            # ── q_start ───────────────────────────────────────────────────
            for qs_key in ("q_start", "start_joints", "start_joint_configs"):
                if qs_key in keys:
                    q_start = f[qs_key][()].astype(np.float32).reshape(6)
                    break
            else:
                q_start = None   # will fall back to HOME_Q in main()

            # ── seed ──────────────────────────────────────────────────────
            seed = int(f["seed"][()]) if "seed" in keys else -1

    except Exception as e:
        tqdm.write(f"  [SKIP] {os.path.basename(h5_path)}: {e}")
        return None

    n_objects = goal_configs.shape[0]

    # Pad shorter arrays to match n_objects so zip() never misaligns
    while len(all_object_names) < n_objects:
        all_object_names.append("object")
    while len(all_object_ids) < n_objects:
        all_object_ids.append(-1)
    if all_locs.shape[0] < n_objects:
        pad = np.zeros((n_objects - all_locs.shape[0], 3), dtype=np.float64)
        all_locs = np.vstack([all_locs, pad])

    return {
        "img_raw":              img_raw,
        "q_goals":              goal_configs,            # (N, 6) — one per object
        "q_start":              q_start,
        "all_object_names":     all_object_names,        # [str, ...]  len==N
        "all_object_locations": all_locs.tolist(),       # [[x,y,z]]  len==N
        "all_object_ids":       all_object_ids,          # [int, ...]  len==N
        "seed":                 seed,
        "source_file":          os.path.basename(h5_path),
    }

# ── Image processing ──────────────────────────────────────────────────────────

def _process_image(img_raw: np.ndarray, img_size: int) -> torch.Tensor:
    """
    Convert raw uint8 HWC image to (3, H, W) float32 tensor in [0, 1].
    Handles RGBA by dropping the alpha channel.
    Resizes with nearest-neighbour if needed (no extra deps).
    """
    if img_raw.dtype != np.uint8:
        img_raw = np.clip(img_raw, 0, 255).astype(np.uint8)
    if img_raw.ndim == 2:
        img_raw = np.stack([img_raw, img_raw, img_raw], axis=-1)
    if img_raw.shape[-1] == 4:
        img_raw = img_raw[..., :3]

    h, w = img_raw.shape[:2]
    if h != img_size or w != img_size:
        ys = np.linspace(0, h - 1, img_size).astype(int)
        xs = np.linspace(0, w - 1, img_size).astype(int)
        img_raw = img_raw[np.ix_(ys, xs)]

    return torch.from_numpy(img_raw).permute(2, 0, 1).float() / 255.0  # (3, H, W)


# ── Teacher latent encoding ───────────────────────────────────────────────────

def _encode_z_goal(
    teacher,
    q_start: np.ndarray,
    q_goal:  np.ndarray,
    device:  torch.device,
    normalize_coords: bool = True,
) -> torch.Tensor:
    """
    Encode (q_start, q_goal) → z_goal using the teacher network.
    Matches the normalization used in preprocess_multi_configs.py.
    """
    import math
    qs = torch.tensor(q_start, dtype=torch.float32)
    qg = torch.tensor(q_goal,  dtype=torch.float32)

    if normalize_coords:
        norm_factors = torch.tensor(
            [math.pi, math.pi, math.pi, math.pi, math.pi, 0.5],
            dtype=torch.float32,
        )
        qs = qs / norm_factors
        qg = qg / norm_factors

    pair = torch.cat([qs, qg], dim=0).unsqueeze(0).to(device)  # (1, 12)
    with torch.no_grad():
        z_out = teacher.encode_pair_latents(pair)

    # Most teacher implementations return (z_start, z_goal); keep compatibility
    # with variants that directly return a single tensor.
    if isinstance(z_out, (tuple, list)):
        if len(z_out) < 2:
            raise RuntimeError(
                f"encode_pair_latents returned {type(z_out).__name__} of len={len(z_out)}"
            )
        z_goal = z_out[1]
    else:
        z_goal = z_out

    if not isinstance(z_goal, torch.Tensor):
        raise RuntimeError(
            f"encode_pair_latents z_goal must be torch.Tensor, got {type(z_goal).__name__}"
        )
    return z_goal.squeeze(0).cpu().float()                      # (z_dim,)


# ── File discovery ────────────────────────────────────────────────────────────

def _collect_h5_paths(input_path: Path, pattern: str, recursive: bool) -> List[str]:
    if input_path.is_file():
        return [str(input_path)]
    paths = sorted(glob.glob(str(input_path / pattern)))
    if not paths and recursive:
        paths = sorted(str(p) for p in input_path.rglob(Path(pattern).name) if p.is_file())
    out = []
    for p in paths:
        pp = Path(p)
        if "__MACOSX" in pp.parts or pp.name.startswith("._"):
            continue
        out.append(str(pp.resolve()))
    return sorted(set(out))


# ── Shard writing ─────────────────────────────────────────────────────────────

def _flush_shard(
    data: List[Dict],
    out_dir: Path,
    shard_idx: int,
    first: int,
) -> str:
    idx  = first + shard_idx
    path = out_dir / f"test_shard_{idx:03d}.pt"
    torch.save(data, path)
    tqdm.write(f"  Wrote {path.name}  ({len(data)} scenes)")
    return str(path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",  "-i", type=Path, default=Path(DEFAULT_INPUT))
    parser.add_argument("--output-dir", "-o", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--teacher-checkpoint", "-t", required=True)
    parser.add_argument("--teacher-data-path", default=None)
    parser.add_argument("--glob",  "-g", default="grasp_multi3_*.h5")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--shard-size",  type=int, default=SHARD_SIZE)
    parser.add_argument("--img-size",    type=int, default=IMG_SIZE)
    parser.add_argument("--first-shard-index", type=int, default=0)
    parser.add_argument(
        "--no-normalize-coords", action="store_true",
        help="Skip joint coord normalization before teacher encode.",
    )
    parser.add_argument(
        "--home-q", type=float, nargs=6, default=None,
        metavar=("J0","J1","J2","J3","J4","J5"),
        help="Override q_start fallback (rad).",
    )
    args = parser.parse_args()

    q_start_fallback = (
        torch.tensor(args.home_q, dtype=torch.float32)
        if args.home_q is not None
        else HOME_Q.clone()
    )
    normalize_coords = not args.no_normalize_coords

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading teacher: {args.teacher_checkpoint}")
    teacher, z_dim = load_teacher(args.teacher_checkpoint, device, args.teacher_data_path)
    teacher = teacher.to(device)
    print(f"Teacher z_dim: {z_dim}")

    h5_files = _collect_h5_paths(
        args.input.expanduser().resolve(), args.glob, args.recursive
    )
    if not h5_files:
        raise FileNotFoundError(f"No H5 files matching {args.glob!r} under {args.input}")
    print(f"Found {len(h5_files)} H5 file(s).")

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_size   = max(1, args.shard_size)
    first        = args.first_shard_index
    write_buffer: List[Dict] = []
    shard_paths:  List[str]  = []
    num_shards   = 0
    num_samples  = 0
    num_skipped  = 0

    for h5_path in tqdm(h5_files, desc="Processing H5 files"):
        raw = _read_h5_scene(h5_path)
        if raw is None:
            num_skipped += 1
            continue

        image = _process_image(raw["img_raw"], args.img_size)
        q_start = raw["q_start"] if raw["q_start"] is not None else q_start_fallback.numpy()
        n_objects = len(raw["all_object_names"])

        for obj_idx in range(n_objects):
            q_goal = raw["q_goals"][obj_idx]
            object_name = raw["all_object_names"][obj_idx]
            object_loc = raw["all_object_locations"][obj_idx]

            try:
                z_goal = _encode_z_goal(
                    teacher, q_start, q_goal, device, normalize_coords
                )
            except Exception as e:
                tqdm.write(f"  [SKIP z_goal] {raw['source_file']} obj{obj_idx}: {e}")
                num_skipped += 1
                continue

            record: Dict[str, Any] = {
                "image":                image,
                "object_name":          object_name,
                "object_location":      object_loc,
                "all_object_locations": raw["all_object_locations"],
                "all_object_names":     raw["all_object_names"],
                "all_object_ids":       raw["all_object_ids"],
                "z_goal":               z_goal,
                "q_goal":               torch.tensor(q_goal, dtype=torch.float32),
                "q_start":              torch.tensor(q_start, dtype=torch.float32),
                "target_obj_idx":       obj_idx,
                "seed":                 raw["seed"],
                "source_file":          raw["source_file"],
            }

            write_buffer.append(record)
            num_samples += 1

            if len(write_buffer) >= shard_size:
                path = _flush_shard(write_buffer[:shard_size], out_dir, num_shards, first)
                shard_paths.append(path)
                del write_buffer[:shard_size]
                num_shards += 1

    # ── Flush remainder ───────────────────────────────────────────────────────
    if write_buffer:
        path = _flush_shard(write_buffer, out_dir, num_shards, first)
        shard_paths.append(path)
        num_shards += 1

    if num_samples == 0:
        raise RuntimeError(f"No valid scenes processed (skipped {num_skipped}).")

    # ── Write manifest ────────────────────────────────────────────────────────
    manifest = {
        "num_samples":   num_samples,
        "num_shards":    num_shards,
        "shards":        shard_paths,
        "img_size":      args.img_size,
        "z_dim":         z_dim,
        "q_start_fallback": q_start_fallback.tolist(),
        "normalize_coords": normalize_coords,
        "source":        "build_test_dataset",
        "input_glob":    args.glob,
        "fields": [
            "image", "object_name", "object_location",
            "all_object_locations", "all_object_names", "all_object_ids",
            "z_goal", "q_goal", "q_start", "target_obj_idx", "seed", "source_file",
        ],
    }
    manifest_path = out_dir / "manifest.pt"
    torch.save(manifest, manifest_path)

    print(f"\nDone.")
    print(f"  Scenes processed : {num_samples}")
    print(f"  Shards written   : {num_shards}")
    print(f"  Scenes skipped   : {num_skipped}")
    print(f"  Manifest         : {manifest_path}")
    print(f"\nRecord fields per scene:")
    for field in manifest["fields"]:
        print(f"  {field}")


if __name__ == "__main__":
    main()