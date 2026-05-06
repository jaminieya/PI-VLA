#!/usr/bin/env python3
"""
Convert per-object collector shards (grasp_<object_id>_stepNN_*.h5) into the same
list-of-dicts .pt shard format as preprocess_multi_configs.py / pt_shards_multi.

Each input H5 is expected to contain (from collect_multi_obj_data_for_student.py):
  - image:         (H, W, 3) uint8
  - goal_joint_config: (6,) float32
  - object_name:   (1,) str
  attrs: joint_dim, step, num_steps


python3 student_model_training/preprocess_dataset/convert_grasp_h5_to_pt_shards.py \
  /home/hojinsohn/VLM-NT/PI-VLA/output/multi_obj \
  --teacher-checkpoint teacher_model.pt \
  --output-dir student_model_training/data/pt_shards_from_grasp_h5 \
  --glob 'grasp_*.h5'

object_id_folder is parsed from the basename: grasp_011_banana_step00_*.h5 -> 011_banana

object_location is not stored in these H5 files; this script uses a (3,) zero tensor
unless you pass --object-location-file (JSON: basename -> [x,y,z]).

z_goal is teacher.encode_pair_latents([q_start | q_goal]) with the same normalization
as preprocess_multi_configs. Default q_start matches that script (Isaac warm-up pose);
pass --home-q to match collect_multi_obj_data_for_student HOME_DOF if desired.

Example (from PI-VLA repo root). Nested runs under ``output/multi_obj/...`` are
found automatically (recursive ``**/grasp_*.h5`` if nothing in the top folder)::

  python student_model_training/preprocess_dataset/convert_grasp_h5_to_pt_shards.py \\
    output/multi_obj \\
    --teacher-checkpoint teacher_model.pt \\
    --output-dir student_model_training/data/pt_shards_from_grasp_h5 \\
    --glob 'grasp_*.h5'
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_ST_ROOT = _SCRIPT_DIR.parent  # student_model_training
_PI_VLA_ROOT = _ST_ROOT.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(_PI_VLA_ROOT) not in sys.path:
    sys.path.insert(0, str(_PI_VLA_ROOT))
# Teacher lives under ntrl-demo/models/... (same as preprocess_multi_configs runners).
_NTRL_DEMO = _PI_VLA_ROOT / "ntrl-demo"
if str(_NTRL_DEMO) not in sys.path:
    sys.path.insert(0, str(_NTRL_DEMO))

from preprocess_multi_configs import (  # noqa: E402
    HOME_Q,
    IMG_SIZE,
    SHARD_SIZE,
    compute_z_goals,
    load_teacher,
    process_img,
)

# Filename: grasp_<object_id_folder>_step<dd>_<timestamp>.h5
_GRASP_NAME_RE = re.compile(
    r"^grasp_(?P<folder>.+)_step(?P<step>\d+)_(?P<rest>.+)\.h5$",
    re.IGNORECASE,
)


def _parse_object_id_folder(basename: str) -> str:
    m = _GRASP_NAME_RE.match(basename)
    if m:
        return m.group("folder")
    stem = basename[:-3] if basename.lower().endswith(".h5") else basename
    if stem.startswith("grasp_"):
        stem = stem[len("grasp_") :]
    parts = stem.split("_step")
    return parts[0] if parts else stem


def _decode_object_name(ds) -> str:
    raw = ds[0]
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    if isinstance(raw, np.bytes_):
        return str(raw, "utf-8", errors="replace")
    return str(raw)


def _load_location_map(path: Optional[str]) -> Dict[str, List[float]]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for k, v in data.items():
        if isinstance(v, (list, tuple)) and len(v) == 3:
            out[os.path.basename(k)] = [float(v[0]), float(v[1]), float(v[2])]
    return out


def collect_h5_paths(input_path: Path, pattern: str) -> List[str]:
    if input_path.is_file():
        return [str(input_path)]
    if input_path.is_dir():
        # Non-recursive first (only files directly in input_path).
        paths = sorted(glob.glob(str(input_path / pattern)))
        if not paths:
            # Collector layout nests under e.g. output/multi_obj/DATE/gpuN/RUN_ID/
            paths = sorted(
                glob.glob(str(input_path / "**" / pattern), recursive=True)
            )
        if not paths:
            raise FileNotFoundError(
                f"No files matching {pattern!r} under {input_path} "
                f"(tried non-recursive and **/{pattern})."
            )
        return paths
    raise FileNotFoundError(f"Input not found: {input_path}")


def load_rows_from_h5_files(
    h5_paths: List[str],
    loc_map: Dict[str, List[float]],
) -> Tuple[
    List[np.ndarray],
    List[np.ndarray],
    List[str],
    List[str],
    List[str],
    List[Optional[List[float]]],
]:
    images: List[np.ndarray] = []
    q_goals: List[np.ndarray] = []
    object_names: List[str] = []
    folders: List[str] = []
    basenames: List[str] = []
    locs: List[Optional[List[float]]] = []

    for p in h5_paths:
        base = os.path.basename(p)
        folder = _parse_object_id_folder(base)
        with h5py.File(p, "r") as f:
            if "image" not in f or "goal_joint_config" not in f or "object_name" not in f:
                raise KeyError(
                    f"{p}: need datasets 'image', 'goal_joint_config', 'object_name'"
                )
            img = np.array(f["image"])
            qg = np.array(f["goal_joint_config"], dtype=np.float32).reshape(6)
            oname = _decode_object_name(f["object_name"])

        images.append(img)
        q_goals.append(qg)
        object_names.append(oname)
        folders.append(folder)
        basenames.append(base)
        locs.append(loc_map.get(base))

    return images, q_goals, object_names, folders, basenames, locs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert grasp_*_step*.h5 collector files to pt_shards_multi-style .pt lists."
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to one .h5 file or a directory of them.",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        "-t",
        required=True,
        help="Teacher checkpoint (.pt), same as preprocess_multi_configs.py.",
    )
    parser.add_argument(
        "--teacher-data-path",
        default=None,
        help="Optional data path for teacher model init.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        help="Directory for shard_*.pt and manifest.pt",
    )
    parser.add_argument(
        "--glob",
        "-g",
        default="grasp_*.h5",
        help=(
            "Glob when input is a directory (default: grasp_*.h5). "
            "Searches direct children first, then all subdirs (**/pattern) if none match."
        ),
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=SHARD_SIZE,
        help=f"Samples per shard (default {SHARD_SIZE}).",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=IMG_SIZE,
        help=f"Square resize (default {IMG_SIZE}).",
    )
    parser.add_argument(
        "--no-normalize-coords",
        action="store_true",
        help="Do not divide joint coords by pi/0.5 before teacher encode.",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Omit object_location, object_id_folder, source_file in each sample.",
    )
    parser.add_argument(
        "--object-location-file",
        default=None,
        help="Optional JSON map basename -> [x,y,z] for object_location.",
    )
    parser.add_argument(
        "--home-q",
        type=float,
        nargs=6,
        default=None,
        metavar=("J0", "J1", "J2", "J3", "J4", "J5"),
        help=(
            "q_start (rad) for teacher pair encoding. "
            "Default: same as preprocess_multi_configs.HOME_Q "
            f"({HOME_Q.tolist()})."
        ),
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    if not input_path.is_absolute():
        input_path = (_PI_VLA_ROOT / input_path).resolve()
    h5_paths = collect_h5_paths(input_path, args.glob)
    loc_map = _load_location_map(args.object_location_file)

    if args.home_q is not None:
        q_start_home = torch.tensor(args.home_q, dtype=torch.float32)
    else:
        q_start_home = HOME_Q.clone()
    print("q_start_home:", q_start_home.tolist())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    teacher, z_dim = load_teacher(
        os.path.abspath(args.teacher_checkpoint),
        device,
        args.teacher_data_path,
    )
    teacher = teacher.to(device)
    print("Teacher z_dim:", z_dim)

    normalize_coords = not args.no_normalize_coords

    (
        images_np,
        q_goals_list,
        object_names,
        folders,
        basenames,
        locs_opt,
    ) = load_rows_from_h5_files(h5_paths, loc_map)

    q_goals_t = torch.stack([torch.from_numpy(q) for q in q_goals_list], dim=0)
    q_starts_t = q_start_home.unsqueeze(0).expand(q_goals_t.shape[0], -1).clone()
    z_goals_t = compute_z_goals(
        teacher, q_starts_t, q_goals_t, device, normalize_coords
    )

    all_datapoints: List[Dict] = []
    for i in range(len(h5_paths)):
        img_t = process_img(images_np[i], args.img_size)
        dp: Dict = {
            "image": img_t,
            "object_name": object_names[i],
            "z_goal": z_goals_t[i].clone(),
            "q_goal": q_goals_t[i].clone(),
        }
        if not args.no_metadata:
            if locs_opt[i] is not None:
                loc = torch.tensor(locs_opt[i], dtype=torch.float32)
            else:
                loc = torch.zeros(3, dtype=torch.float32)
            dp["object_location"] = loc
            dp["object_id_folder"] = folders[i]
            dp["source_file"] = basenames[i]
        all_datapoints.append(dp)

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = (_PI_VLA_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_size = max(1, int(args.shard_size))
    num_shards = int(math.ceil(len(all_datapoints) / shard_size))
    pad = max(1, len(str(max(0, num_shards - 1))))
    shard_paths: List[str] = []

    for si in range(num_shards):
        chunk = all_datapoints[si * shard_size : (si + 1) * shard_size]
        shard_path = out_dir / f"shard_{si:0{pad}d}.pt"
        torch.save(chunk, shard_path)
        shard_paths.append(str(shard_path))
        print(f"Wrote {shard_path} ({len(chunk)} samples)")

    manifest_path = out_dir / "manifest.pt"
    torch.save(
        {
            "num_samples": len(all_datapoints),
            "num_shards": num_shards,
            "shards": shard_paths,
            "img_size": args.img_size,
            "z_dim": z_dim,
            "q_start": q_start_home.tolist(),
            "source": "convert_grasp_h5_to_pt_shards",
            "input_glob": args.glob,
            "object_location_from_json": bool(loc_map),
        },
        manifest_path,
    )
    print("Manifest:", manifest_path)
    print("Done.", len(all_datapoints), "samples ->", num_shards, "shard(s).")


if __name__ == "__main__":
    main()
