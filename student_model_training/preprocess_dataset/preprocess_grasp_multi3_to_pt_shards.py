#!/usr/bin/env python3
"""
Convert grasp_multi3_*.h5 (collect_multi_obj_data_for_student.py) to pt_shards_multi format.

Each shard is a list[dict] compatible with full_train_multi*.py discover_shards():
  - "image"        (3, H, W) float32 in [0, 1]
  - "object_name"  str
  - "z_goal"       (z_dim,) float32  — teacher.encode_pair_latents([q_start | q_goal])
  - "q_goal"       (6,) float32    — goal joints (radians)
  - optional: object_location, object_id_folder, source_file

q_start defaults to Isaac warm-up HOME_Q (same as preprocess_multi_configs.py).

Example (single directory of flat-output H5 files):
  python preprocess_grasp_multi3_to_pt_shards.py \
    --input /home/hojinsohn/VLM-NT/PI-VLA/hanwen_grasping/output/multi_obj/20260430 \
    --teacher-checkpoint /home/hojinsohn/VLM-NT/PI-VLA/teacher_model.pt \
    --output-dir /home/hojinsohn/VLM-NT/PI-VLA/student_model_training/data/pt_shards_multi_20260501 \
    --glob 'grasp_multi3_*.h5'

Write shards as shard_03.pt, shard_04.pt, ... (two-digit padding, starting at 3):
  ... --first-shard-index 3 --shard-width 2

From repo root (PI-VLA):
  python student_model_training/preprocess_dataset/preprocess_grasp_multi3_to_pt_shards.py ...
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_ST_ROOT = _SCRIPT_DIR.parent
_PI_VLA_ROOT = _ST_ROOT.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(_PI_VLA_ROOT) not in sys.path:
    sys.path.insert(0, str(_PI_VLA_ROOT))
_NTRL_DEMO = _PI_VLA_ROOT / "ntrl-demo"
if str(_NTRL_DEMO) not in sys.path:
    sys.path.insert(0, str(_NTRL_DEMO))

from preprocess_multi_configs import (  # noqa: E402
    HOME_Q,
    IMG_SIZE,
    SHARD_SIZE,
    chunk,
    load_h5_datapoints,
    load_teacher,
)


DEFAULT_INPUT = (
    "/home/hojinsohn/VLM-NT/PI-VLA/hanwen_grasping/output/multi_obj/20260430"
)
DEFAULT_OUTPUT = (
    "/home/hojinsohn/VLM-NT/PI-VLA/student_model_training/data/pt_shards_multi"
)


def collect_h5_paths(input_path: Path, pattern: str, recursive: bool) -> List[str]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".h5":
            raise ValueError(f"Not an .h5 file: {input_path}")
        return [str(input_path)]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input not found: {input_path}")

    paths = sorted(glob.glob(str(input_path / pattern)))
    if not paths and recursive:
        paths = sorted(input_path.rglob(pattern.replace("\\", "/").split("/")[-1]))
        paths = [str(p) for p in paths if p.is_file()]
    # Skip junk
    out: List[str] = []
    for p in paths:
        pp = Path(p)
        if "__MACOSX" in pp.parts or pp.name.startswith("._"):
            continue
        out.append(str(pp.resolve()))
    return sorted(set(out))


def _shard_width(first: int, num_shards: int, min_width: int) -> int:
    last = first + max(0, num_shards - 1)
    dynamic = len(str(max(first, last)))
    return max(min_width, dynamic)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        type=Path,
        default=Path(DEFAULT_INPUT),
        help=f"H5 file or directory (default: {DEFAULT_INPUT})",
    )
    p.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path(DEFAULT_OUTPUT),
        help=f"Directory for shard_*.pt and manifest.pt (default: {DEFAULT_OUTPUT})",
    )
    p.add_argument(
        "--teacher-checkpoint",
        "-t",
        required=True,
        help="Teacher .pt checkpoint for encode_pair_latents.",
    )
    p.add_argument(
        "--teacher-data-path",
        default=None,
        help="Optional datasets path for teacher Model() init.",
    )
    p.add_argument(
        "--glob",
        "-g",
        default="grasp_multi3_*.h5",
        help="Glob when input is a directory (default: grasp_multi3_*.h5)",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="If no direct matches, search subdirectories for the glob basename.",
    )
    p.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    p.add_argument("--img-size", type=int, default=IMG_SIZE)
    p.add_argument(
        "--first-shard-index",
        type=int,
        default=0,
        help="First shard filename index (e.g. 3 -> shard_03.pt when width>=2).",
    )
    p.add_argument(
        "--shard-width",
        type=int,
        default=0,
        help="Zero-pad width for shard filenames; 0 = auto from indices.",
    )
    p.add_argument(
        "--no-normalize-coords",
        action="store_true",
        help="Do not divide joint coords by pi/0.5 before teacher encode.",
    )
    p.add_argument(
        "--no-metadata",
        action="store_true",
        help="Drop object_location, object_id_folder, source_file.",
    )
    p.add_argument(
        "--home-q",
        type=float,
        nargs=6,
        default=None,
        metavar=("J0", "J1", "J2", "J3", "J4", "J5"),
        help="Override q_start (rad); default Isaac HOME_Q.",
    )
    args = p.parse_args()

    if args.home_q is not None:
        q_start_home = torch.tensor(args.home_q, dtype=torch.float32)
        print(f"Using custom home_q: {q_start_home.tolist()}")
    else:
        q_start_home = HOME_Q.clone()
        print(f"Using default home_q: {q_start_home.tolist()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading teacher: {args.teacher_checkpoint}")
    teacher, z_dim = load_teacher(
        args.teacher_checkpoint, device, args.teacher_data_path
    )
    teacher = teacher.to(device)
    print(f"Teacher z_dim: {z_dim}")

    h5_files = collect_h5_paths(args.input.expanduser().resolve(), args.glob, args.recursive)
    if not h5_files:
        raise FileNotFoundError(
            f"No H5 files matching {args.glob!r} under {args.input} "
            f"(try --recursive)"
        )
    print(f"Found {len(h5_files)} H5 file(s).")

    normalize_coords = not args.no_normalize_coords
    skipped = 0

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    first = int(args.first_shard_index)
    shard_size = max(1, int(args.shard_size))
    min_width = int(args.shard_width) if args.shard_width > 0 else 2

    shard_paths: List[str] = []
    write_buffer: List[Dict] = []
    num_samples = 0
    num_shards = 0

    def _flush_shard(shard_data: List[Dict], shard_idx: int) -> None:
        nonlocal num_shards
        width = _shard_width(first, first + shard_idx, min_width)
        idx = first + shard_idx
        shard_path = out_dir / f"shard_{idx:0{width}d}.pt"
        torch.save(shard_data, shard_path)
        shard_paths.append(str(shard_path))
        num_shards += 1
        print(f"  Wrote {shard_path.name} ({len(shard_data)} samples)")

    for h5_path in tqdm(h5_files, desc="H5 files"):
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
        except Exception as e:
            skipped += 1
            tqdm.write(f"  [SKIP] {os.path.basename(h5_path)}: {e}")
            continue

        if not dps:
            continue

        num_samples += len(dps)
        write_buffer.extend(dps)
        while len(write_buffer) >= shard_size:
            shard_data = write_buffer[:shard_size]
            _flush_shard(shard_data, num_shards)
            del write_buffer[:shard_size]

    if write_buffer:
        _flush_shard(write_buffer, num_shards)
        write_buffer = []

    if num_samples == 0:
        raise RuntimeError(f"No datapoints (skipped {skipped} files).")

    manifest_path = out_dir / "manifest.pt"
    torch.save(
        {
            "num_samples": num_samples,
            "num_shards": num_shards,
            "shards": shard_paths,
            "img_size": args.img_size,
            "z_dim": z_dim,
            "q_start": q_start_home.tolist(),
            "source": "preprocess_grasp_multi3_to_pt_shards",
            "input_glob": args.glob,
        },
        manifest_path,
    )
    print(f"Manifest -> {manifest_path}")
    print(f"Done. {num_samples} samples, {num_shards} shard(s), skipped files: {skipped}")


if __name__ == "__main__":
    main()
