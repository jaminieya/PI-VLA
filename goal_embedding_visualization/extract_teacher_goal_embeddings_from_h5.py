#!/usr/bin/env python3
"""
Compute NTField teacher z_goal latents from grasp HDF5 produced by
``trajectory_evaluation/rrtconnect/collect_data.py``.

Each HDF5 stores ``joint_configs`` (trajectory) and ``final_joint_config`` (goal).
The teacher matches training in ``hanwen_grasping/train_goal_rep_alignment.py``:

  coords = concat(q_start, q_goal)  # optional /SCALE
  _, z_goal = teacher.encode_pair_latents(coords)

Outputs (one folder per source HDF5, named after the file stem)::

  PI-VLA/output/embedding_visualization/<grasp_6dof_demo_TIMESTAMP>/teacher_z_goal_bundle.npz

Typical workflow (after collecting under ``PI-VLA/output/trajectory_evaluation/``):

  cd PI-VLA/goal_embedding_visualization
  python extract_teacher_goal_embeddings_from_h5.py \\
    --h5_glob '../output/trajectory_evaluation/*/*.h5' \\
    --checkpoint ../ntrl-demo/Experiments/UR5_trajectory/.../Model_Epoch_*.pt

  python pca_embedding_3d.py --h5 ../output/trajectory_evaluation/.../grasp_6dof_demo_*.h5

Requires: torch, h5py, numpy; teacher code on PYTHONPATH via hanwen_grasping + ntrl-demo.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import List, Tuple

import numpy as np
import torch

_FILE = os.path.abspath(__file__)
_GOAL_VIS_DIR = os.path.dirname(_FILE)
_PI_VLA_ROOT = os.path.dirname(_GOAL_VIS_DIR)
_OUTPUT_EMBED_ROOT = os.path.join(_PI_VLA_ROOT, "output", "embedding_visualization")
_HANWEN = os.path.join(_PI_VLA_ROOT, "hanwen_grasping")
_NTRL = os.path.join(_PI_VLA_ROOT, "ntrl-demo")

for _p in (_HANWEN, _NTRL):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from train_goal_rep_alignment import build_coords_batch, load_teacher  # noqa: E402


def _resolve_checkpoint(path: str) -> str:
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(_PI_VLA_ROOT, path))


def _iter_pairs_from_h5(
    path: str,
    *,
    stride: int,
    one_per_demo: bool,
) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    """Return list of (frame_index, q_start[6], q_goal[6])."""
    import h5py

    out: List[Tuple[int, np.ndarray, np.ndarray]] = []
    with h5py.File(path, "r") as f:
        if "joint_configs" not in f or "final_joint_config" not in f:
            raise KeyError(f"{path}: need joint_configs and final_joint_config")
        jc = np.asarray(f["joint_configs"], dtype=np.float32)
        qg = np.asarray(f["final_joint_config"], dtype=np.float32).reshape(-1)[:6]
    n = jc.shape[0]
    if n == 0:
        return out
    indices = [0] if one_per_demo else list(range(0, n, max(stride, 1)))
    for i in indices:
        out.append((int(i), jc[i, :6].copy(), qg.copy()))
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Extract teacher z_goal embeddings from collect_data.py HDF5 files"
    )
    p.add_argument(
        "--h5",
        type=str,
        default="",
        help="Single grasp_6dof_demo_*.h5 path (use this or --h5_glob)",
    )
    p.add_argument(
        "--h5_glob",
        type=str,
        default="",
        help="Glob for multiple HDF5 files (e.g. ../output/trajectory_evaluation/*/*.h5)",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="NTField teacher checkpoint (Model_Epoch_*.pt), path relative to PI-VLA ok",
    )
    p.add_argument(
        "--normalize_coords",
        action="store_true",
        help="Match train_goal_rep_alignment: divide joints by π/0.5 before teacher",
    )
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Inference batch size for encode_pair_latents",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Use every stride-th frame along joint_configs (ignored if --one_per_demo)",
    )
    p.add_argument(
        "--one_per_demo",
        action="store_true",
        help="Only first frame per HDF5 (one point per collected episode)",
    )
    p.add_argument(
        "--out_npz",
        type=str,
        default="",
        help="Bundle filename or absolute path. Default: output/embedding_visualization/<h5_stem>/teacher_z_goal_bundle.npz "
        "per file. Absolute path allowed only with a single --h5.",
    )
    args = p.parse_args()

    ckpt = _resolve_checkpoint(args.checkpoint)
    if not os.path.isfile(ckpt):
        sys.exit(f"Checkpoint not found: {ckpt}")

    paths: List[str] = []
    if args.h5:
        paths = [os.path.abspath(args.h5)]
    if args.h5_glob:
        g = sorted(glob.glob(os.path.abspath(args.h5_glob)))
        paths.extend([x for x in g if os.path.isfile(x)])
    paths = sorted(set(paths))
    if not paths:
        sys.exit("No HDF5 files: pass --h5 or a matching --h5_glob")

    if args.out_npz and os.path.isabs(args.out_npz) and len(paths) > 1:
        sys.exit("Absolute --out_npz is only supported with a single --h5 (not a glob of many files).")

    device = torch.device(args.device)
    teacher, nt_h = load_teacher(ckpt, device)
    teacher.eval()

    n_written = 0
    for h5_path in paths:
        try:
            pairs = _iter_pairs_from_h5(
                h5_path, stride=args.stride, one_per_demo=args.one_per_demo
            )
        except Exception as e:
            print(f"Skip {h5_path}: {e}")
            continue
        if not pairs:
            continue

        stem = os.path.splitext(os.path.basename(h5_path))[0]
        out_dir = os.path.join(_OUTPUT_EMBED_ROOT, stem)

        if args.out_npz:
            if os.path.isabs(args.out_npz):
                out_path = args.out_npz
            else:
                out_path = os.path.join(out_dir, os.path.basename(args.out_npz))
        else:
            out_path = os.path.join(out_dir, "teacher_z_goal_bundle.npz")

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        qs_list = [p[1] for p in pairs]
        qg_list = [p[2] for p in pairs]
        q_start_b = torch.from_numpy(np.stack(qs_list, axis=0)).to(device)
        q_goal_b = torch.from_numpy(np.stack(qg_list, axis=0)).to(device)

        z_s_chunks: List[torch.Tensor] = []
        z_g_chunks: List[torch.Tensor] = []
        bs = max(1, args.batch_size)
        with torch.no_grad():
            for s in range(0, q_start_b.shape[0], bs):
                e = min(s + bs, q_start_b.shape[0])
                coords = build_coords_batch(q_start_b[s:e], q_goal_b[s:e], args.normalize_coords)
                zs, zg = teacher.encode_pair_latents(coords)
                z_s_chunks.append(zs.detach().float().cpu())
                z_g_chunks.append(zg.detach().float().cpu())
        z_start = torch.cat(z_s_chunks, dim=0).numpy().astype(np.float32)
        z_goal = torch.cat(z_g_chunks, dim=0).numpy().astype(np.float32)
        nrows = z_goal.shape[0]

        # Single-demo bundle: labels_demo is 0 everywhere; use labels_frame for trajectory index.
        labels_demo = np.zeros(nrows, dtype=np.int32)
        labels_frame = np.asarray([p[0] for p in pairs], dtype=np.int32)

        np.savez(
            out_path,
            z_start=z_start,
            z_goal=z_goal,
            labels_demo=labels_demo,
            labels_frame=labels_frame,
            h5_basename=np.array(os.path.basename(h5_path)),
            source_h5=np.array(os.path.abspath(h5_path)),
            teacher_checkpoint=np.array(ckpt),
            normalize_coords=np.array(bool(args.normalize_coords)),
            ntfield_h=np.array(int(nt_h)),
        )
        print(
            f"Saved {nrows} x {z_goal.shape[1]} (z_start + z_goal) -> {out_path}\n"
            f"  Note: z_goal is usually constant along one trajectory (fixed final_joint_config); "
            f"use z_start for PCA on a single demo."
        )
        n_written += 1

    if n_written == 0:
        sys.exit("No embeddings extracted (empty or invalid HDF5 list).")
    print(f"Done: {n_written} bundle(s); parent dir = {_OUTPUT_EMBED_ROOT} (one subfolder per .h5 stem)")
    print(f"ntfield_h={nt_h}, normalize_coords={args.normalize_coords}")


if __name__ == "__main__":
    main()
