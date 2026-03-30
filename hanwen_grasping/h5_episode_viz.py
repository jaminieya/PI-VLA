"""Helpers for HDF5 episode visualization (object_location, prompt, session output dirs)."""
from __future__ import annotations

import os
import re
from typing import List, Optional, Tuple

import numpy as np


def h5_session_stamp(h5_path: str) -> Optional[str]:
    m = re.search(r"(\d{8}_\d{6})", os.path.basename(h5_path))
    return m.group(1) if m else None


def trajectory_evaluation_session_dir(pi_vla_root: str, h5_path: str) -> Tuple[Optional[str], Optional[str]]:
    stamp = h5_session_stamp(h5_path)
    if not stamp:
        return None, None
    return stamp, os.path.join(pi_vla_root, "output", "trajectory_evaluation", stamp)


def resolve_h5_path(h5_path: str, pi_vla_root: str, file_dir: str) -> Optional[str]:
    if not h5_path:
        return None
    if os.path.isfile(h5_path):
        return os.path.abspath(h5_path)
    base_candidates = [
        pi_vla_root,
        file_dir,
        os.path.join(pi_vla_root, "collected_data"),
    ]
    for base in base_candidates:
        cand = os.path.join(base, os.path.basename(h5_path))
        if os.path.isfile(cand):
            return os.path.abspath(cand)
        if not os.path.isabs(h5_path):
            cand2 = os.path.join(base, h5_path)
            if os.path.isfile(cand2):
                return os.path.abspath(cand2)
    return None


def read_h5_object_and_prompt(h5_path: str) -> Tuple[Optional[np.ndarray], str]:
    import h5py

    with h5py.File(h5_path, "r") as f:
        if "object_location" in f:
            ol = np.array(f["object_location"][:], dtype=np.float64).reshape(-1)
            ol = ol[:3] if ol.size >= 3 else None
        else:
            ol = None
        prompt = str(f.attrs.get("prompt", "") or "")
    return ol, prompt


def infer_ycb_index_from_prompt(prompt: str, object_asset_files: List[str]) -> Optional[int]:
    if not prompt or not object_asset_files:
        return None
    p = prompt.lower()
    best_i, best_score = None, 0
    for i, path in enumerate(object_asset_files):
        name = path.replace("\\", "/").split("/")[-1].lower()
        if not name.endswith(".urdf"):
            continue
        key = name.replace(".urdf", "")
        parts = [x for x in key.split("_") if len(x) > 2 and not x.isdigit()]
        score = sum(1 for t in parts if t in p)
        if score > best_score:
            best_score, best_i = score, i
    return best_i if best_score > 0 else None


def write_session_meta(session_dir: str, lines: List[str]) -> None:
    os.makedirs(session_dir, exist_ok=True)
    with open(os.path.join(session_dir, "episode_meta.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


def append_session_meta(session_dir: str, lines: List[str]) -> None:
    if not lines or not os.path.isdir(session_dir):
        return
    with open(os.path.join(session_dir, "episode_meta.txt"), "a") as f:
        f.write("\n".join(lines) + "\n")
