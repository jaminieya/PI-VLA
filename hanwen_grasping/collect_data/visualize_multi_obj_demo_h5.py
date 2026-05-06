#!/usr/bin/env python3
"""
Visualize grasp_multi3_demo_*.h5 (multi-object scene bundle): print metadata,
save start_image PNG, and save an overview figure with labels.

Example::

  python collect_data/visualize_multi_obj_demo_h5.py \\
    ../../multi_obj_dataset/grasp_multi3_demo_20260420_000719_307450.h5

Run from hanwen_grasping (or pass absolute paths).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, List

import h5py
import numpy as np


def _decode_str_array(ds: h5py.Dataset) -> List[str]:
    out: List[str] = []
    for i in range(len(ds)):
        s = ds[i]
        if isinstance(s, bytes):
            s = s.decode("utf-8", errors="replace")
        elif isinstance(s, np.bytes_):
            s = str(s, "utf-8", errors="replace")
        else:
            s = str(s)
        out.append(s)
    return out


def _load_bundle(path: str) -> dict[str, Any]:
    with h5py.File(path, "r") as f:
        if "start_image" not in f:
            raise KeyError(f"{path}: missing dataset 'start_image'")
        data = {
            "start_image": np.array(f["start_image"]),
            "goal_joint_configs": np.array(f["goal_joint_configs"]),
            "object_locations": np.array(f["object_locations"]),
            "object_names": _decode_str_array(f["object_names"]),
            "object_id_folders": _decode_str_array(f["object_id_folders"]),
            "attrs": {k: f.attrs[k] for k in f.attrs},
        }
    return data


def _save_start_image_png(img: np.ndarray, out_path: str) -> None:
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    try:
        from PIL import Image

        Image.fromarray(img).save(out_path)
    except ImportError:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(14, 8))
        ax.imshow(img)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
    print(f"Wrote {out_path}")


def _save_overview_figure(
    img: np.ndarray,
    names: List[str],
    folders: List[str],
    locs: np.ndarray,
    goals: np.ndarray,
    attrs: dict,
    out_path: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lines = [
        os.path.basename(out_path).replace("_overview.png", ".h5"),
        f"attrs: {attrs}",
        "",
    ]
    for i, (name, folder) in enumerate(zip(names, folders)):
        xyz = locs[i]
        g = goals[i]
        lines.append(
            f"[{i}] {name}  ({folder})"
        )
        lines.append(f"     xyz (m): {xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f}")
        lines.append(f"     goal joints (rad): {np.array2string(g, precision=4, max_line_width=120)}")

    fig = plt.figure(figsize=(14, 10))
    ax_img = fig.add_axes([0.02, 0.28, 0.96, 0.70])
    ax_img.imshow(img)
    ax_img.set_title("start_image (top camera)", fontsize=12)
    ax_img.axis("off")

    ax_txt = fig.add_axes([0.02, 0.02, 0.96, 0.24])
    ax_txt.axis("off")
    ax_txt.text(
        0.0,
        1.0,
        "\n".join(lines),
        transform=ax_txt.transAxes,
        fontsize=9,
        family="monospace",
        verticalalignment="top",
        wrap=True,
    )
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize grasp_multi3_demo_*.h5 and save PNGs."
    )
    parser.add_argument(
        "h5_path",
        help="Path to grasp_multi3_demo_*.h5",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Directory for outputs (default: same directory as the .h5 file)",
    )
    parser.add_argument(
        "--no_overview",
        action="store_true",
        help="Only save start_image PNG, skip matplotlib overview figure",
    )
    args = parser.parse_args()

    path = os.path.abspath(args.h5_path)
    if not os.path.isfile(path):
        print(f"Not found: {path}", file=sys.stderr)
        sys.exit(1)

    bundle = _load_bundle(path)
    img = bundle["start_image"]
    names = bundle["object_names"]
    folders = bundle["object_id_folders"]
    locs = bundle["object_locations"]
    goals = bundle["goal_joint_configs"]
    attrs = bundle["attrs"]

    print(f"File: {path}")
    print(f"  start_image shape: {img.shape} dtype={img.dtype}")
    print(f"  num_objects: {len(names)}")
    for i in range(len(names)):
        print(f"    [{i}] {names[i]}  ({folders[i]})")

    stem = os.path.splitext(os.path.basename(path))[0]
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(path)
    os.makedirs(out_dir, exist_ok=True)

    png_start = os.path.join(out_dir, f"{stem}_start_image.png")
    _save_start_image_png(img, png_start)

    if not args.no_overview:
        png_over = os.path.join(out_dir, f"{stem}_overview.png")
        _save_overview_figure(img, names, folders, locs, goals, attrs, png_over)


if __name__ == "__main__":
    main()
