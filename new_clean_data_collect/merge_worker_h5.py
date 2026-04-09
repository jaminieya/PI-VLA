#!/usr/bin/env python3
"""Merge per-worker H5 outputs into one flat directory for prepare_trajectory_dataset.py."""

from __future__ import annotations

import argparse
import glob
import os
import shutil


def main() -> None:
    p = argparse.ArgumentParser(description="Merge worker_gpu*/ntfield_rrt_ep_*.h5 into one folder.")
    p.add_argument("--input_root", type=str, required=True, help="Root containing worker_gpu*/ subdirs.")
    p.add_argument("--output_dir", type=str, required=True, help="Flat destination directory.")
    p.add_argument("--copy", action="store_true", help="Copy files instead of hard-linking.")
    args = p.parse_args()

    input_root = os.path.abspath(args.input_root)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    worker_dirs = sorted(glob.glob(os.path.join(input_root, "worker_gpu*")))
    if not worker_dirs:
        raise SystemExit(f"No worker_gpu* folders found under {input_root}")

    n = 0
    for wdir in worker_dirs:
        worker = os.path.basename(wdir)
        for src in sorted(glob.glob(os.path.join(wdir, "ntfield_rrt_ep_*.h5"))):
            base = os.path.basename(src)
            dst_name = f"{worker}__{base}"
            dst = os.path.join(output_dir, dst_name)
            if os.path.exists(dst):
                continue
            if args.copy:
                shutil.copy2(src, dst)
            else:
                os.link(src, dst)
            n += 1

    print(f"Merged {n} files into: {output_dir}")
    print("Next:")
    print(
        f"  python ntrl-demo/dataprocessing/prepare_trajectory_dataset.py "
        f"--data_dir {output_dir} --output_dir ntrl-demo/datasets/arm/UR5_rrt_clean_YYYYMMDD --num_pairs 100000"
    )


if __name__ == "__main__":
    main()
