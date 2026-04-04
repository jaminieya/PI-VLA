#!/usr/bin/env python3
"""
Sweep NTField (UR5 trajectory) training over dataset sizes 50, 100, ..., 900.

For each size N: randomly sample N HDF5 demos from output/data_collection/all,
build points.npy / tau_obs.npy in
    output/learning_rate/UR5_trajectory/<N>/
then run ntrl-demo/train/train_arm_trajectory.py (checkpoints under
    output/learning_rate/UR5_trajectory/<N>/trajectory_<timestamp>/).

Usage (from PI-VLA root):
    python scripts/train_ntfield_ur5_data_sweep.py
    python scripts/train_ntfield_ur5_data_sweep.py --device cpu --epochs 100
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List


def _pi_vla_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _prepare_subset_symlinks(
    chosen: List[Path],
    h5_subset_dir: Path,
) -> None:
    if h5_subset_dir.exists():
        shutil.rmtree(h5_subset_dir)
    h5_subset_dir.mkdir(parents=True)
    for src in chosen:
        dst = h5_subset_dir / src.name
        try:
            os.symlink(src.resolve(), dst)
        except OSError:
            shutil.copy2(src, dst)


def main() -> None:
    root = _pi_vla_root()
    ntrl_demo = root / "ntrl-demo"
    prep_script = ntrl_demo / "dataprocessing" / "prepare_trajectory_dataset.py"
    train_script = ntrl_demo / "train" / "train_arm_trajectory.py"

    parser = argparse.ArgumentParser(description="UR5 trajectory NTField sweep over data sizes.")
    parser.add_argument(
        "--h5-root",
        type=Path,
        default=None,
        help="Directory of .h5 files (default: output/data_collection/all)",
    )
    parser.add_argument(
        "--out-base",
        type=Path,
        default=root / "output" / "learning_rate" / "UR5_trajectory",
        help="Base output directory (default: output/learning_rate/UR5_trajectory)",
    )
    parser.add_argument("--min-n", type=int, default=50, help="Smallest N (default 50)")
    parser.add_argument("--max-n", type=int, default=900, help="Largest N inclusive (default 900)")
    parser.add_argument("--step-n", type=int, default=50, help="Step between sizes (default 50)")
    parser.add_argument("--seed", type=int, default=42, help="Master RNG seed for file selection")
    parser.add_argument("--num-pairs", type=int, default=100_000, help="prepare_trajectory_dataset --num_pairs")
    parser.add_argument("--prep-seed-mode", choices=("global", "per_n"), default="per_n", help="Seed for pair sampling")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--batches-per-epoch", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned steps without running prepare/train",
    )
    args = parser.parse_args()

    h5_root = args.h5_root.resolve() if args.h5_root else (root / "output" / "data_collection" / "all")
    all_h5 = sorted(h5_root.glob("*.h5"))
    if not all_h5:
        raise FileNotFoundError(f"No .h5 files under {h5_root}")

    out_base = args.out_base.resolve()
    sizes = list(range(args.min_n, args.max_n + 1, args.step_n))
    for n in sizes:
        if n > len(all_h5):
            raise ValueError(f"Need at least {n} HDF5 files, found {len(all_h5)} in {h5_root}")

    if not prep_script.is_file() or not train_script.is_file():
        raise FileNotFoundError(f"Missing scripts under {ntrl_demo}")

    rng = random.Random(args.seed)

    for n in sizes:
        out_dir = out_base / str(n)
        h5_subset = out_dir / "h5_subset"
        out_dir.mkdir(parents=True, exist_ok=True)

        pool = list(all_h5)
        rng.shuffle(pool)
        chosen = sorted(pool[:n], key=lambda p: p.name)

        manifest = {
            "n_h5": n,
            "master_seed": args.seed,
            "h5_root": str(h5_root),
            "files": [str(p.resolve()) for p in chosen],
        }
        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        print(f"\n=== data size {n} -> {out_dir} ===")
        if args.dry_run:
            print(f"Would symlink {len(chosen)} files to {h5_subset}")
            continue

        _prepare_subset_symlinks(chosen, h5_subset)

        prep_seed = args.seed if args.prep_seed_mode == "global" else (args.seed + n)
        prep_cmd = [
            sys.executable,
            str(prep_script),
            "--data_dir",
            str(h5_subset),
            "--output_dir",
            str(out_dir),
            "--num_pairs",
            str(args.num_pairs),
            "--seed",
            str(prep_seed),
        ]
        print("+", " ".join(prep_cmd))
        subprocess.check_call(prep_cmd)

        train_cmd = [
            sys.executable,
            str(train_script),
            "--data_path",
            str(out_dir),
            "--model_path",
            str(out_dir),
            "--device",
            args.device,
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--save_every",
            str(args.save_every),
            "--print_every",
            str(args.print_every),
        ]
        if args.batches_per_epoch is not None:
            train_cmd += ["--batches_per_epoch", str(args.batches_per_epoch)]

        print("+", " ".join(train_cmd), f"(cwd={ntrl_demo})")
        subprocess.check_call(train_cmd, cwd=str(ntrl_demo))


if __name__ == "__main__":
    main()
