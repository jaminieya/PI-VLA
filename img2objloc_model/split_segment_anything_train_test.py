#!/usr/bin/env python3
"""Move existing output/segment_anything/<timestamp>/ samples into train/ and test/ (default 80/20)."""

import argparse
import os
import random
import shutil
from typing import List


def list_sample_dirs(segment_root: str) -> List[str]:
    train_dir = os.path.join(segment_root, "train")
    test_dir = os.path.join(segment_root, "test")
    out: List[str] = []
    for name in sorted(os.listdir(segment_root)):
        path = os.path.join(segment_root, name)
        if not os.path.isdir(path):
            continue
        if path in (train_dir, test_dir):
            continue
        if name in ("train", "test"):
            continue
        meta = os.path.join(path, "object_location.json")
        if not os.path.exists(meta):
            continue
        out.append(name)
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Split segment_anything timestamp folders into train/ and test/."
    )
    p.add_argument(
        "--segment-root",
        type=str,
        default="output/segment_anything",
        help="Root that currently holds <timestamp>/ directories",
    )
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if not 0.0 < args.train_ratio < 1.0:
        raise SystemExit("--train-ratio must be between 0 and 1")

    segment_root = os.path.abspath(args.segment_root)
    train_root = os.path.join(segment_root, "train")
    test_root = os.path.join(segment_root, "test")

    sample_names = list_sample_dirs(segment_root)
    if not sample_names:
        raise SystemExit(f"No sample directories found under {segment_root}")

    rng = random.Random(args.seed)
    rng.shuffle(sample_names)

    n = len(sample_names)
    n_train = int(n * args.train_ratio)
    n_train = max(1, min(n_train, n - 1)) if n > 1 else n
    train_names = set(sample_names[:n_train])
    test_names = set(sample_names[n_train:])

    os.makedirs(train_root, exist_ok=True)
    os.makedirs(test_root, exist_ok=True)

    print(f"segment_root: {segment_root}")
    print(f"samples: {n} | train: {len(train_names)} | test: {len(test_names)} | seed: {args.seed}")
    if args.dry_run:
        print("dry-run: no moves performed")
        print("train (first 5):", sorted(train_names)[:5])
        print("test  (first 5):", sorted(test_names)[:5])
        return

    for name in sorted(train_names):
        src = os.path.join(segment_root, name)
        dst = os.path.join(train_root, name)
        if os.path.exists(dst):
            raise SystemExit(f"Refusing to overwrite existing destination: {dst}")
        shutil.move(src, dst)
        print(f"mv train: {name}")

    for name in sorted(test_names):
        src = os.path.join(segment_root, name)
        dst = os.path.join(test_root, name)
        if os.path.exists(dst):
            raise SystemExit(f"Refusing to overwrite existing destination: {dst}")
        shutil.move(src, dst)
        print(f"mv test:  {name}")

    print("Done.")


if __name__ == "__main__":
    main()
