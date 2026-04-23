#!/usr/bin/env python3
"""
Split a directory of *.h5 files into subdirectories of fixed size after a seeded shuffle.

Example:
  python make_chunks.py \\
    --source_dir ../output/data_collection/20260402 \\
    --out_root chunks/seed0 \\
    --chunk_size 50 \\
    --seed 0"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def chunk_list(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Shuffle H5 paths and symlink into chunk_NNN/")
    parser.add_argument(
        "--source_dir",
        type=Path,
        required=True,
        help="Directory containing *.h5 files (e.g. output/data_collection/20260402)",
    )
    parser.add_argument(
        "--out_root",
        type=Path,
        required=True,
        help="Output root, e.g. NTField_scaling/chunks/seed0",
    )
    parser.add_argument("--chunk_size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print plan only; do not create dirs or links",
    )
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    if not source_dir.is_dir():
        raise SystemExit(f"source_dir is not a directory: {source_dir}")

    h5_files = sorted(source_dir.glob("*.h5"))
    if not h5_files:
        raise SystemExit(f"No .h5 files under {source_dir}")

    rng = __import__("random").Random(args.seed)
    paths = list(h5_files)
    rng.shuffle(paths)

    chunks = chunk_list(paths, args.chunk_size)
    out_root = args.out_root.resolve()
    manifest = {
        "source_dir": str(source_dir),
        "out_root": str(out_root),
        "seed": args.seed,
        "chunk_size": args.chunk_size,
        "total_files": len(paths),
        "num_chunks": len(chunks),
        "chunks": [],
    }

    for idx, group in enumerate(chunks):
        chunk_name = f"chunk_{idx:03d}"
        chunk_dir = out_root / chunk_name
        manifest["chunks"].append(
            {
                "id": chunk_name,
                "count": len(group),
                "files": [p.name for p in group],
            }
        )

        if args.dry_run:
            print(f"{chunk_name}: {len(group)} files -> {chunk_dir}")
            continue

        chunk_dir.mkdir(parents=True, exist_ok=True)
        for src in group:
            dst = chunk_dir / src.name
            rel = os.path.relpath(src, chunk_dir)
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(rel)

    manifest_path = out_root / "manifest.json"
    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"Wrote {manifest_path}")
    print(
        f"Done: {len(paths)} files -> {len(chunks)} chunks "
        f"(sizes {[len(c) for c in chunks]}) under {out_root}"
    )


if __name__ == "__main__":
    main()
