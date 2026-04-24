#!/usr/bin/env python3
"""
Build cumulative merged H5 directories: each cumulative_K contains symlinks to all
*.h5 from chunk_000 .. chunk_K (inclusive), in chunk order.

Expects chunk layout from make_chunks.py:
  chunks/seed0/chunk_000/*.h5, chunk_001, ...

Usage:
  python build_cumulative_merged_h5.py \\
    --chunks_root chunks/seed0 \\
    --merged_root merged_h5/seed0
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List


def sorted_chunk_dirs(chunks_root: Path) -> List[Path]:
    dirs = [p for p in chunks_root.iterdir() if p.is_dir() and p.name.startswith("chunk_")]
    return sorted(dirs, key=lambda p: p.name)


def main() -> None:
    ap = argparse.ArgumentParser(description="Cumulative symlink merge of chunk H5 dirs")
    ap.add_argument(
        "--chunks_root",
        type=Path,
        default=Path(__file__).resolve().parent / "chunks" / "seed0",
        help="Directory containing chunk_000, chunk_001, ...",
    )
    ap.add_argument(
        "--merged_root",
        type=Path,
        default=Path(__file__).resolve().parent / "merged_h5" / "seed0",
        help="Output root for cumulative_000, cumulative_001, ...",
    )
    ap.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing cumulative_* under merged_root before building",
    )
    args = ap.parse_args()

    chunks_root = args.chunks_root.resolve()
    merged_root = args.merged_root.resolve()

    if not chunks_root.is_dir():
        raise SystemExit(f"chunks_root not found: {chunks_root}")

    chunk_dirs = sorted_chunk_dirs(chunks_root)
    if not chunk_dirs:
        raise SystemExit(f"No chunk_* directories under {chunks_root}")

    if args.clean and merged_root.exists():
        import shutil

        for p in merged_root.iterdir():
            if p.is_dir() and p.name.startswith("cumulative_"):
                shutil.rmtree(p)

    merged_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "chunks_root": str(chunks_root),
        "merged_root": str(merged_root),
        "levels": [],
    }

    for k in range(len(chunk_dirs)):
        out_dir = merged_root / f"cumulative_{k:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        n_links = 0
        for j in range(k + 1):
            ch = chunk_dirs[j]
            for h5 in sorted(ch.glob("*.h5")):
                dst = out_dir / h5.name
                target = os.path.relpath(h5.resolve(), out_dir)
                if dst.is_symlink() or dst.exists():
                    dst.unlink()
                dst.symlink_to(target)
                n_links += 1
        summary["levels"].append({"id": out_dir.name, "num_files": n_links})

    manifest_path = merged_root / "cumulative_manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {manifest_path}")
    for row in summary["levels"]:
        print(f"  {row['id']}: {row['num_files']} files")


if __name__ == "__main__":
    main()
