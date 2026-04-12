#!/usr/bin/env python3
"""
preprocess_multi_object_data.py

Preprocesses grasp_multi* HDF5 demos into sharded .pt files for training.

Each HDF5 file contains ONE scene with N objects. We expand it into N independent
training samples — one per object — each with a generated prompt:

    (image, object_location, prompt)  e.g. ("grasp banana", loc_xyz, img_tensor)

So a file with 3 objects → 3 training samples sharing the same start_image.

Each HDF5 file is expected to contain:
  - /start_image          (720, 1280, 3) uint8
  - /object_locations     (N, 3)         float32   -- XYZ per object
  - /object_names         (N,)           object    -- e.g. b'banana'
  - @num_objects          int            (attribute, used for validation)

Each shard .pt stores a dict with:
  Tensors (stackable, shape [B, ...]):
    "image"            (B, 3, S, S)  float32
    "object_location"  (B, 3)        float32   -- XYZ of the target object
  Metadata lists (length B):
    "prompt"           list[str]     -- e.g. "grasp banana"
    "object_name"      list[str]     -- e.g. "banana"
    "source_file"      list[str]     -- originating .h5 path
    "object_idx"       list[int]     -- index within the scene's object array

A manifest.pt is written at the end with shard metadata.

Usage:
    python preprocess_multi_object_data.py
    python preprocess_multi_object_data.py --dataset-root /path/to/data --output-dir /path/to/out
    python preprocess_multi_object_data.py --img-size 224 --shard-size 1000 --verbose
"""
from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_DATASET_ROOT = Path("/home/hojinsohn/VLM-NT/PI-VLA/output/multi_obj_layout")
DEFAULT_OUTPUT_DIR   = Path("/home/hojinsohn/VLM-NT/PI-VLA/output/multi_obj_layout_shards")
DEFAULT_IMG_SIZE     = 224
DEFAULT_SHARD_SIZE   = 500


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_h5_files(root: Path) -> list[Path]:
    """
    Recursively find all real HDF5 files under root.
    Skips macOS zip artifacts (__MACOSX dirs and ._*.h5 AppleDouble files).
    """
    out: list[Path] = []
    for p in sorted(root.rglob("*.h5")):
        if "__MACOSX" in p.parts:
            continue
        if p.name.startswith("._"):
            continue
        out.append(p)
    return out


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------

def process_img(img: np.ndarray, img_size: int) -> torch.Tensor:
    """
    Resize (H, W, 3) uint8 numpy array -> (3, img_size, img_size) float32 tensor in [0, 1].
    Uses nearest-neighbour resize (no external dependency).
    """
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]

    h, w = img.shape[:2]
    if h != img_size or w != img_size:
        ys = np.linspace(0, h - 1, img_size, dtype=np.float32).astype(np.int32)
        xs = np.linspace(0, w - 1, img_size, dtype=np.float32).astype(np.int32)
        img = img[np.ix_(ys, xs)]

    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------

def make_prompt(object_name: str) -> str:
    """Generate a grasp instruction prompt for a given object name."""
    return f"grasp {object_name}"


# ---------------------------------------------------------------------------
# Per-file loading  →  N samples (one per object)
# ---------------------------------------------------------------------------

def load_samples(path: Path, img_size: int) -> list[dict]:
    """
    Load one HDF5 file and return a list of N flat sample dicts,
    one per object in the scene. All samples share the same image.

    Each HDF5 file must contain:
        /start_image        (H, W, 3)  uint8
        /object_locations   (N, 3)     float32
        /object_names       (N,)       bytes/str
        @num_objects                   int  (attribute, optional but validated if present)

    Each returned dict has:
        image            torch.Tensor  (3, S, S)  float32
        object_location  torch.Tensor  (3,)       float32
        prompt           str
        object_name      str
        object_idx       int
        source_file      str
    """
    required_datasets = {"start_image", "object_locations", "object_names"}

    with h5py.File(path, "r") as f:
        missing = required_datasets - set(f.keys())
        if missing:
            raise KeyError(f"Missing datasets: {missing}")

        # ── Validate num_objects attribute if present ──────────────────────
        if "num_objects" in f.attrs:
            declared = int(f.attrs["num_objects"])
            actual   = f["object_locations"].shape[0]
            if declared != actual:
                raise ValueError(
                    f"@num_objects attribute ({declared}) does not match "
                    f"object_locations rows ({actual})"
                )

        # ── Load image (shared across all per-object samples) ──────────────
        img_np = f["start_image"][:]                        # (H, W, 3) uint8
        image  = process_img(img_np, img_size)              # (3, S, S) float32

        # ── Load per-object data ───────────────────────────────────────────
        obj_locs      = np.array(f["object_locations"], dtype=np.float32)  # (N, 3)
        obj_names_raw = f["object_names"][:]                               # (N,)

    # Decode bytes → str
    obj_names: list[str] = [
        n.decode() if isinstance(n, bytes) else str(n)
        for n in obj_names_raw
    ]

    num_objects = obj_locs.shape[0]
    if len(obj_names) != num_objects:
        raise ValueError(
            f"object_names length ({len(obj_names)}) != "
            f"object_locations rows ({num_objects})"
        )

    # ── Build one sample per object ────────────────────────────────────────
    samples: list[dict] = []
    for i in range(num_objects):
        samples.append({
            "image":           image,                              # shared tensor
            "object_location": torch.from_numpy(obj_locs[i]),     # (3,) float32
            "prompt":          make_prompt(obj_names[i]),
            "object_name":     obj_names[i],
            "object_idx":      i,
            "source_file":     str(path),
        })

    return samples


# ---------------------------------------------------------------------------
# Shard flushing
# ---------------------------------------------------------------------------

TENSOR_KEYS = ("image", "object_location")
META_KEYS   = ("prompt", "object_name", "object_idx", "source_file")


def flush_shard(
    buffer: list[dict],
    shard_idx: int,
    output_dir: Path,
    img_size: int,
) -> Path:
    """Stack buffer into tensors and save a shard .pt file."""
    stacked = {k: torch.stack([s[k] for s in buffer], dim=0) for k in TENSOR_KEYS}
    meta    = {k: [s[k] for s in buffer] for k in META_KEYS}

    shard_path = output_dir / f"shard_{shard_idx:05d}.pt"
    torch.save(
        {
            **stacked,
            **meta,
            "img_size":    img_size,
            "num_samples": len(buffer),
        },
        shard_path,
    )
    return shard_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    dataset_root: Path = args.dataset_root.expanduser().resolve()
    output_dir:   Path = args.output_dir.expanduser().resolve()
    img_size:     int  = args.img_size
    shard_size:   int  = args.shard_size

    files = discover_h5_files(dataset_root)
    if not files:
        raise ValueError(f"No .h5 files found under {dataset_root}")
    print(f"Found {len(files)} HDF5 file(s) under {dataset_root}")

    output_dir.mkdir(parents=True, exist_ok=True)

    buffer:       list[dict] = []
    shard_paths:  list[str]  = []
    shard_idx     = 0
    total_samples = 0
    skipped_files = 0

    for path in tqdm(files, desc="Files"):
        try:
            samples = load_samples(path, img_size)   # list of N dicts (N = num_objects)
        except Exception as e:
            skipped_files += 1
            tqdm.write(f"  [SKIP] {path.name}: {e}")
            if args.verbose:
                traceback.print_exc()
            continue

        if args.verbose:
            tqdm.write(
                f"  {path.name}: {len(samples)} sample(s) — "
                + ", ".join(s["object_name"] for s in samples)
            )

        for sample in samples:
            buffer.append(sample)
            total_samples += 1

            if len(buffer) >= shard_size:
                sp = flush_shard(buffer, shard_idx, output_dir, img_size)
                shard_paths.append(str(sp))
                tqdm.write(
                    f"  Saved shard {shard_idx:05d} "
                    f"({len(buffer)} samples) -> {sp.name}"
                )
                shard_idx += 1
                buffer = []

    # Final partial shard
    if buffer:
        sp = flush_shard(buffer, shard_idx, output_dir, img_size)
        shard_paths.append(str(sp))
        tqdm.write(
            f"  Saved shard {shard_idx:05d} "
            f"({len(buffer)} samples) -> {sp.name}"
        )

    if total_samples == 0:
        raise RuntimeError(
            f"No valid samples produced. "
            f"Checked {len(files)} file(s), skipped {skipped_files}."
        )

    manifest = {
        "num_samples":   total_samples,
        "num_shards":    len(shard_paths),
        "shards":        shard_paths,
        "img_size":      img_size,
        "shard_size":    shard_size,
        "dataset_root":  str(dataset_root),
        "tensor_keys":   list(TENSOR_KEYS),
        "meta_keys":     list(META_KEYS),
    }
    manifest_path = output_dir / "manifest.pt"
    torch.save(manifest, manifest_path)

    print(
        f"\nDone.\n"
        f"  H5 files   : {len(files)} found, {skipped_files} skipped\n"
        f"  Samples    : {total_samples} "
        f"(~{total_samples / max(len(files) - skipped_files, 1):.1f} per file)\n"
        f"  Shards     : {len(shard_paths)}\n"
        f"  Manifest   : {manifest_path}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT,
        help=f"Root folder containing HDF5 demos (default: {DEFAULT_DATASET_ROOT})",
    )
    p.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Where to write shards + manifest (default: {DEFAULT_OUTPUT_DIR})",
    )
    p.add_argument(
        "--img-size", type=int, default=DEFAULT_IMG_SIZE,
        help=f"Square resize target for start_image (default: {DEFAULT_IMG_SIZE})",
    )
    p.add_argument(
        "--shard-size", type=int, default=DEFAULT_SHARD_SIZE,
        help=f"Max samples per shard file (default: {DEFAULT_SHARD_SIZE})",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Print per-file object names and full tracebacks for skipped files",
    )
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())