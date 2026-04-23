# Preprocess H5 multi-obj demos into shards: image -> 3 x z_goal latent.
import sys
sys.path.append("/media/corallab-s1/4tbhdd/junheelim/PI-VLA/ntrl-demo")

from pathlib import Path
from typing import Optional, Tuple, List
import os

import h5py
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

DATASET_ROOT = Path("/media/corallab-s1/4tbhdd/junheelim/PI-VLA/output/multi_obj/20260421")
OUTPUT_DIR = Path("/media/corallab-s1/4tbhdd/junheelim/PI-VLA/output/mult_obj_img2latent/shards")
IMG_SIZE = 224
SHARD_SIZE = 5000

TEACHER_CHECKPOINT = "/media/corallab-s1/4tbhdd/junheelim/PI-VLA/ntrl-demo/Experiments/UR5_trajectory_no_wall_accuracy_check/trajectory_03_25_20_28/Model_Epoch_05000_ValLoss_7.820605e-01.pt"
TEACHER_DATA_PATH: Optional[str] = None
NORMALIZE_COORDS = True

# NTField input normalization
SCALE = float(np.pi / 0.5)

# ---------------------------------------------------------------------------
# Teacher helpers
# ---------------------------------------------------------------------------

def load_teacher(
    checkpoint: str,
    device: torch.device,
    data_path: Optional[str] = None,
) -> Tuple[nn.Module, int]:
    from models.metric_arm import model_test_metric as md

    model_path = os.path.dirname(os.path.abspath(checkpoint))
    if data_path is None:
        data_path = os.path.join("datasets", "arm", "UR5_trajectory")
    model = md.Model(model_path, data_path, dim=6, source=[0.0] * 6, device=str(device))
    model.load(checkpoint)
    model.network.eval()
    for p in model.network.parameters():
        p.requires_grad_(False)

    h = 256
    if hasattr(model.network, "encoder") and len(model.network.encoder) > 0:
        lin = model.network.encoder[-1]
        if hasattr(lin, "out_features"):
            h = int(lin.out_features)
    return model.network, h

def build_coords_batch(
    q_start: torch.Tensor, q_goal: torch.Tensor, use_scale: bool
) -> torch.Tensor:
    if use_scale:
        q_start = q_start / SCALE
        q_goal = q_goal / SCALE
    return torch.cat([q_start, q_goal], dim=1)


@torch.no_grad()
def compute_z_goal(
    B: nn.Module,
    qs: torch.Tensor,
    qg: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    coords = build_coords_batch(qs, qg, NORMALIZE_COORDS).to(device)
    _, zg = B.encode_pair_latents(coords)
    return zg.cpu()

# ---------------------------------------------------------------------------
# H5 helpers
# ---------------------------------------------------------------------------

def discover_h5_files(root: Path) -> List[Path]:
    out: List[Path] = []
    for p in sorted(root.rglob("*.h5")):
        if "__MACOSX" in p.parts:
            continue
        if p.name.startswith("._"):
            continue
        out.append(p)
    return out

def process_img(img: np.ndarray, img_size: int) -> torch.Tensor:
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    h, w = img.shape[:2]
    # Center crop to square
    size = min(h, w)
    y_start = (h - size) // 2
    x_start = (w - size) // 2
    img = img[y_start:y_start+size, x_start:x_start+size]
    
    if size != img_size:
        ys = np.linspace(0, size - 1, img_size).astype(int)
        xs = np.linspace(0, size - 1, img_size).astype(int)
        img = img[np.ix_(ys, xs)]
    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    print("Loading teacher encoder...", flush=True)
    teacher, z_dim = load_teacher(TEACHER_CHECKPOINT, device, TEACHER_DATA_PATH)
    teacher = teacher.to(device)
    print(f"Teacher loaded. z_goal dim: {z_dim}", flush=True)

    files = discover_h5_files(DATASET_ROOT)
    if not files:
        raise ValueError(f"No .h5 files found under {DATASET_ROOT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    shard_images: List[torch.Tensor] = []
    shard_z_goals: List[torch.Tensor] = []   # (3, H) per sample
    shard_paths: List[str] = []
    shard_idx = 0
    total_samples = 0
    skipped_files = 0

    def flush_shard() -> None:
        nonlocal shard_images, shard_z_goals, shard_idx
        if not shard_images:
            return
        images_tensor = torch.stack(shard_images, dim=0)    # (N, 3, 224, 224)
        z_goals_tensor = torch.stack(shard_z_goals, dim=0)  # (N, 3, H)
        shard_path = OUTPUT_DIR / f"grasp_multi_dataset_shard_{shard_idx:05d}.pt"
        torch.save(
            {
                "images": images_tensor,
                "z_goals": z_goals_tensor,
                "img_size": IMG_SIZE,
                "z_dim": z_dim,
            },
            shard_path,
        )
        shard_paths.append(str(shard_path))
        print(f"Saved shard {shard_idx} with {images_tensor.shape[0]} samples -> {shard_path}")
        shard_idx += 1
        shard_images = []
        shard_z_goals = []

    print(f"Found {len(files)} h5 files. Processing...", flush=True)

    for path in tqdm(files, desc="Files"):
        try:
            with h5py.File(path, "r") as f:
                if "start_image" not in f:
                    raise KeyError("missing 'start_image'")
                if "trajectory_joint_configs" not in f or "goal_joint_configs" not in f:
                    raise KeyError("missing joint configs")

                img = f["start_image"][:]
                qs_np = f["trajectory_joint_configs"][:, 0, :]  # (3, 6)
                qg_np = f["goal_joint_configs"][:]  # (3, 6)

            if qs_np.shape[0] != 3 or qg_np.shape[0] != 3:
                raise ValueError(f"Expected 3 objects, got {qs_np.shape[0]}")

            qs_t = torch.from_numpy(qs_np)  # (3, 6)
            qg_t = torch.from_numpy(qg_np)  # (3, 6)
            
            # compute z_goal for all 3 objects at once
            z_goals = compute_z_goal(teacher, qs_t, qg_t, device)  # (3, H)

            total_samples += 1

            shard_images.append(process_img(img, IMG_SIZE))
            shard_z_goals.append(z_goals)
            
            if len(shard_images) >= SHARD_SIZE:
                flush_shard()

        except (OSError, KeyError, ValueError) as e:
            skipped_files += 1
            print(f"Skipping invalid file: {path} ({e})")
            continue

    flush_shard()

    if total_samples == 0:
        raise RuntimeError(f"No valid samples found. Checked {len(files)} files, skipped {skipped_files}.")

    manifest_path = OUTPUT_DIR / "manifest.pt"
    torch.save(
        {
            "num_samples": total_samples,
            "num_shards": len(shard_paths),
            "shards": shard_paths,
            "img_size": IMG_SIZE,
            "z_dim": z_dim,
        },
        manifest_path,
    )

    print(f"Done. Processed samples: {total_samples} | z_dim: {z_dim} | Skipped files: {skipped_files} | Shards: {len(shard_paths)} | Manifest: {manifest_path}")

if __name__ == "__main__":
    main()
