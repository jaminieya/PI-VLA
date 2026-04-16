# Preprocess H5 demos into shards: image -> z_goal latent (from teacher encoder).
# Each frame in an episode is labeled with z_goal = teacher.encode_pair_latents(qs, qg)[1].
# Output keys: "images" (N,3,224,224), "z_goals" (N,H).
from pathlib import Path
from typing import Optional
import os

import h5py
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

DATASET_ROOT = Path("/scratch/scholar/sohn31/collected_data_full_traj")
OUTPUT_DIR = Path("/scratch/scholar/sohn31/grasp_zgoal_fourier_dataset_shards")
IMG_SIZE = 224
SHARD_SIZE = 5000

TEACHER_CHECKPOINT = "teacher_model.pt"  # <-- update
TEACHER_DATA_PATH: Optional[str] = None                 # set if needed
NORMALIZE_COORDS = True

# NTField input normalization (same as planning/gradient_planner_trajectory.py)
SCALE = float(np.pi / 0.5)

# ---------------------------------------------------------------------------
# Teacher helpers (same as train_config.py)
# ---------------------------------------------------------------------------

def load_B(
    checkpoint: str,
    device: torch.device,
    data_path: Optional[str] = None,
) -> tuple[nn.Module, int]:
    from models.metric_arm import model_test_metric as md

    model_path = os.path.dirname(os.path.abspath(checkpoint))
    if data_path is None:
        data_path = os.path.join("datasets", "arm", "UR5_trajectory")
    model = md.Model(model_path, data_path, dim=6, source=[0.0] * 6, device=str(device))
    model.load(checkpoint)
    return model.B

def build_coords_batch(
    q_start: torch.Tensor, q_goal: torch.Tensor, use_scale: bool
) -> torch.Tensor:
    """q_start, q_goal: (B, 6) radians -> (B, 12) teacher input."""
    if use_scale:
        q_start = q_start / SCALE
        q_goal = q_goal / SCALE
    return torch.cat([q_start, q_goal], dim=1)


def projection_matrix(B: torch.Tensor, in_dim: int) -> torch.Tensor:
    """
    Return projection matrix with shape (in_dim, K) for x @ w.
    Supports teacher checkpoints that store B as either (in_dim, K) or (K, in_dim).
    """
    if B.ndim != 2:
        raise ValueError(f"B must be 2D, got shape {tuple(B.shape)}")
    if int(B.shape[0]) == in_dim:
        return B
    if int(B.shape[1]) == in_dim:
        return B.T
    raise ValueError(
        f"B shape {tuple(B.shape)} is incompatible with input dim {in_dim}. "
        "Expected (in_dim, K) or (K, in_dim)."
    )


def input_mapping(B: nn.Module, x: torch.Tensor) -> torch.Tensor:
    w = 2.0 * np.pi * projection_matrix(B, int(x.shape[-1]))
    x_proj = x @ w
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)    #  2*len(B)

@torch.no_grad()
def compute_z_goal(
    B: nn.Module,
    qg: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    # Keep behavior consistent with build_coords_batch(..., use_scale=NORMALIZE_COORDS).
    # Also ensure qg lives on the same device as B for matmul.
    if NORMALIZE_COORDS:
        qg = qg / SCALE
    qg = qg.to(device)
    return input_mapping(B, qg)


# ---------------------------------------------------------------------------
# H5 helpers
# ---------------------------------------------------------------------------

def discover_h5_files(root: Path) -> list[Path]:
    out: list[Path] = []
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
    if h != img_size or w != img_size:
        ys = np.linspace(0, h - 1, img_size).astype(int)
        xs = np.linspace(0, w - 1, img_size).astype(int)
        img = img[np.ix_(ys, xs)]
    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


def read_start_goal_configs(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (qs, qg) each shape (D,).
    Prefer explicit 'initial_joint_config' / 'final_joint_config' keys;
    fall back to first / last row of 'joint_configs' (T, D).
    """
    if "joint_configs" in f:
        jc = np.asarray(f["joint_configs"], dtype=np.float32)
        if jc.ndim != 2:
            raise ValueError(f"joint_configs must be 2D, got {jc.shape}")
        qs = jc[0]
        qg = jc[-1]
        # Override with explicit keys if present
        if "initial_joint_config" in f:
            qs = np.asarray(f["initial_joint_config"], dtype=np.float32).reshape(-1)
        if "final_joint_config" in f:
            qg = np.asarray(f["final_joint_config"], dtype=np.float32).reshape(-1)
        return qs, qg

    # No joint_configs at all — need both explicit keys
    if "initial_joint_config" in f and "final_joint_config" in f:
        qs = np.asarray(f["initial_joint_config"], dtype=np.float32).reshape(-1)
        qg = np.asarray(f["final_joint_config"], dtype=np.float32).reshape(-1)
        return qs, qg

    raise KeyError(
        "Need 'joint_configs' (T,D), or both "
        "'initial_joint_config' and 'final_joint_config' in H5."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    print("Loading teacher encoder...", flush=True)
    B = load_B(TEACHER_CHECKPOINT, device, TEACHER_DATA_PATH)
    print(torch.std(B))
    B = B.to(device)
    print(f"B: {B.shape}", flush=True)
    # input_mapping returns [sin(x@2πB), cos(x@2πB)] so z_dim = 2 * K.
    w = projection_matrix(B, in_dim=6)
    proj_dim = int(w.shape[1])
    z_dim = 2 * proj_dim
    print(f"Projection dim: {proj_dim} | z_dim: {z_dim}", flush=True)

    files = discover_h5_files(DATASET_ROOT)
    if not files:
        raise ValueError(f"No .h5 files found under {DATASET_ROOT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    shard_images: list[torch.Tensor] = []
    shard_z_goals: list[torch.Tensor] = []   # per-frame z_goal (all same within episode)
    shard_paths: list[str] = []
    shard_idx = 0
    total_samples = 0
    skipped_files = 0

    def flush_shard() -> None:
        nonlocal shard_images, shard_z_goals, shard_idx
        if not shard_images:
            return
        images_tensor = torch.stack(shard_images, dim=0)    # (N, 3, 224, 224)
        z_goals_tensor = torch.stack(shard_z_goals, dim=0)  # (N, H)
        shard_path = OUTPUT_DIR / f"grasp_dataset_shard_{shard_idx:05d}.pt"
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
        print(
            f"Saved shard {shard_idx} with {images_tensor.shape[0]} samples "
            f"-> {shard_path}"
        )
        shard_idx += 1
        shard_images = []
        shard_z_goals = []

    print(f"Found {len(files)} h5 files. Processing...", flush=True)

    for path in tqdm(files, desc="Files"):
        try:
            with h5py.File(path, "r") as f:
                if "images" not in f:
                    raise KeyError("missing 'images'")
                imgs = f["images"][:]
                qs_np, qg_np = read_start_goal_configs(f)

            # Encode once per episode — z_goal is the same for every frame
            qg_t = torch.from_numpy(qg_np).unsqueeze(0)  # (1, D)
            z_goal = compute_z_goal(B, qg_t, device).squeeze(0).cpu()

            num_frames = imgs.shape[0]
            total_samples += num_frames

            for i in range(num_frames):
                shard_images.append(process_img(imgs[i], IMG_SIZE))
                shard_z_goals.append(z_goal)  # same tensor ref — cheap
                if len(shard_images) >= SHARD_SIZE:
                    flush_shard()

        except (OSError, KeyError, ValueError) as e:
            skipped_files += 1
            print(f"Skipping invalid file: {path} ({e})")
            continue

    flush_shard()

    if total_samples == 0:
        raise RuntimeError(
            f"No valid samples found. Checked {len(files)} files, "
            f"skipped {skipped_files}."
        )

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

    print(
        "Done. "
        f"Processed samples: {total_samples} | "
        f"z_dim: {z_dim} | "
        f"Skipped files: {skipped_files} | "
        f"Shards: {len(shard_paths)} | "
        f"Manifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()