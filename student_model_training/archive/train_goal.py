from pathlib import Path
from typing import Optional, Tuple
import os

import torch
from torch import nn
from torchvision import models, transforms

DATASET_ROOT = Path("/scratch/scholar/sohn31/grasp_dataset_shards")
TEACHER_CHECKPOINT = "/path/to/teacher/checkpoint.pth"   # <-- update
TEACHER_DATA_PATH: Optional[str] = None                  # set if needed
NORMALIZE_COORDS = True                                   # match your args

LOG_EVERY_BATCHES = 100


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
        from models.metric_arm import _NTRL_DEMO
        data_path = os.path.join(_NTRL_DEMO, "datasets", "arm", "UR5_trajectory")
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
    qs: torch.Tensor,
    qg: torch.Tensor,
    normalize: bool,
) -> torch.Tensor:
    """
    Replicate whatever build_coords_batch does in your codebase.
    qs, qg: (B, D) joint configs.
    Returns coords tensor expected by teacher.encode_pair_latents.
    """
    # If you have the real build_coords_batch, import and use it directly.
    # This is a passthrough placeholder — adjust to match yours.
    from models.metric_arm.utils import build_coords_batch as _build
    return _build(qs, qg, normalize)


@torch.no_grad()
def compute_z_goal(
    teacher: nn.Module,
    qs: torch.Tensor,
    qg: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Encode (qs, qg) with frozen teacher and return z_goal only."""
    coords = build_coords_batch(qs, qg, NORMALIZE_COORDS).to(device)
    _, zg = teacher.encode_pair_latents(coords)
    return zg  # (B, H)


# ---------------------------------------------------------------------------
# Shard discovery
# ---------------------------------------------------------------------------

def discover_shards(root: Path) -> Tuple[list, list]:
    shard_files = sorted(root.glob("grasp_dataset_shard_*.pt"))
    if not shard_files:
        raise ValueError(f"No shard files found under {root}")

    cumulative = []
    total = 0
    print("Scanning shard files to build index...", flush=True)
    for shard_path in shard_files:
        shard = torch.load(shard_path, map_location="cpu")
        for key in ("images", "configs"):
            if key not in shard:
                raise ValueError(f"Missing key '{key}' in {shard_path}")
        n = shard["images"].shape[0]
        total += n
        cumulative.append(total)

    print(f"Found {total} total samples across {len(shard_files)} shards.", flush=True)
    return shard_files, cumulative


def extract_start_goal_configs(
    configs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    configs: (N, T, D)  — T timesteps, D joint dims
             or (N, D)  — treated as goal only (qs = zeros)
    Returns qs (N, D) and qg (N, D).
    """
    if configs.dim() == 3:
        qs = configs[:, 0, :]   # first timestep = start
        qg = configs[:, -1, :]  # last  timestep = goal
    elif configs.dim() == 2:
        qg = configs
        qs = torch.zeros_like(qg)
    else:
        raise ValueError(f"Unexpected configs shape: {configs.shape}")
    return qs, qg


# ---------------------------------------------------------------------------
# Train / val split
# ---------------------------------------------------------------------------

def build_train_val_split(
    n_total: int, val_fraction: float = 0.1, seed: int = 42
) -> Tuple[set, set, int, int]:
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=generator)
    val_size = int(val_fraction * n_total)
    train_size = n_total - val_size
    return (
        set(perm[:train_size].tolist()),
        set(perm[train_size:].tolist()),
        train_size,
        val_size,
    )


def train_val_indices_for_shard(
    shard_idx: int, cumulative: list, train_idx: set, val_idx: set
) -> Tuple[list, list]:
    start = 0 if shard_idx == 0 else cumulative[shard_idx - 1]
    shard_len = cumulative[shard_idx] - start
    train_locals, val_locals = [], []
    for j in range(shard_len):
        g = start + j
        if g in train_idx:
            train_locals.append(j)
        elif g in val_idx:
            val_locals.append(j)
    return train_locals, val_locals


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_batches(
    model: nn.Module,
    teacher: nn.Module,
    optimizer,
    criterion,
    device: torch.device,
    shard_path: Path,
    local_indices: list,
    normalize,
    batch_size: int,
    train: bool,
    epoch: int,
    epochs: int,
    batch_counter: list,
) -> Tuple[float, int]:
    if not local_indices:
        return 0.0, 0

    shard = torch.load(shard_path, map_location="cpu")
    images = shard["images"]
    qs_all, qg_all = extract_start_goal_configs(shard["configs"])

    perm = (
        torch.randperm(len(local_indices))
        if train
        else torch.arange(len(local_indices))
    )
    locals_shuffled = [local_indices[i] for i in perm.tolist()]

    running_loss = 0.0
    n_batches = 0
    model.train() if train else model.eval()

    for start in range(0, len(locals_shuffled), batch_size):
        chunk = locals_shuffled[start : start + batch_size]

        x = images[chunk].to(device, non_blocking=True)
        x = normalize(x)

        # Encode (qs, qg) → z_goal with frozen teacher
        qs_b = qs_all[chunk].to(device, non_blocking=True)
        qg_b = qg_all[chunk].to(device, non_blocking=True)
        y = compute_z_goal(teacher, qs_b, qg_b, device)   # (B, H)

        if train:
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        else:
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)
            running_loss += loss.item()

        n_batches += 1
        batch_counter[0] += 1

        if train and batch_counter[0] % LOG_EVERY_BATCHES == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} | batch {batch_counter[0]} | "
                f"last loss {loss.item():.6f}",
                flush=True,
            )

    return running_loss, n_batches


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_shardwise(
    model: nn.Module,
    teacher: nn.Module,
    device: torch.device,
    shard_files: list,
    cumulative: list,
    val_idx: set,
    normalize,
    batch_size: int,
) -> Tuple[float, float]:
    model.eval()
    mse_sum = 0.0
    mae_sum = 0.0
    n_samples = 0
    mse_crit = nn.MSELoss(reduction="sum")
    mae_crit = nn.L1Loss(reduction="sum")

    for si, shard_path in enumerate(shard_files):
        _, val_locals = train_val_indices_for_shard(si, cumulative, set(), val_idx)
        if not val_locals:
            continue

        shard = torch.load(shard_path, map_location="cpu")
        images = shard["images"]
        qs_all, qg_all = extract_start_goal_configs(shard["configs"])

        for start in range(0, len(val_locals), batch_size):
            chunk = val_locals[start : start + batch_size]
            x = images[chunk].to(device, non_blocking=True)
            x = normalize(x)

            qs_b = qs_all[chunk].to(device, non_blocking=True)
            qg_b = qg_all[chunk].to(device, non_blocking=True)
            y = compute_z_goal(teacher, qs_b, qg_b, device)

            pred = model(x)
            n_samples += x.size(0)
            mse_sum += mse_crit(pred, y).item()
            mae_sum += mae_crit(pred, y).item()

    if n_samples == 0:
        return 0.0, 0.0
    return mse_sum / n_samples, mae_sum / n_samples


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def get_model(output_dim: int) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, output_dim)
    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # Load frozen teacher encoder
    print("Loading teacher encoder...", flush=True)
    teacher, z_dim = load_teacher(TEACHER_CHECKPOINT, device, TEACHER_DATA_PATH)
    teacher = teacher.to(device)
    print(f"Teacher loaded. z_goal dim: {z_dim}", flush=True)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    shard_files, cumulative = discover_shards(DATASET_ROOT)
    n_total = cumulative[-1]
    train_idx, val_idx, train_size, val_size = build_train_val_split(n_total)
    print(f"Train: {train_size} | Val: {val_size}", flush=True)

    model = get_model(output_dim=z_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    epochs = 20
    eval_every = 2
    batch_size = 128
    best_val_mae = float("inf")

    for epoch in range(epochs):
        shard_order = torch.randperm(len(shard_files)).tolist()
        running_loss = 0.0
        n_batches_total = 0
        batch_counter = [0]

        for si in shard_order:
            train_locals, _ = train_val_indices_for_shard(
                si, cumulative, train_idx, val_idx
            )
            loss_sum, nb = run_batches(
                model, teacher, optimizer, criterion, device,
                shard_files[si], train_locals, normalize,
                batch_size, True, epoch, epochs, batch_counter,
            )
            running_loss += loss_sum
            n_batches_total += nb

        avg_train_loss = running_loss / max(n_batches_total, 1)
        print(
            f"Epoch {epoch + 1}/{epochs} | Train MSE: {avg_train_loss:.6f}",
            end="",
            flush=True,
        )

        if (epoch + 1) % eval_every == 0 or (epoch + 1) == epochs:
            val_mse, val_mae = evaluate_shardwise(
                model, teacher, device,
                shard_files, cumulative, val_idx,
                normalize, batch_size,
            )
            print(f" | Val MSE: {val_mse:.6f} | Val MAE: {val_mae:.6f}", flush=True)
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                torch.save(model.state_dict(), "best_z_goal_model.pth")
                print(
                    f" -> New best model saved. Val MAE: {best_val_mae:.6f}",
                    flush=True,
                )
        else:
            print(flush=True)


if __name__ == "__main__":
    train()