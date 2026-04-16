from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import models, transforms

# Shards from preprocess_config.py (images + z_goals = teacher latent of goal).
# DATASET_ROOT = Path("/scratch/scholar/sohn31/grasp_zgoal_wonorm_dataset_shards")
DATASET_ROOT = Path("/home/hojinsohn/VLM-NT/grasp_zgoal_wonorm_dataset_shards")
PCA_OUTPUT_DIR = Path("/home/hojinsohn/VLM-NT/PI-VLA/output/pca_training_plots_wonorm")
LOG_EVERY_BATCHES = 100


def build_resnet18():
    """Build ResNet18 with pretrained weights across torchvision versions."""
    if hasattr(models, "ResNet18_Weights"):
        return models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return models.resnet18(pretrained=True)

# ---------------------------------------------------------------------------
# Shard helpers
# ---------------------------------------------------------------------------

def label_tensor_from_shard(shard: dict) -> torch.Tensor:
    """
    Returns (N, H) z_goal latents.
    Falls back to legacy 'configs' / 'obj_locs' keys for backward compatibility.
    """
    if "z_goals" in shard:
        return shard["z_goals"]       # (N, H) — pre-computed by preprocess
    if "configs" in shard:
        configs = shard["configs"]
        return configs[:, -1, :] if configs.dim() == 3 else configs
    if "obj_locs" in shard:
        return shard["obj_locs"]
    raise ValueError("Shard must contain 'z_goals', 'configs', or 'obj_locs'.")


def discover_shards(root: Path):
    shard_files = sorted(root.glob("grasp_dataset_shard_*.pt"))
    if not shard_files:
        raise ValueError(f"No shard files found under {root}")

    cumulative = []
    total = 0
    z_dim = None
    print("Scanning shard files to build index...", flush=True)
    for shard_path in shard_files:
        shard = torch.load(shard_path, map_location="cpu")
        if "images" not in shard:
            raise ValueError(f"Missing 'images' in {shard_path}")
        labels = label_tensor_from_shard(shard)
        n = shard["images"].shape[0]
        total += n
        cumulative.append(total)
        if z_dim is None:
            z_dim = int(labels.shape[-1])

    print(
        f"Found {total} total samples across {len(shard_files)} shards. "
        f"z_dim: {z_dim}",
        flush=True,
    )
    return shard_files, cumulative, z_dim


# ---------------------------------------------------------------------------
# Train / val split
# ---------------------------------------------------------------------------

def build_train_val_split(n_total: int, val_fraction: float = 0.1, seed: int = 42):
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
):
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
# Loss helpers
# ---------------------------------------------------------------------------

class HybridDistillationLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.mse = nn.MSELoss()
        self.cosine = nn.CosineSimilarity(dim=1)
        self.alpha = alpha

    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        # Cosine similarity is 1 when identical, so we minimize (1 - cos_sim)
        cos_loss = torch.mean(1 - self.cosine(pred, target))
        return (self.alpha * mse_loss) + ((1 - self.alpha) * cos_loss)


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_batches(
    model,
    optimizer,
    criterion,
    device,
    shard_path: Path,
    local_indices: list,
    normalize,
    batch_size: int,
    train: bool,
    epoch: int,
    epochs: int,
    batch_counter: list,
):
    if not local_indices:
        return 0.0, 0

    shard = torch.load(shard_path, map_location="cpu")
    images = shard["images"]
    z_goals = label_tensor_from_shard(shard)  # (N, H) — ready to use directly

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
        y = z_goals[chunk].to(device, non_blocking=True)  # (B, H)
        x = normalize(x)

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
# Fixed evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_shardwise(
    model, device, shard_files, cumulative, val_idx, normalize, batch_size
):
    model.eval()
    mse_sum  = 0.0
    mae_sum  = 0.0
    cos_sum  = 0.0  # now accumulated as sum over samples, not mean over batches
    n_samples = 0

    mse_criterion = nn.MSELoss(reduction="sum")
    mae_criterion = nn.L1Loss(reduction="sum")

    for si, shard_path in enumerate(shard_files):
        _, val_locals = train_val_indices_for_shard(si, cumulative, set(), val_idx)
        if not val_locals:
            continue
        shard  = torch.load(shard_path, map_location="cpu")
        images = shard["images"]
        z_goals = label_tensor_from_shard(shard)

        for start in range(0, len(val_locals), batch_size):
            chunk = val_locals[start : start + batch_size]
            x = images[chunk].to(device, non_blocking=True)
            y = z_goals[chunk].to(device, non_blocking=True)
            x = normalize(x)
            pred = model(x)

            b = x.size(0)
            n_samples += b
            mse_sum += mse_criterion(pred, y).item()
            mae_sum += mae_criterion(pred, y).item()
            # accumulate sum (not mean) so final division by n_samples is correct
            cos_sim = nn.functional.cosine_similarity(pred, y, dim=1)  # (B,)
            cos_sum += (1 - cos_sim).sum().item()

    if n_samples == 0:
        return 0.0, 0.0, 0.0, 0
    return (
        mse_sum  / n_samples,
        mae_sum  / n_samples,
        cos_sum  / n_samples,
        n_samples,
    )

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class StudentHeadWonorm(nn.Module):
    def __init__(self, in_features, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class StudentModelWonorm(nn.Module):
    """Defined at module level so torch.save/load and pickle work correctly."""
    def __init__(self, output_dim: int):
        super().__init__()
        backbone = build_resnet18()
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = StudentHeadWonorm(in_features, output_dim)

    def forward(self, x):
        return self.head(self.backbone(x))

def get_model_wonorm(output_dim: int) -> nn.Module:
    return StudentModelWonorm(output_dim)


# ---------------------------------------------------------------------------
# PCA helpers (no scikit-learn, matches pca_embedding_3d.py exactly)
# ---------------------------------------------------------------------------

def fit_transform_pca(x: np.ndarray, n_components: int = 3):
    """Centers x, runs SVD, returns (scores, components, explained_var_ratio)."""
    n, d = x.shape
    mean = x.mean(axis=0)
    xc = x - mean
    u, s, vt = np.linalg.svd(xc, full_matrices=False)
    explained = (s ** 2) / max(n - 1, 1)
    ratio = explained[:n_components] / explained.sum()
    scores = xc @ vt[:n_components].T
    return scores, vt[:n_components], ratio


@torch.no_grad()
def collect_val_embeddings(model, device, shard_files, cumulative, val_idx, normalize, batch_size, max_points=2000):
    """Returns (z_pred, z_target) as float32 numpy arrays, capped at max_points.
    Uses a fixed random subset of val_idx for consistency across epochs.
    """
    model.eval()

    # Fix the subset of val indices used for PCA — same every epoch
    all_val = sorted(val_idx)
    rng = np.random.default_rng(0)
    if len(all_val) > max_points:
        chosen_global = set(rng.choice(all_val, size=max_points, replace=False).tolist())
    else:
        chosen_global = set(all_val)

    preds, targets = [], []

    for si, shard_path in enumerate(shard_files):
        start_g = 0 if si == 0 else cumulative[si - 1]
        shard_len = cumulative[si] - start_g

        # find local indices that are in chosen_global
        local_indices = [
            j for j in range(shard_len)
            if (start_g + j) in chosen_global
        ]
        if not local_indices:
            continue

        shard = torch.load(shard_path, map_location="cpu")
        images = shard["images"]
        z_goals = label_tensor_from_shard(shard)

        for start in range(0, len(local_indices), batch_size):
            chunk = local_indices[start: start + batch_size]
            x = normalize(images[chunk].to(device, non_blocking=True))
            y = z_goals[chunk]
            pred = model(x).cpu()
            preds.append(pred)
            targets.append(y)

    z_pred   = torch.cat(preds,   dim=0).float().numpy()
    z_target = torch.cat(targets, dim=0).float().numpy()
    return z_pred, z_target

def generate_pca_plot(z_pred: np.ndarray, z_target: np.ndarray, epoch: int, out_dir: Path):
    """Save two PCA views: target-axes (stable) and joint-axes (full picture)."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    z_target64 = z_target.astype(np.float64)
    z_pred64 = z_pred.astype(np.float64)
    n = len(z_pred64)

    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Plot 1: fixed axes from targets only (comparable alignment view)
    # ------------------------------------------------------------------
    scores_target, components, ratio = fit_transform_pca(z_target64, n_components=3)
    mean = z_target64.mean(axis=0, keepdims=True)
    scores_pred = (z_pred64 - mean) @ components.T

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(*scores_target.T, c="#1f77b4", s=6, alpha=0.55, label="ground truth (z_goal)")
    ax.scatter(*scores_pred.T,   c="#ff7f0e", s=6, alpha=0.55, label="predicted")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(f"Epoch {epoch} — PCA (target axes)")
    ax.legend(loc="upper left", fontsize=9)

    var_txt = ", ".join(f"{v*100:.1f}%" for v in ratio)
    fig.text(0.02, 0.02, f"PC1–3 explained var: {var_txt}  |  N={n}  |  D={z_pred.shape[1]}", fontsize=9)

    fig.tight_layout()
    out_path_target = out_dir / f"pca_target_axes_epoch_{epoch:03d}.png"
    plt.savefig(out_path_target, dpi=130)
    plt.close(fig)
    print(f"  -> PCA plot saved: {out_path_target}", flush=True)

    # ------------------------------------------------------------------
    # Plot 2: PCA fitted on joint pred+target (full-variance per epoch)
    # ------------------------------------------------------------------
    x_all = np.vstack([z_pred64, z_target64])
    scores_joint, _, ratio_joint = fit_transform_pca(x_all, n_components=3)
    scores_pred_joint = scores_joint[:n]
    scores_target_joint = scores_joint[n:]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(*scores_target_joint.T, c="#1f77b4", s=6, alpha=0.55, label="ground truth (z_goal)")
    ax.scatter(*scores_pred_joint.T, c="#ff7f0e", s=6, alpha=0.55, label="predicted")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(f"Epoch {epoch} — PCA (joint axes)")
    ax.legend(loc="upper left", fontsize=9)

    var_txt_joint = ", ".join(f"{v*100:.1f}%" for v in ratio_joint)
    fig.text(0.02, 0.02, f"PC1–3 explained var: {var_txt_joint}  |  N={n}  |  D={z_pred.shape[1]}", fontsize=9)

    fig.tight_layout()
    out_path_joint = out_dir / f"pca_joint_axes_epoch_{epoch:03d}.png"
    plt.savefig(out_path_joint, dpi=130)
    plt.close(fig)
    print(f"  -> PCA plot saved: {out_path_joint}", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    shard_files, cumulative, z_dim = discover_shards(DATASET_ROOT)
    n_total = cumulative[-1]
    train_idx, val_idx, train_size, val_size = build_train_val_split(n_total)
    print(f"Train: {train_size} | Val: {val_size}", flush=True)

    model = get_model_wonorm(output_dim=z_dim).to(device)

    epochs = 20
    eval_every = 2
    batch_size = 128
    best_val_mae = float("inf")
    best_val_cos = float("inf")
    best_val_n   = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = HybridDistillationLoss(alpha=0.5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        shard_order = torch.randperm(len(shard_files)).tolist()
        running_loss    = 0.0
        n_batches_total = 0
        batch_counter   = [0]

        for si in shard_order:
            train_locals, _ = train_val_indices_for_shard(si, cumulative, train_idx, val_idx)
            loss_sum, nb = run_batches(
                model, optimizer, criterion, device,
                shard_files[si], train_locals, normalize,
                batch_size, True, epoch, epochs, batch_counter,
            )
            running_loss    += loss_sum
            n_batches_total += nb

        scheduler.step()
        avg_train_loss = running_loss / max(n_batches_total, 1)
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} "
            f"| LR: {scheduler.get_last_lr()[0]:.8f}",
            flush=True,
        )

        if (epoch + 1) % eval_every == 0 or (epoch + 1) == epochs:
            val_mse, val_mae, val_cos, val_n = evaluate_shardwise(
                model, device, shard_files, cumulative, val_idx, normalize, batch_size,
            )
            print(
                f" -> Val MSE: {val_mse:.6f} | Val MAE: {val_mae:.6f} "
                f"| Val Cos: {val_cos:.6f} | N: {val_n}",
                flush=True,
            )

            # --- PCA visualization ---
            z_pred, z_target = collect_val_embeddings(
                model, device, shard_files, cumulative, val_idx, normalize,
                batch_size=batch_size, max_points=2000,
            )
            generate_pca_plot(z_pred, z_target, epoch + 1, PCA_OUTPUT_DIR)
            # -------------------------

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_val_cos = val_cos
                best_val_n   = val_n
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "z_dim":   z_dim,
                    "epoch":   epoch + 1,
                    "val_mae": best_val_mae,
                    "val_cos": best_val_cos,
                }, "best_z_goal_model_wonorm_mse_cos.pth")
                print(
                    f" -> New best saved. MAE: {best_val_mae:.6f} "
                    f"| Cos: {best_val_cos:.6f}",
                    flush=True,
                )


if __name__ == "__main__":
    train()
