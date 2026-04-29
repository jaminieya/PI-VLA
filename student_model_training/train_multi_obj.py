from pathlib import Path
import numpy as np
import torch
from torch import nn
from torchvision import models, transforms
import os

DATASET_ROOT = Path("/media/corallab-s1/4tbhdd/junheelim/PI-VLA/output/mult_obj_img2latent/shards")
PCA_OUTPUT_DIR = Path("/media/corallab-s1/4tbhdd/junheelim/PI-VLA/output/pca_training_plots_multi_obj")
LOG_EVERY_BATCHES = 50

def build_resnet18():
    if hasattr(models, "ResNet18_Weights"):
        return models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return models.resnet18(pretrained=True)

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def discover_shards(root: Path):
    shard_files = sorted(root.glob("grasp_multi_dataset_shard_*.pt"))
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
        labels = shard["z_goals"] # (N, 3, H)
        n = shard["images"].shape[0]
        total += n
        cumulative.append(total)
        if z_dim is None:
            z_dim = int(labels.shape[-1])

    print(f"Found {total} total samples across {len(shard_files)} shards. z_dim: {z_dim}", flush=True)
    return shard_files, cumulative, z_dim

def build_train_val_split(n_total: int, val_fraction: float = 0.1, seed: int = 42):
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=generator)
    val_size = int(val_fraction * n_total)
    train_size = n_total - val_size
    return set(perm[:train_size].tolist()), set(perm[train_size:].tolist()), train_size, val_size

def train_val_indices_for_shard(shard_idx: int, cumulative: list, train_idx: set, val_idx: set):
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
        # pred and target are (B, 3, H)
        # flatten to (B*3, H) for cosine similarity
        pred_flat = pred.view(-1, pred.shape[-1])
        target_flat = target.view(-1, target.shape[-1])
        
        mse_loss = self.mse(pred, target)
        cos_loss = torch.mean(1 - self.cosine(pred_flat, target_flat))
        return (self.alpha * mse_loss) + ((1 - self.alpha) * cos_loss)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class StudentHeadWonormMulti(nn.Module):
    def __init__(self, in_features, output_dim, num_objects=3):
        super().__init__()
        self.num_objects = num_objects
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, output_dim * num_objects),
        )

    def forward(self, x):
        out = self.net(x)
        return out.view(-1, self.num_objects, self.output_dim)

class StudentModelWonormMulti(nn.Module):
    def __init__(self, output_dim: int, num_objects: int = 3):
        super().__init__()
        backbone = build_resnet18()
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = StudentHeadWonormMulti(in_features, output_dim, num_objects)

    def forward(self, x):
        return self.head(self.backbone(x))

def get_model_wonorm_multi(output_dim: int) -> nn.Module:
    return StudentModelWonormMulti(output_dim, num_objects=3)

# ---------------------------------------------------------------------------
# Training Logic
# ---------------------------------------------------------------------------

def run_batches(
    model, optimizer, criterion, device, shard_path: Path, local_indices: list,
    normalize, batch_size: int, train: bool, epoch: int, epochs: int, batch_counter: list
):
    if not local_indices:
        return 0.0, 0

    shard = torch.load(shard_path, map_location="cpu")
    images = shard["images"]
    z_goals = shard["z_goals"]  # (N, 3, H)

    perm = torch.randperm(len(local_indices)) if train else torch.arange(len(local_indices))
    locals_shuffled = [local_indices[i] for i in perm.tolist()]

    running_loss = 0.0
    n_batches = 0
    model.train() if train else model.eval()

    for start in range(0, len(locals_shuffled), batch_size):
        chunk = locals_shuffled[start : start + batch_size]
        x = images[chunk].to(device, non_blocking=True)
        y = z_goals[chunk].to(device, non_blocking=True)  # (B, 3, H)
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
            print(f"Epoch {epoch + 1}/{epochs} | batch {batch_counter[0]} | last loss {loss.item():.6f}", flush=True)

    return running_loss, n_batches

@torch.no_grad()
def evaluate_shardwise(model, device, shard_files, cumulative, val_idx, normalize, batch_size):
    model.eval()
    mse_sum = 0.0
    mae_sum = 0.0
    cos_sum = 0.0
    n_samples = 0

    mse_criterion = nn.MSELoss(reduction="sum")
    mae_criterion = nn.L1Loss(reduction="sum")

    for si, shard_path in enumerate(shard_files):
        _, val_locals = train_val_indices_for_shard(si, cumulative, set(), val_idx)
        if not val_locals:
            continue
        shard = torch.load(shard_path, map_location="cpu")
        images = shard["images"]
        z_goals = shard["z_goals"]

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
            
            pred_flat = pred.view(-1, pred.shape[-1])
            y_flat = y.view(-1, y.shape[-1])
            cos_sim = nn.functional.cosine_similarity(pred_flat, y_flat, dim=1)
            cos_sum += (1 - cos_sim).sum().item()

    if n_samples == 0:
        return 0.0, 0.0, 0.0, 0
    # cos_sum was computed over B*3 elements, so we divide by n_samples * 3
    return mse_sum / (n_samples * 3), mae_sum / (n_samples * 3), cos_sum / (n_samples * 3), n_samples

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    shard_files, cumulative, z_dim = discover_shards(DATASET_ROOT)
    n_total = cumulative[-1]
    train_idx, val_idx, train_size, val_size = build_train_val_split(n_total)
    print(f"Train: {train_size} | Val: {val_size}", flush=True)

    model = get_model_wonorm_multi(output_dim=z_dim).to(device)

    epochs = 50
    eval_every = 2
    batch_size = 64
    best_val_mae = float("inf")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = HybridDistillationLoss(alpha=0.5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        shard_order = torch.randperm(len(shard_files)).tolist()
        running_loss = 0.0
        n_batches_total = 0
        batch_counter = [0]

        for si in shard_order:
            train_locals, _ = train_val_indices_for_shard(si, cumulative, train_idx, val_idx)
            loss_sum, nb = run_batches(
                model, optimizer, criterion, device,
                shard_files[si], train_locals, normalize,
                batch_size, True, epoch, epochs, batch_counter,
            )
            running_loss += loss_sum
            n_batches_total += nb

        scheduler.step()
        avg_train_loss = running_loss / max(n_batches_total, 1)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.8f}", flush=True)

        if (epoch + 1) % eval_every == 0 or (epoch + 1) == epochs:
            val_mse, val_mae, val_cos, val_n = evaluate_shardwise(
                model, device, shard_files, cumulative, val_idx, normalize, batch_size,
            )
            print(f" -> Val MSE: {val_mse:.6f} | Val MAE: {val_mae:.6f} | Val Cos: {val_cos:.6f} | N: {val_n}", flush=True)

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "z_dim": z_dim,
                    "epoch": epoch + 1,
                    "val_mae": best_val_mae,
                }, "best_z_goal_model_multi_obj.pth")
                print(f" -> New best saved. MAE: {best_val_mae:.6f}", flush=True)

if __name__ == "__main__":
    train()
