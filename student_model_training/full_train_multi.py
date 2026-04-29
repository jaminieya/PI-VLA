import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# wandb (optional — auto-disables if not installed)
# ---------------------------------------------------------------------------
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    _WANDB_AVAILABLE = False

# Shards from preprocess_config.py (images + z_goals = teacher latent of goal).
# DATASET_ROOT = Path("/scratch/scholar/sohn31/grasp_zgoal_wonorm_dataset_shards")
DATASET_ROOT = Path("/home/hojinsohn/VLM-NT/PI-VLA/student_model_training/data/pt_shards_multi")
PCA_OUTPUT_DIR = Path("/home/hojinsohn/VLM-NT/PI-VLA/output/pca_training_plots_multi_obj")
LOG_EVERY_BATCHES = 100

# ---------------------------------------------------------------------------
# wandb config
# ---------------------------------------------------------------------------
WANDB_ENABLED = True                          # master switch
WANDB_PROJECT = "pi-vla-latent-goal"
WANDB_ENTITY = None                           # set to your team/user name or leave None
WANDB_RUN_NAME = "multi_obj_mse_cos"                          # None → wandb auto-generates
WANDB_LOG_BATCHES = True                      # dense per-batch loss/lr logging
WANDB_BATCH_LOG_EVERY = 10                    # log every N train batches


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_opt_str(name: str, default):
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip()
    return default if v == "" else v

# ---------------------------------------------------------------------------
# Integrated-pipeline evaluation config
# ---------------------------------------------------------------------------
PI_VLA_ROOT = Path("/home/hojinsohn/VLM-NT/PI-VLA")
INTEGRATION_SCRIPT = PI_VLA_ROOT / "final_integrate" / "run_integrated_pipeline_latent_multi_obj.py"
NTFIELD_CHECKPOINT = str(PI_VLA_ROOT / "teacher_model.pt")
INTEGRATION_OUTPUT_ROOT = PI_VLA_ROOT / "output" / "integration_eval_during_training"
RUN_INTEGRATION_EVERY = 20          # run integration eval every N epochs
INTEGRATION_NUM_TRIALS = 10        # how many trials per eval
INTEGRATION_PER_TRIAL_TIMEOUT = 300 # seconds per subprocess trial
INTEGRATION_NTFIELD_TOL = 0.01
INTEGRATION_NTFIELD_MAX_STEPS = 200
INTEGRATION_NTFIELD_STEP_SIZE = 0.02
INTEGRATION_VIDEOS_PER_EVAL = 2   # how many trial videos to upload per eval (<= NUM_TRIALS)
# Temp checkpoint written every eval so the subprocess loads the *current* model.
INTEGRATION_LIVE_CKPT = PI_VLA_ROOT / "final_integrate" / "_training_live_latent.pth"


def build_resnet18():
    """Build ResNet18 with pretrained weights across torchvision versions."""
    if hasattr(models, "ResNet18_Weights"):
        return models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return models.resnet18(pretrained=True)


# ---------------------------------------------------------------------------
# wandb helpers
# ---------------------------------------------------------------------------

def _wandb_active() -> bool:
    return WANDB_ENABLED and _WANDB_AVAILABLE and wandb.run is not None


def _wandb_log(data: dict, step: Optional[int] = None) -> None:
    """Safe wandb.log wrapper — silently skips if wandb isn't running."""
    if not _wandb_active():
        return
    try:
        if step is not None:
            wandb.log(data, step=step)
        else:
            wandb.log(data)
    except Exception as e:
        print(f"[wandb] log failed: {e}", flush=True)


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


def prompt_from_object_name(object_name: str) -> str:
    """Convert object name into a simple instruction prompt."""
    return f"grasp {str(object_name).strip().lower()}"


def _tokenize_prompt(text: str):
    return re.findall(r"[a-z0-9]+", text.lower())


def build_text_vocab(object_names):
    token_set = set()
    for name in object_names:
        token_set.update(_tokenize_prompt(prompt_from_object_name(name)))
    token_to_id = {"<pad>": 0, "<unk>": 1}
    for tok in sorted(token_set):
        if tok not in token_to_id:
            token_to_id[tok] = len(token_to_id)
    return token_to_id


def encode_prompts(prompts, token_to_id, max_len):
    pad_id = token_to_id["<pad>"]
    unk_id = token_to_id["<unk>"]
    out = torch.full((len(prompts), max_len), pad_id, dtype=torch.long)
    for i, p in enumerate(prompts):
        toks = _tokenize_prompt(p)[:max_len]
        if not toks:
            toks = ["<unk>"]
        ids = [token_to_id.get(t, unk_id) for t in toks]
        out[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return out


def _is_list_shard(shard_obj):
    return isinstance(shard_obj, list)


def discover_shards(root: Path):
    shard_files = sorted(root.glob("grasp_dataset_shard_*.pt"))
    if not shard_files:
        shard_files = sorted(root.glob("shard_*.pt"))
    if not shard_files:
        raise ValueError(f"No shard files found under {root}")

    cumulative = []
    total = 0
    z_dim = None
    object_names = set()
    print("Scanning shard files to build index...", flush=True)
    for shard_path in shard_files:
        shard = torch.load(shard_path, map_location="cpu")
        if _is_list_shard(shard):
            if len(shard) == 0:
                continue
            n = len(shard)
            sample = shard[0]
            if "z_goal" not in sample or "image" not in sample:
                raise ValueError(f"List shard sample must contain 'image' and 'z_goal' in {shard_path}")
            if z_dim is None:
                z_dim = int(sample["z_goal"].numel())
            for dp in shard:
                if "object_name" in dp:
                    object_names.add(str(dp["object_name"]))
        else:
            if "images" not in shard:
                raise ValueError(f"Missing 'images' in {shard_path}")
            labels = label_tensor_from_shard(shard)
            n = shard["images"].shape[0]
            if z_dim is None:
                z_dim = int(labels.shape[-1])
        total += n
        cumulative.append(total)

    print(
        f"Found {total} total samples across {len(shard_files)} shards. "
        f"z_dim: {z_dim}",
        flush=True,
    )
    return shard_files, cumulative, z_dim, sorted(object_names)


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
# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

class MultiPositiveInfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1, atol=1e-5):
        super().__init__()
        self.temperature = temperature
        self.atol = atol

    def forward(self, pred, target):
        # Normalize predictions and targets for cosine similarity
        pred_n = nn.functional.normalize(pred, dim=1)
        target_n = nn.functional.normalize(target, dim=1)

        # Logits: (B, B) matrix where Row i, Col j is similarity between pred[i] and target[j]
        logits = torch.matmul(pred_n, target_n.T) / self.temperature

        # Create positive mask: 1 if target vectors are effectively identical, 0 otherwise.
        # We use torch.isclose with a tight tolerance to handle floating-point nuances in latents.
        mask = torch.isclose(
            target.unsqueeze(1), 
            target.unsqueeze(0), 
            atol=self.atol
        ).all(dim=-1).float()

        # Compute log probabilities: log( exp(sim) / sum(exp(sim)) )
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

        # Mean over all true positives for each anchor, then mean over the batch
        loss = - (mask * log_prob).sum(dim=1) / mask.sum(dim=1)

        return loss.mean()

class HybridDistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, infonce_weight=0.0, temperature=0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.cosine = nn.CosineSimilarity(dim=1)
        self.alpha = alpha
        self.infonce_weight = infonce_weight
        
        if self.infonce_weight > 0:
            self.infonce = MultiPositiveInfoNCELoss(temperature=temperature)

    def forward(self, pred, target):
        # Base losses
        mse_loss = self.mse(pred, target)
        cos_loss = torch.mean(1 - self.cosine(pred, target))
        base_loss = (self.alpha * mse_loss) + ((1 - self.alpha) * cos_loss)

        # Add InfoNCE if configured
        if self.infonce_weight > 0:
            info_loss = self.infonce(pred, target)
            return base_loss + (self.infonce_weight * info_loss)
            
        return base_loss

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
    wandb_log_batches: bool,
    wandb_batch_log_every: int,
    token_to_id: dict,
    max_prompt_len: int,
):
    if not local_indices:
        return 0.0, 0

    shard = torch.load(shard_path, map_location="cpu")
    list_shard = _is_list_shard(shard)
    if not list_shard:
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
        if list_shard:
            dps = [shard[i] for i in chunk]
            x = torch.stack([dp["image"] for dp in dps], dim=0).to(device, non_blocking=True)
            y = torch.stack([dp["z_goal"] for dp in dps], dim=0).to(device, non_blocking=True)
            prompts = [prompt_from_object_name(dp.get("object_name", "object")) for dp in dps]
            text_tokens = encode_prompts(prompts, token_to_id, max_prompt_len).to(device, non_blocking=True)
        else:
            x = images[chunk].to(device, non_blocking=True)
            y = z_goals[chunk].to(device, non_blocking=True)  # (B, H)
            prompts = ["grasp object"] * len(chunk)
            text_tokens = encode_prompts(prompts, token_to_id, max_prompt_len).to(device, non_blocking=True)
        x = normalize(x)

        if train:
            optimizer.zero_grad(set_to_none=True)
            pred = model(x, text_tokens)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        else:
            with torch.no_grad():
                pred = model(x, text_tokens)
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

        if train and wandb_log_batches and (batch_counter[0] % wandb_batch_log_every == 0):
            _wandb_log(
                {
                    "batch/train_loss": loss.item(),
                    "batch/train_lr": optimizer.param_groups[0]["lr"],
                    "batch/epoch": epoch + 1,
                    "batch/step": batch_counter[0],
                },
                step=batch_counter[0],
            )

    return running_loss, n_batches



# ---------------------------------------------------------------------------
# Fixed evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_shardwise(
    model, device, shard_files, cumulative, val_idx, normalize, batch_size, token_to_id, max_prompt_len
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
        list_shard = _is_list_shard(shard)
        if not list_shard:
            images = shard["images"]
            z_goals = label_tensor_from_shard(shard)

        for start in range(0, len(val_locals), batch_size):
            chunk = val_locals[start : start + batch_size]
            if list_shard:
                dps = [shard[i] for i in chunk]
                x = torch.stack([dp["image"] for dp in dps], dim=0).to(device, non_blocking=True)
                y = torch.stack([dp["z_goal"] for dp in dps], dim=0).to(device, non_blocking=True)
                prompts = [prompt_from_object_name(dp.get("object_name", "object")) for dp in dps]
            else:
                x = images[chunk].to(device, non_blocking=True)
                y = z_goals[chunk].to(device, non_blocking=True)
                prompts = ["grasp object"] * len(chunk)
            text_tokens = encode_prompts(prompts, token_to_id, max_prompt_len).to(device, non_blocking=True)
            x = normalize(x)
            pred = model(x, text_tokens)

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

# # ---------------------------------------------------------------------------
# # Model
# # ---------------------------------------------------------------------------
# class StudentHeadWonorm(nn.Module):
#     def __init__(self, in_features, output_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(in_features, 512),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(512, output_dim),
#         )

#     def forward(self, x):
#         return self.net(x)


# class TextPromptEncoder(nn.Module):
#     def __init__(self, vocab_size: int, embed_dim: int = 128):
#         super().__init__()
#         self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

#     def forward(self, token_ids):
#         emb = self.embed(token_ids)  # (B, T, D)
#         mask = (token_ids != 0).unsqueeze(-1).float()
#         denom = mask.sum(dim=1).clamp_min(1.0)
#         return (emb * mask).sum(dim=1) / denom


# class StudentModelWonorm(nn.Module):
#     """Defined at module level so torch.save/load and pickle work correctly."""
#     def __init__(self, output_dim: int, vocab_size: int, text_embed_dim: int = 128):
#         super().__init__()
#         backbone = build_resnet18()
#         in_features = backbone.fc.in_features
#         backbone.fc = nn.Identity()
#         self.backbone = backbone
#         self.text_encoder = TextPromptEncoder(vocab_size=vocab_size, embed_dim=text_embed_dim)
#         self.head = StudentHeadWonorm(in_features + text_embed_dim, output_dim)

#     def forward(self, x, text_tokens):
#         image_feat = self.backbone(x)
#         text_feat = self.text_encoder(text_tokens)
#         fused = torch.cat([image_feat, text_feat], dim=1)
#         return self.head(fused)

# def get_model_wonorm(output_dim: int, vocab_size: int) -> nn.Module:
#     return StudentModelWonorm(output_dim, vocab_size=vocab_size)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class StudentHeadWonorm(nn.Module):
    def __init__(self, in_features, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),   # reduced from 512
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class TextPromptEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 32):  # reduced from 128
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

    def forward(self, token_ids):
        emb = self.embed(token_ids)          # (B, T, D)
        mask = (token_ids != 0).unsqueeze(-1).float()
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (emb * mask).sum(dim=1) / denom


class StudentModelWonorm(nn.Module):
    """
    Lightweight student model for latent goal prediction.

    Architecture:
      - Frozen ResNet18 backbone (pretrained ImageNet) → 512-dim image feature
      - Small learned text embedding (mean-pooled) → 32-dim text feature
      - Adapter: Linear(512 → 256) + ReLU  [only trainable vision params]
      - Head: Linear(256 + 32 → output_dim)

    Trainable params: ~160K (vs ~11.5M in original)
    Frozen params:    ~11.2M (ResNet18 backbone)
    """
    def __init__(
        self,
        output_dim: int,
        vocab_size: int,
        text_embed_dim: int = 32,
        freeze_backbone: bool = True,
        unfreeze_layers: tuple = ("layer4",),   # set to () to freeze all
    ):
        super().__init__()

        # --- Vision backbone ---
        backbone = build_resnet18()
        in_features = backbone.fc.in_features   # 512 for ResNet18
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # Freeze backbone first, then selectively unfreeze requested layers
        for param in self.backbone.parameters():
            param.requires_grad = not freeze_backbone

        if freeze_backbone and unfreeze_layers:
            for name, param in self.backbone.named_parameters():
                if any(name.startswith(layer) for layer in unfreeze_layers):
                    param.requires_grad = True

        # --- Lightweight adapter on top of frozen features ---
        self.adapter = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
        )

        # --- Text encoder ---
        self.text_encoder = TextPromptEncoder(
            vocab_size=vocab_size,
            embed_dim=text_embed_dim,
        )

        # --- Fusion head ---
        self.head = StudentHeadWonorm(
            in_features=256 + text_embed_dim,
            output_dim=output_dim,
        )

    def forward(self, x, text_tokens):
        with torch.set_grad_enabled(
            any(p.requires_grad for p in self.backbone.parameters())
        ):
            image_feat = self.backbone(x)       # (B, 512)

        adapted = self.adapter(image_feat)      # (B, 256)
        text_feat = self.text_encoder(text_tokens)  # (B, 32)
        fused = torch.cat([adapted, text_feat], dim=1)  # (B, 288)
        return self.head(fused)                 # (B, output_dim)

    def count_parameters(self):
        """Utility to print trainable vs frozen param counts."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen    = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        total     = trainable + frozen
        print(
            f"Parameters | trainable: {trainable:,}  "
            f"frozen: {frozen:,}  total: {total:,}",
            flush=True,
        )
        return trainable, frozen, total


def get_model_wonorm(output_dim: int, vocab_size: int) -> nn.Module:
    model = StudentModelWonorm(
        output_dim=output_dim,
        vocab_size=vocab_size,
        text_embed_dim=32,
        freeze_backbone=True,
        unfreeze_layers=(),       # freeze ALL of ResNet18
    )
    model.count_parameters()
    return model


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
def collect_val_embeddings(
    model, device, shard_files, cumulative, val_idx, normalize, batch_size, token_to_id, max_prompt_len, max_points=2000
):
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
        list_shard = _is_list_shard(shard)
        if not list_shard:
            images = shard["images"]
            z_goals = label_tensor_from_shard(shard)

        for start in range(0, len(local_indices), batch_size):
            chunk = local_indices[start: start + batch_size]
            if list_shard:
                dps = [shard[i] for i in chunk]
                x = torch.stack([dp["image"] for dp in dps], dim=0).to(device, non_blocking=True)
                y = torch.stack([dp["z_goal"] for dp in dps], dim=0)
                prompts = [prompt_from_object_name(dp.get("object_name", "object")) for dp in dps]
            else:
                x = images[chunk].to(device, non_blocking=True)
                y = z_goals[chunk]
                prompts = ["grasp object"] * len(chunk)
            text_tokens = encode_prompts(prompts, token_to_id, max_prompt_len).to(device, non_blocking=True)
            x = normalize(x)
            pred = model(x, text_tokens).cpu()
            preds.append(pred)
            targets.append(y)

    z_pred   = torch.cat(preds,   dim=0).float().numpy()
    z_target = torch.cat(targets, dim=0).float().numpy()
    return z_pred, z_target

def generate_pca_plot(z_pred: np.ndarray, z_target: np.ndarray, epoch: int, out_dir: Path):
    """Save two PCA views: target-axes (stable) and joint-axes (full picture).
    Returns (path_target_axes, path_joint_axes) for optional wandb logging.
    """
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

    return out_path_target, out_path_joint


# ---------------------------------------------------------------------------
# Integrated-pipeline evaluation (Isaac Gym + NTField + trained latent model)
# ---------------------------------------------------------------------------

def _save_live_checkpoint_for_integration(
    model: nn.Module, z_dim: int, ckpt_path: Path, vocab_size: Optional[int] = None
) -> None:
    """
    Save current weights in the exact format the integration script expects:
        {"model_state_dict": ..., "z_dim": ...}
    The wonorm head matches `_get_latent_model` in the integration script
    exactly, so we pass the state_dict through unchanged.
    """
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "z_dim": z_dim,
        "vocab_size": vocab_size,
    }, ckpt_path)


def _parse_integration_trial_result(session_dir: Path) -> dict:
    """
    Read pipeline_summary.json from an integration run and extract success stats.

    For run_integrated_pipeline_latent_multi_obj.py, primary success comes from:
      - summary["status"] in {"Success", "Failure"}
      - summary["predicted_video"]
      - summary["true_latent_error"] (optional metrics bundle)

    Backward compatibility:
      - legacy video key: summary["videos"]["predicted_latent_goal"]
      - legacy success key: summary["success_check"]["success"]
      - legacy latent error key: summary["latent_goal_comparison"]["l2"]
    """
    root_summary = session_dir / "pipeline_summary.json"
    all_candidates = sorted(
        session_dir.glob("**/pipeline_summary.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    nested_candidates = [p for p in all_candidates if p != root_summary]
    if nested_candidates:
        # Prefer timestamped child-session summaries generated by the latest
        # integration script over stale root-level files from older runs.
        summary_path = nested_candidates[0]
    elif root_summary.is_file():
        summary_path = root_summary
    else:
        return {"success": False, "reason": "no_summary", "summary": None, "video": None}

    try:
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
    except Exception as e:
        return {"success": False, "reason": f"summary_parse_error: {e}", "summary": None, "video": None}

    video = summary.get("predicted_video")
    if not video:
        video = (summary.get("videos") or {}).get("predicted_latent_goal")
    video_exists = bool(video) and Path(video).is_file()

    status_str = str(summary.get("status", "")).strip().lower()
    if status_str in {"success", "failure"}:
        success = status_str == "success"
    else:
        # Backward compatibility fallback for legacy summaries.
        success_check = summary.get("success_check") or {}
        success_from_summary = success_check.get("success")
        success = bool(success_from_summary) if success_from_summary is not None else video_exists

    true_latent_error = summary.get("true_latent_error")
    if not isinstance(true_latent_error, dict):
        true_latent_error = summary.get("latent_goal_comparison")
    latent_error_l2 = None
    if isinstance(true_latent_error, dict):
        try:
            latent_error_l2_val = true_latent_error.get("l2")
            latent_error_l2 = (
                float(latent_error_l2_val)
                if latent_error_l2_val is not None
                else None
            )
        except (TypeError, ValueError):
            latent_error_l2 = None
    if latent_error_l2 is None:
        # Fallback to an always-available planner metric so logs don't show
        # None when grasp-derived true latent isn't available for a trial.
        try:
            final_dist = summary.get("ntfield_final_latent_dist")
            latent_error_l2 = float(final_dist) if final_dist is not None else None
        except (TypeError, ValueError):
            latent_error_l2 = None

    return {
        "success": success,
        "summary": summary,
        "video": video if video_exists else None,
        "latent_error_l2": latent_error_l2,
    }


def run_integration_eval(
    model: nn.Module,
    z_dim: int,
    epoch: int,
    live_ckpt_path: Path,
    vocab_size: Optional[int],
    num_trials: int = INTEGRATION_NUM_TRIALS,
) -> dict:
    """
    Run the end-to-end Isaac Gym + NTField pipeline `num_trials` times using
    the *current* model weights (saved to a temporary checkpoint) and report
    success rate + timing stats.

    Uses subprocess per trial because Isaac Gym has strict import-order
    requirements and holds global state that conflicts with an already-running
    torch/CUDA training process.
    """
    was_training = model.training
    model.eval()

    _save_live_checkpoint_for_integration(model, z_dim, live_ckpt_path, vocab_size=vocab_size)

    eval_dir = INTEGRATION_OUTPUT_ROOT / f"epoch_{epoch:03d}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    trial_records = []
    per_trial_times = []
    per_trial_latent_l2 = []
    n_success = 0

    # wandb artifacts collected across trials
    wandb_table_rows = []
    wandb_video_payload: dict = {}
    wandb_topview_payload: dict = {}

    print(
        f" -> Running integrated evaluation: {num_trials} trials "
        f"(epoch {epoch}), outputs at {eval_dir}",
        flush=True,
    )

    for trial in range(num_trials):
        trial_dir = eval_dir / f"trial_{trial:02d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(INTEGRATION_SCRIPT),
            "--ntfield_checkpoint", str(NTFIELD_CHECKPOINT),
            "--latent_checkpoint", str(live_ckpt_path),
            "--output_dir", str(trial_dir),
            "--seed", str(1000 + trial),                # reproducible per-trial scenes
            "--ntfield_step_size", str(INTEGRATION_NTFIELD_STEP_SIZE),
            "--ntfield_max_steps", str(INTEGRATION_NTFIELD_MAX_STEPS),
            "--ntfield_tol", str(INTEGRATION_NTFIELD_TOL),
        ]

        log_path = trial_dir / "run.log"
        t0 = time.time()
        try:
            with log_path.open("w") as logf:
                proc = subprocess.run(
                    cmd,
                    cwd=str(PI_VLA_ROOT),
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    timeout=INTEGRATION_PER_TRIAL_TIMEOUT,
                    check=False,
                )
            elapsed = time.time() - t0
            exit_code = proc.returncode
            timed_out = False
        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            exit_code = -1
            timed_out = True

        parsed = _parse_integration_trial_result(trial_dir)
        # Some Isaac Gym runs can report shutdown segfaults after writing
        # outputs; rely on the explicit success_check payload for task success.
        success = (not timed_out) and bool(parsed.get("success", False))
        latent_error_l2 = parsed.get("latent_error_l2")
        if success:
            n_success += 1
        per_trial_times.append(elapsed)
        if latent_error_l2 is not None:
            per_trial_latent_l2.append(float(latent_error_l2))

        trial_records.append({
            "trial": trial,
            "elapsed_sec": elapsed,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "success": success,
            "latent_error_l2": latent_error_l2,
            "trial_dir": str(trial_dir),
            "video": parsed.get("video"),
        })
        print(
            f"    trial {trial+1:02d}/{num_trials}: "
            f"success={success} | latent_error_l2={latent_error_l2} | "
            f"time={elapsed:6.2f}s | exit={exit_code}"
            + (" | TIMEOUT" if timed_out else ""),
            flush=True,
        )

        # -- wandb per-trial assets --
        if _wandb_active():
            video_path = parsed.get("video")
            top_view_path = trial_dir / "top_view.png"

            # Upload up to INTEGRATION_VIDEOS_PER_EVAL videos per eval.
            if (
                video_path is not None
                and Path(video_path).is_file()
                and trial < INTEGRATION_VIDEOS_PER_EVAL
            ):
                try:
                    wandb_video_payload[f"integration/video/trial_{trial:02d}"] = wandb.Video(
                        str(video_path),
                        caption=f"epoch {epoch} | trial {trial} | success={success}",
                        format="mp4",
                    )
                except Exception as e:
                    print(f"    [wandb] video upload failed for trial {trial}: {e}", flush=True)

            if top_view_path.is_file() and trial < INTEGRATION_VIDEOS_PER_EVAL:
                try:
                    wandb_topview_payload[f"integration/top_view/trial_{trial:02d}"] = wandb.Image(
                        str(top_view_path),
                        caption=f"epoch {epoch} | trial {trial} | success={success}",
                    )
                except Exception as e:
                    print(f"    [wandb] top_view upload failed for trial {trial}: {e}", flush=True)

            # Row for the per-trial table.
            wandb_table_rows.append([
                epoch,
                trial,
                bool(success),
                float(elapsed),
                int(exit_code),
                bool(timed_out),
                (float(latent_error_l2) if latent_error_l2 is not None else None),
                str(trial_dir),
            ])

    times = np.asarray(per_trial_times, dtype=np.float64) if per_trial_times else np.zeros(0)
    latent_l2_arr = (
        np.asarray(per_trial_latent_l2, dtype=np.float64)
        if per_trial_latent_l2
        else np.zeros(0)
    )
    agg = {
        "epoch": epoch,
        "num_trials": num_trials,
        "num_success": n_success,
        "success_rate": (n_success / num_trials) if num_trials > 0 else 0.0,
        "mean_time_sec": float(times.mean()) if times.size else 0.0,
        "median_time_sec": float(np.median(times)) if times.size else 0.0,
        "min_time_sec": float(times.min()) if times.size else 0.0,
        "max_time_sec": float(times.max()) if times.size else 0.0,
        "std_time_sec": float(times.std()) if times.size else 0.0,
        "num_trials_with_latent_error_l2": int(latent_l2_arr.size),
        "mean_true_latent_error_l2": float(latent_l2_arr.mean()) if latent_l2_arr.size else None,
        "median_true_latent_error_l2": float(np.median(latent_l2_arr)) if latent_l2_arr.size else None,
        "min_true_latent_error_l2": float(latent_l2_arr.min()) if latent_l2_arr.size else None,
        "max_true_latent_error_l2": float(latent_l2_arr.max()) if latent_l2_arr.size else None,
        "trials": trial_records,
    }

    with (eval_dir / "aggregate.json").open("w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)

    # Append one row to a running CSV so it's trivial to plot performance vs. epoch.
    INTEGRATION_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = INTEGRATION_OUTPUT_ROOT / "integration_eval_history.csv"
    new_file = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8") as f:
        if new_file:
            f.write("epoch,num_trials,num_success,success_rate,mean_time_sec,"
                    "median_time_sec,min_time_sec,max_time_sec,"
                    "num_trials_with_latent_error_l2,mean_true_latent_error_l2\n")
        mean_latent_l2_csv = (
            ""
            if agg["mean_true_latent_error_l2"] is None
            else f"{agg['mean_true_latent_error_l2']:.6f}"
        )
        f.write(
            f"{agg['epoch']},{agg['num_trials']},{agg['num_success']},"
            f"{agg['success_rate']:.6f},{agg['mean_time_sec']:.4f},"
            f"{agg['median_time_sec']:.4f},{agg['min_time_sec']:.4f},"
            f"{agg['max_time_sec']:.4f},"
            f"{agg['num_trials_with_latent_error_l2']},"
            f"{mean_latent_l2_csv}\n"
        )

    print(
        f" -> Integrated eval @ epoch {epoch}: "
        f"success {n_success}/{num_trials} ({100.0*agg['success_rate']:.1f}%) | "
        f"mean time {agg['mean_time_sec']:.2f}s | "
        f"median {agg['median_time_sec']:.2f}s",
        flush=True,
    )

    # ---- wandb: log aggregate metrics + videos + table in ONE call ----
    if _wandb_active():
        payload: dict = {
            "epoch": epoch,
            "integration/success_rate":  agg["success_rate"],
            "integration/num_success":   agg["num_success"],
            "integration/num_trials":    agg["num_trials"],
            "integration/mean_time_sec": agg["mean_time_sec"],
            "integration/median_time_sec": agg["median_time_sec"],
            "integration/min_time_sec":    agg["min_time_sec"],
            "integration/max_time_sec":    agg["max_time_sec"],
            "integration/std_time_sec":    agg["std_time_sec"],
            "integration/num_trials_with_latent_error_l2": agg["num_trials_with_latent_error_l2"],
        }
        if agg["mean_true_latent_error_l2"] is not None:
            payload["integration/mean_true_latent_error_l2"] = agg["mean_true_latent_error_l2"]
            payload["integration/median_true_latent_error_l2"] = agg["median_true_latent_error_l2"]
            payload["integration/min_true_latent_error_l2"] = agg["min_true_latent_error_l2"]
            payload["integration/max_true_latent_error_l2"] = agg["max_true_latent_error_l2"]
        payload.update(wandb_video_payload)
        payload.update(wandb_topview_payload)

        # Per-trial table (nice sortable/filterable view in wandb UI).
        try:
            trials_table = wandb.Table(
                columns=["epoch", "trial", "success", "elapsed_sec",
                         "exit_code", "timed_out", "latent_error_l2", "trial_dir"],
                data=wandb_table_rows,
            )
            payload[f"integration/trials_table_epoch_{epoch:03d}"] = trials_table
        except Exception as e:
            print(f"[wandb] trial table log failed: {e}", flush=True)

        _wandb_log(payload)

    if was_training:
        model.train()
    return agg


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

    shard_files, cumulative, z_dim, object_names = discover_shards(DATASET_ROOT)
    token_to_id = build_text_vocab(object_names)
    max_prompt_len = int(os.getenv("TRAIN_MAX_PROMPT_LEN", "8"))
    n_total = cumulative[-1]
    train_idx, val_idx, train_size, val_size = build_train_val_split(n_total)
    print(f"Train: {train_size} | Val: {val_size}", flush=True)
    print(f"Text vocab size: {len(token_to_id)} | max_prompt_len={max_prompt_len}", flush=True)

    model = get_model_wonorm(output_dim=z_dim, vocab_size=len(token_to_id)).to(device)

    epochs = int(os.getenv("TRAIN_EPOCHS", "30"))
    eval_every = int(os.getenv("TRAIN_EVAL_EVERY", "2"))
    batch_size = int(os.getenv("TRAIN_BATCH_SIZE", "128"))
    lr = float(os.getenv("TRAIN_LR", "1e-4"))
    loss_type = os.getenv("TRAIN_LOSS_TYPE", "hybrid").strip().lower()
    loss_alpha = float(os.getenv("TRAIN_LOSS_ALPHA", "0.5"))
    infonce_weight = float(os.getenv("TRAIN_INFONCE_WEIGHT", "0.1")) # Set to 0.0 to disable
    infonce_temp = float(os.getenv("TRAIN_INFONCE_TEMP", "0.1"))

    best_val_mae = float("inf")
    best_val_cos = float("inf")
    best_val_n   = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if loss_type == "mse":
        criterion = nn.MSELoss()
    elif loss_type == "cos":
        criterion = lambda pred, target: torch.mean(
            1 - nn.functional.cosine_similarity(pred, target, dim=1)
        )
    elif loss_type == "hybrid":
        criterion = HybridDistillationLoss(
            alpha=loss_alpha, 
            infonce_weight=infonce_weight, 
            temperature=infonce_temp
        )
    else:
        raise ValueError(
            f"Unsupported TRAIN_LOSS_TYPE='{loss_type}'. Use one of: hybrid, mse, cos."
        )
        
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    pca_output_dir = Path(os.getenv("TRAIN_PCA_OUTPUT_DIR", str(PCA_OUTPUT_DIR)))
    wandb_enabled = _env_bool("WANDB_ENABLED", WANDB_ENABLED)
    wandb_run_name = _env_opt_str("WANDB_RUN_NAME", WANDB_RUN_NAME)
    wandb_log_batches = _env_bool("WANDB_LOG_BATCHES", WANDB_LOG_BATCHES)
    wandb_batch_log_every = int(os.getenv("WANDB_BATCH_LOG_EVERY", str(WANDB_BATCH_LOG_EVERY)))
    train_ckpt_name = _env_opt_str("TRAIN_CKPT_NAME", None)

    print(
        f"Config | epochs={epochs} eval_every={eval_every} batch_size={batch_size} "
        f"lr={lr:.6g} loss_type={loss_type} loss_alpha={loss_alpha}",
        flush=True,
    )
    print(
        f"WANDB | enabled={wandb_enabled} run_name={wandb_run_name} "
        f"log_batches={wandb_log_batches} batch_log_every={wandb_batch_log_every}",
        flush=True,
    )
    print(
        f"Checkpoint | TRAIN_CKPT_NAME={train_ckpt_name if train_ckpt_name else '<auto>'}",
        flush=True,
    )
    print(f"PCA | output_dir={pca_output_dir}", flush=True)
    run_slug_raw = wandb_run_name if wandb_run_name else "run"
    run_slug = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in run_slug_raw)
    integration_live_ckpt = INTEGRATION_LIVE_CKPT.with_name(f"_training_live_latent_{run_slug}.pth")
    print(f"Integration | live_ckpt={integration_live_ckpt}", flush=True)

    INTEGRATION_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # wandb init
    # -----------------------------------------------------------------------
    if wandb_enabled and not _WANDB_AVAILABLE:
        print("[wandb] wandb is not installed — skipping wandb logging.", flush=True)
    if wandb_enabled and _WANDB_AVAILABLE:
        wandb_config = {
            "dataset_root": str(DATASET_ROOT),
            "n_total": n_total,
            "train_size": train_size,
            "val_size": val_size,
            "z_dim": z_dim,
            "epochs": epochs,
            "batch_size": batch_size,
            "eval_every": eval_every,
            "optimizer": "Adam",
            "lr_initial": lr,
            "scheduler": "CosineAnnealingLR",
            "loss": f"{loss_type}(alpha={loss_alpha})" if loss_type == "hybrid" else loss_type,
            "model": "StudentModelWonorm(ResNet18 + text prompt embedding fusion -> MLP 512 -> z_dim)",
            "prompt_template": "grasp {object_name}",
            "text_vocab_size": len(token_to_id),
            "max_prompt_len": max_prompt_len,
            "run_integration_every": RUN_INTEGRATION_EVERY,
            "integration_num_trials": INTEGRATION_NUM_TRIALS,
            "integration_videos_per_eval": INTEGRATION_VIDEOS_PER_EVAL,
            "integration_ntfield_tol": INTEGRATION_NTFIELD_TOL,
            "integration_ntfield_max_steps": INTEGRATION_NTFIELD_MAX_STEPS,
            "integration_ntfield_step_size": INTEGRATION_NTFIELD_STEP_SIZE,
            "ntfield_checkpoint": NTFIELD_CHECKPOINT,
            "device": str(device),
        }
        try:
            wandb.init(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                name=wandb_run_name,
                config=wandb_config,
                dir=str(PI_VLA_ROOT / "output"),
            )
            print(f"[wandb] run initialized: {wandb.run.name}", flush=True)
        except Exception as e:
            print(f"[wandb] init failed, continuing without it: {e}", flush=True)

    try:
        # Keep wandb batch-step monotonic across all epochs.
        global_batch_counter = [0]
        for epoch in range(epochs):
            shard_order = torch.randperm(len(shard_files)).tolist()
            running_loss    = 0.0
            n_batches_total = 0

            for si in shard_order:
                train_locals, _ = train_val_indices_for_shard(si, cumulative, train_idx, val_idx)
                loss_sum, nb = run_batches(
                    model, optimizer, criterion, device,
                    shard_files[si], train_locals, normalize,
                    batch_size, True, epoch, epochs, global_batch_counter,
                    wandb_log_batches, wandb_batch_log_every,
                    token_to_id, max_prompt_len,
                )
                running_loss    += loss_sum
                n_batches_total += nb

            scheduler.step()
            avg_train_loss = running_loss / max(n_batches_total, 1)
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} "
                f"| LR: {current_lr:.8f}",
                flush=True,
            )

            # Per-epoch training scalars.
            _wandb_log({
                "epoch": epoch + 1,
                "train/loss": avg_train_loss,
                "train/lr": current_lr,
                "train/num_batches": n_batches_total,
            })

            if (epoch + 1) % eval_every == 0 or (epoch + 1) == epochs:
                val_mse, val_mae, val_cos, val_n = evaluate_shardwise(
                    model, device, shard_files, cumulative, val_idx, normalize, batch_size,
                    token_to_id, max_prompt_len,
                )
                print(
                    f" -> Val MSE: {val_mse:.6f} | Val MAE: {val_mae:.6f} "
                    f"| Val Cos: {val_cos:.6f} | N: {val_n}",
                    flush=True,
                )

                # --- PCA visualization ---
                z_pred, z_target = collect_val_embeddings(
                    model, device, shard_files, cumulative, val_idx, normalize,
                    batch_size=batch_size, token_to_id=token_to_id,
                    max_prompt_len=max_prompt_len, max_points=2000,
                )
                pca_target_path, pca_joint_path = generate_pca_plot(
                    z_pred, z_target, epoch + 1, pca_output_dir
                )
                # -------------------------

                # Val scalars + PCA images in a single wandb.log call.
                val_payload = {
                    "epoch": epoch + 1,
                    "val/mse": val_mse,
                    "val/mae": val_mae,
                    "val/cos_distance": val_cos,   # 1 - cos_sim (0 == perfect)
                    "val/n_samples": val_n,
                }
                if _wandb_active():
                    try:
                        val_payload["val/pca_target_axes"] = wandb.Image(
                            str(pca_target_path),
                            caption=f"epoch {epoch+1} — PCA (target axes)",
                        )
                        val_payload["val/pca_joint_axes"] = wandb.Image(
                            str(pca_joint_path),
                            caption=f"epoch {epoch+1} — PCA (joint axes)",
                        )
                    except Exception as e:
                        print(f"[wandb] PCA image log failed: {e}", flush=True)
                _wandb_log(val_payload)

                if val_mae < best_val_mae:
                    best_val_mae = val_mae
                    best_val_cos = val_cos
                    best_val_n   = val_n
                    ckpt_suffix = wandb_run_name if wandb_run_name else "run"
                    default_ckpt_name = f"best_z_goal_model_{ckpt_suffix}.pth"
                    if train_ckpt_name:
                        ckpt_name = train_ckpt_name.replace("{run_name}", ckpt_suffix)
                    else:
                        ckpt_name = default_ckpt_name
                    if not ckpt_name.endswith(".pth"):
                        ckpt_name = f"{ckpt_name}.pth"
                    ckpt_path = ckpt_name
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "z_dim":   z_dim,
                        "vocab_size": len(token_to_id),
                        "token_to_id": token_to_id,
                        "max_prompt_len": max_prompt_len,
                        "epoch":   epoch + 1,
                        "val_mae": best_val_mae,
                        "val_cos": best_val_cos,
                    }, ckpt_path)
                    print(
                        f" -> New best saved. MAE: {best_val_mae:.6f} "
                        f"| Cos: {best_val_cos:.6f}",
                        flush=True,
                    )
                    _wandb_log({
                        "epoch": epoch + 1,
                        "val/best_mae": best_val_mae,
                        "val/best_cos_distance": best_val_cos,
                        "val/best_epoch": epoch + 1,
                    })
                    # Also upload the checkpoint as a wandb artifact so you
                    # can pull down whichever best ckpt you want later.
                    if _wandb_active():
                        try:
                            art = wandb.Artifact(
                                name=f"best_model_wonorm",
                                type="model",
                                metadata={
                                    "epoch": epoch + 1,
                                    "val_mae": best_val_mae,
                                    "val_cos": best_val_cos,
                                    "z_dim": z_dim,
                                },
                            )
                            art.add_file(ckpt_path)
                            wandb.log_artifact(art, aliases=["best", f"epoch_{epoch+1}"])
                        except Exception as e:
                            print(f"[wandb] artifact upload failed: {e}", flush=True)

            # ------------------------------------------------------------------
            # Periodic integrated-pipeline evaluation (Isaac Gym + NTField)
            # ------------------------------------------------------------------
            if (epoch + 1) % RUN_INTEGRATION_EVERY == 0 or (epoch + 1) == epochs:
                try:
                    run_integration_eval(
                        model,
                        z_dim=z_dim,
                        epoch=epoch + 1,
                        live_ckpt_path=integration_live_ckpt,
                        vocab_size=len(token_to_id),
                        num_trials=INTEGRATION_NUM_TRIALS,
                    )
                except Exception as e:
                    # Never let evaluation kill training.
                    print(f" -> Integrated evaluation failed with error: {e}", flush=True)

    finally:
        if _wandb_active():
            try:
                wandb.finish()
            except Exception:
                pass


if __name__ == "__main__":
    train()