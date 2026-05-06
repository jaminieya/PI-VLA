import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models, transforms
from student_model_cvae import CVAEStudent

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
# When INTEGRATION_ON_VAL_IMPROVE_ONLY=0, run integration every N epochs (and last epoch).
RUN_INTEGRATION_EVERY = 20
INTEGRATION_NUM_TRIALS = 10        # how many trials per eval
INTEGRATION_PER_TRIAL_TIMEOUT = 300  # seconds per trial budget (scaled for batch subprocess)
# Total timeout for one integration subprocess = num_trials * per-trial * mult (override with INTEGRATION_EVAL_TIMEOUT).
INTEGRATION_BATCH_TIMEOUT_MULT = float(os.getenv("INTEGRATION_BATCH_TIMEOUT_MULT", "1.25"))
INTEGRATION_NTFIELD_TOL = 0.01
INTEGRATION_NTFIELD_MAX_STEPS = 200
INTEGRATION_NTFIELD_STEP_SIZE = 0.02
# First N trials get wandb.Video uploads per integration run (env INTEGRATION_VIDEOS_PER_EVAL).
INTEGRATION_VIDEOS_PER_EVAL = 8
# One subprocess per eval with --num_trials (see run_integrated_pipeline_latent_multi_obj.py).
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
    if not data:
        return
    try:
        if step is not None:
            wandb.log(data, step=step)
        else:
            wandb.log(data)
    except Exception as e:
        print(f"[wandb] log failed: {e}", flush=True)
        traceback.print_exc()


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
# KL-beta scheduling
# ---------------------------------------------------------------------------

def compute_kl_beta(
    step: int,
    schedule: str,
    beta_start: float,
    beta_end: float,
    anneal_steps: int,
    cycle_steps: int,
) -> float:
    """
    Compute KL weight beta for the current optimization step.
    Supported schedules:
      - linear: ramp beta_start -> beta_end over anneal_steps, then hold.
      - cyclical: ramp beta_start -> beta_end each cycle over cycle_steps.
    """
    if schedule == "linear":
        if anneal_steps <= 0:
            return beta_end
        progress = min(max(step, 0) / anneal_steps, 1.0)
        return beta_start + (beta_end - beta_start) * progress

    if schedule == "cyclical":
        if cycle_steps <= 0:
            return beta_end
        cycle_step = step % cycle_steps
        progress = cycle_step / cycle_steps
        return beta_start + (beta_end - beta_start) * progress

    raise ValueError(f"Unsupported KL beta schedule: {schedule}")


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_batches(
    model, optimizer, criterion, device,
    shard_path, local_indices, normalize,
    batch_size, train, epoch, epochs,
    batch_counter, wandb_log_batches, wandb_batch_log_every,
    token_to_id, max_prompt_len,
    kl_beta_schedule, kl_beta_start, kl_beta_end, kl_anneal_steps, kl_cycle_steps,
    z_align_weight: float = 0.1,
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
            y = z_goals[chunk].to(device, non_blocking=True)
            prompts = ["grasp object"] * len(chunk)
            text_tokens = encode_prompts(prompts, token_to_id, max_prompt_len).to(device, non_blocking=True)
        x = normalize(x)

        if train:
            optimizer.zero_grad(set_to_none=True)
            z_pred, kl_loss, aux = model(x, text_tokens, z_goal=y, return_aux=True)
            recon_loss = criterion(z_pred, y)
            prior_mu = aux.get("prior_mu")
            posterior_mu = aux.get("posterior_mu")
            if prior_mu is None or posterior_mu is None:
                z_alignment_loss = torch.zeros((), device=x.device, dtype=x.dtype)
            else:
                z_alignment_loss = F.mse_loss(prior_mu, posterior_mu.detach())
            beta = compute_kl_beta(
                step=batch_counter[0],
                schedule=kl_beta_schedule,
                beta_start=kl_beta_start,
                beta_end=kl_beta_end,
                anneal_steps=kl_anneal_steps,
                cycle_steps=kl_cycle_steps,
            )
            loss = recon_loss + beta * kl_loss + float(z_align_weight) * z_alignment_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        else:
            with torch.no_grad():
                z_pred, _ = model(x, text_tokens, z_goal=None)
                loss = criterion(z_pred, y)
            running_loss += loss.item()

        n_batches += 1
        batch_counter[0] += 1

        if train and batch_counter[0] % LOG_EVERY_BATCHES == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} | batch {batch_counter[0]} | "
                f"last loss {loss.item():.6f}",
                flush=True,
            )

        
        # Update wandb logging to track both losses:
        if train and wandb_log_batches and (batch_counter[0] % wandb_batch_log_every == 0):
            _wandb_log({
                "batch/train_loss": loss.item(),
                "batch/recon_loss": recon_loss.item(),
                "batch/kl_loss": kl_loss.item(),
                "batch/z_alignment_loss": z_alignment_loss.item(),
                "batch/z_align_weight": float(z_align_weight),
                "batch/kl_beta": beta,
                "batch/train_lr": optimizer.param_groups[0]["lr"],
                "batch/epoch": epoch + 1,
                "batch/step": batch_counter[0],
            }, step=batch_counter[0])

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
            pred, _ = model(x, text_tokens, z_goal=None)

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


def get_model_wonorm(output_dim: int, vocab_size: int, dropout_p: float = 0.2) -> nn.Module:
    model = CVAEStudent(
        output_dim=output_dim,
        vocab_size=vocab_size,
        latent_dim=128,
        dropout_p=dropout_p,
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
            pred, _ = model(x, text_tokens, z_goal=None)
            pred = pred.cpu()
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
        "latent_dim": int(getattr(model, "latent_dim", 128)),
    }, ckpt_path)


def _resolve_integration_media_path(
    maybe_path: Optional[str],
    base_dir: Optional[Union[str, Path]],
) -> Optional[str]:
    """
    Turn a path from ``pipeline_summary.json`` into an absolute path the training
    process can open. Handles paths saved relative to ``trial_dir`` / ``session_dir``.
    """
    if not maybe_path or not isinstance(maybe_path, str):
        return None
    p = Path(maybe_path).expanduser()
    if p.is_file():
        return str(p.resolve())
    if base_dir is None:
        return None
    b = Path(base_dir)
    for cand in (b / maybe_path, b / p.name):
        try:
            if cand.is_file():
                return str(cand.resolve())
        except OSError:
            continue
    return None


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

    session_for_path = str(summary.get("session_dir") or session_dir)
    video_raw = summary.get("predicted_video")
    if not video_raw:
        video_raw = (summary.get("videos") or {}).get("predicted_latent_goal")
    video = _resolve_integration_media_path(
        video_raw if isinstance(video_raw, str) else None,
        session_for_path,
    )
    video_exists = video is not None

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


def _find_latest_integration_session(eval_dir: Path) -> Optional[Path]:
    """Newest directory under ``eval_dir`` that contains integration outputs."""
    if not eval_dir.is_dir():
        return None
    hits: List[Path] = []
    for p in eval_dir.rglob("batch_summary.json"):
        hits.append(p.parent)
    for p in eval_dir.rglob("pipeline_summary.json"):
        hits.append(p.parent)
    if not hits:
        return None
    hits.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return hits[0]


def _merge_batch_trials_into_metrics(session_dir: Path) -> Dict[str, Any]:
    """
    Parse ``batch_summary.json`` from ``run_integrated_pipeline_latent_multi_obj.py``
    (``--num_trials`` batch). Falls back to a single ``pipeline_summary.json`` under
    ``session_dir`` for older layouts.
    """
    batch_path = session_dir / "batch_summary.json"
    if batch_path.is_file():
        with batch_path.open("r", encoding="utf-8") as f:
            batch = json.load(f)
        trials_out: List[Dict[str, Any]] = []
        for t in batch.get("trials") or []:
            st = str(t.get("status", "")).strip().lower()
            ok = st == "success"
            if st == "error":
                ok = False
            td = t.get("session_dir") or t.get("trial_dir")
            td_s = str(td) if td else str(session_dir)
            vid_raw = t.get("predicted_video")
            vid = _resolve_integration_media_path(
                vid_raw if isinstance(vid_raw, str) else None,
                td_s,
            )
            dist = t.get("ntfield_final_latent_dist")
            dist_f: Optional[float] = None
            if isinstance(dist, (int, float)) and dist == dist:
                dist_f = float(dist)
            lg = t.get("latent_goal_l2")
            lg_f: Optional[float] = None
            if isinstance(lg, (int, float)) and lg == lg:
                lg_f = float(lg)
            ee = t.get("ee_to_target_dist_m")
            sc_raw = t.get("success_check")
            sc = sc_raw if isinstance(sc_raw, dict) else {}
            if ee is None:
                ee = sc.get("ee_to_target_dist_m")
            ee_f: Optional[float] = None
            if isinstance(ee, (int, float)) and ee == ee:
                ee_f = float(ee)
            ee_succ = sc.get("success")
            trials_out.append(
                {
                    "trial": int(t.get("trial_index", len(trials_out))),
                    "success": ok,
                    "latent_error_l2": dist_f,
                    "planner_final_latent_dist": dist_f,
                    "latent_goal_l2": lg_f,
                    "path_length": t.get("path_length"),
                    "planner_stop_reason": t.get("planner_stop_reason"),
                    "trial_dir": td_s,
                    "video": vid,
                    "exit_code": 0 if st != "error" else 1,
                    "timed_out": False,
                    "error": t.get("error"),
                    "ee_to_target_dist_m": ee_f,
                    "ee_thresh_success": ee_succ,
                }
            )
        return {
            "batch": batch,
            "trials": trials_out,
            "success_rate": float(batch.get("success_rate", 0.0)),
            "num_success": int(batch.get("success_count", 0)),
        }

    pr = _parse_integration_trial_result(session_dir)
    success = bool(pr.get("success", False))
    dist = pr.get("latent_error_l2")
    summ = pr.get("summary") or {}
    td = summ.get("session_dir") or str(session_dir)
    vid = pr.get("video")
    ee0 = summ.get("ee_to_target_dist_m")
    sc0 = summ.get("success_check") if isinstance(summ.get("success_check"), dict) else {}
    if ee0 is None:
        ee0 = sc0.get("ee_to_target_dist_m")
    ee0_f: Optional[float] = None
    if isinstance(ee0, (int, float)) and ee0 == ee0:
        ee0_f = float(ee0)
    return {
        "batch": None,
        "trials": [
            {
                "trial": 0,
                "success": success,
                "latent_error_l2": dist,
                "planner_final_latent_dist": dist,
                "latent_goal_l2": summ.get("latent_goal_l2"),
                "path_length": summ.get("path_length"),
                "planner_stop_reason": summ.get("planner_stop_reason"),
                "trial_dir": str(td),
                "video": str(vid) if vid else None,
                "exit_code": 0,
                "timed_out": False,
                "error": None,
                "ee_to_target_dist_m": ee0_f,
                "ee_thresh_success": sc0.get("success"),
            }
        ],
        "success_rate": 1.0 if success else 0.0,
        "num_success": 1 if success else 0,
    }


def _run_integration_subprocess(
    eval_dir: Path,
    live_ckpt_path: Path,
    epoch: int,
    num_trials: int,
) -> Dict[str, Any]:
    """
    One subprocess: ``run_integrated_pipeline_latent_multi_obj.py --num_trials N``
    writes ``<eval_dir>/<timestamp>/batch_summary.json`` and per-trial folders.
    """
    eval_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(INTEGRATION_SCRIPT),
        "--ntfield_checkpoint",
        str(NTFIELD_CHECKPOINT),
        "--latent_checkpoint",
        str(live_ckpt_path),
        "--output_dir",
        str(eval_dir.resolve()),
        "--num_trials",
        str(num_trials),
        "--seed",
        str(10_000 + int(epoch)),
        "--ntfield_step_size",
        str(INTEGRATION_NTFIELD_STEP_SIZE),
        "--ntfield_max_steps",
        str(INTEGRATION_NTFIELD_MAX_STEPS),
        "--ntfield_tol",
        str(INTEGRATION_NTFIELD_TOL),
    ]
    log_path = eval_dir / f"integration_subprocess_epoch_{epoch:03d}.log"
    t_budget = int(
        os.getenv(
            "INTEGRATION_EVAL_TIMEOUT",
            str(
                int(num_trials * INTEGRATION_PER_TRIAL_TIMEOUT * INTEGRATION_BATCH_TIMEOUT_MULT + 120)
            ),
        )
    )
    t0 = time.time()
    timed_out = False
    exit_code = -999
    try:
        with log_path.open("w") as logf:
            proc = subprocess.run(
                cmd,
                cwd=str(PI_VLA_ROOT),
                stdout=logf,
                stderr=subprocess.STDOUT,
                timeout=max(120, t_budget),
                check=False,
            )
            exit_code = int(proc.returncode)
        elapsed = time.time() - t0
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        exit_code = -1
        timed_out = True

    session = _find_latest_integration_session(eval_dir)
    merged: Dict[str, Any] = {
        "exit_code": exit_code,
        "timed_out": timed_out,
        "elapsed_sec": elapsed,
        "session_dir": str(session) if session else None,
        "trials": [],
        "batch": None,
        "success_rate": 0.0,
        "num_success": 0,
    }
    if session is not None:
        try:
            parsed = _merge_batch_trials_into_metrics(session)
            merged.update(parsed)
        except Exception as e:
            merged["parse_error"] = str(e)
    return merged


def run_integration_eval(
    model: nn.Module,
    z_dim: int,
    epoch: int,
    live_ckpt_path: Path,
    vocab_size: Optional[int],
    num_trials: int = INTEGRATION_NUM_TRIALS,
    integration_max_workers: Optional[int] = None,
    wandb_step: Optional[int] = None,
) -> dict:
    """
    Run the end-to-end Isaac Gym + NTField pipeline ``num_trials`` times in a
    **single** subprocess (``--num_trials`` on ``run_integrated_pipeline_latent_multi_obj.py``),
    then parse ``batch_summary.json`` / per-trial ``pipeline_summary.json``.

    ``integration_max_workers`` is kept for API compatibility but is ignored:
    parallel multi-process integration is disabled in favor of one batch run
    per epoch (matches NTField load-once inside the integration script).

    ``wandb_step`` should be the same global training step used for batch logs
    (e.g. ``global_batch_counter[0]``) so media and scalars stay monotonic in W&B.

    Override ``INTEGRATION_VIDEOS_PER_EVAL`` env to upload more than the default
    first-two-trial videos each epoch.
    """
    _ = integration_max_workers  # API compat; batch subprocess only

    videos_cap = max(0, int(os.getenv("INTEGRATION_VIDEOS_PER_EVAL", str(INTEGRATION_VIDEOS_PER_EVAL))))
    wandb_vid_fps = max(1, int(round(float(os.getenv("INTEGRATION_WANDB_VIDEO_FPS", "30")))))

    was_training = model.training
    model.eval()

    _save_live_checkpoint_for_integration(model, z_dim, live_ckpt_path, vocab_size=vocab_size)

    eval_dir = INTEGRATION_OUTPUT_ROOT / f"epoch_{epoch:03d}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    trial_records: List[Dict[str, Any]] = []
    per_trial_times: List[float] = []
    per_trial_latent_l2: List[float] = []
    per_trial_path_len: List[float] = []
    per_trial_goal_l2: List[float] = []
    per_trial_ee: List[float] = []
    n_success = 0

    wandb_table_rows: List[List[Any]] = []
    wandb_video_payload: Dict[str, Any] = {}
    wandb_topview_payload: Dict[str, Any] = {}

    print(
        f" -> Running integrated evaluation: {num_trials} trials in one subprocess "
        f"(epoch {epoch}), outputs under {eval_dir}/<timestamp>/",
        flush=True,
    )

    batch = _run_integration_subprocess(eval_dir, live_ckpt_path, epoch, num_trials)
    exit_code = int(batch.get("exit_code", -999))
    timed_out = bool(batch.get("timed_out", False))
    elapsed_total = float(batch.get("elapsed_sec", 0.0))
    session_dir_s = batch.get("session_dir")
    raw_trials: List[Dict[str, Any]] = list(batch.get("trials") or [])
    if not raw_trials and num_trials > 0:
        for t in range(num_trials):
            raw_trials.append(
                {
                    "trial": t,
                    "success": False,
                    "latent_error_l2": None,
                    "trial_dir": str(eval_dir),
                    "video": None,
                    "exit_code": exit_code,
                    "timed_out": timed_out,
                    "error": batch.get("parse_error") or batch.get("run_error"),
                    "ee_to_target_dist_m": None,
                    "ee_thresh_success": None,
                }
            )

    n_t = max(len(raw_trials), 1)
    per_wall = elapsed_total / float(n_t)

    for rec in raw_trials:
        trial = int(rec.get("trial", 0))
        elapsed = float(rec.get("elapsed_sec", per_wall))
        ex = int(rec.get("exit_code", exit_code))
        tout = bool(rec.get("timed_out", timed_out))
        success = bool(rec.get("success", False))
        latent_error_l2 = rec.get("latent_error_l2")
        trial_dir = Path(str(rec.get("trial_dir", eval_dir)))
        path_len = rec.get("path_length")
        stop_r = rec.get("planner_stop_reason")
        goal_l2 = rec.get("latent_goal_l2")
        ee_dist = rec.get("ee_to_target_dist_m")
        ee_thresh_ok = rec.get("ee_thresh_success")

        if success:
            n_success += 1
        per_trial_times.append(elapsed)
        if isinstance(latent_error_l2, (int, float)) and latent_error_l2 == latent_error_l2:
            per_trial_latent_l2.append(float(latent_error_l2))
        if isinstance(path_len, int):
            per_trial_path_len.append(float(path_len))
        if isinstance(goal_l2, (int, float)) and goal_l2 == goal_l2:
            per_trial_goal_l2.append(float(goal_l2))
        if isinstance(ee_dist, (int, float)) and ee_dist == ee_dist:
            per_trial_ee.append(float(ee_dist))

        trial_records.append(
            {
                "trial": trial,
                "elapsed_sec": elapsed,
                "exit_code": ex,
                "timed_out": tout,
                "success": success,
                "latent_error_l2": latent_error_l2,
                "path_length": path_len,
                "planner_stop_reason": stop_r,
                "latent_goal_l2": goal_l2,
                "planner_final_latent_dist": rec.get("planner_final_latent_dist"),
                "trial_dir": str(trial_dir),
                "video": rec.get("video"),
                "error": rec.get("error"),
                "ee_to_target_dist_m": ee_dist,
                "ee_thresh_success": ee_thresh_ok,
            }
        )
        print(
            f"    trial {trial + 1:02d}/{num_trials}: "
            f"success={success} | planner_final_latent_dist={latent_error_l2} | "
            f"path_len={path_len} | ee_m={ee_dist} | stop={stop_r} | "
            f"time={elapsed:6.2f}s | exit={ex}"
            + (" | TIMEOUT" if tout else ""),
            flush=True,
        )

        if _wandb_active():
            video_path = rec.get("video")
            top_view_path = trial_dir / "top_view.png"
            vp: Optional[Path] = None
            if video_path:
                vp = Path(str(video_path))
                if not vp.is_file():
                    alt = trial_dir / vp.name
                    if alt.is_file():
                        vp = alt
                    else:
                        vp = None
                        if trial < videos_cap:
                            print(
                                f"    [wandb] skip video trial {trial}: path not found {video_path!r}",
                                flush=True,
                            )

            vid_key = f"integration/video/epoch_{epoch:03d}/trial_{trial:02d}"
            if (
                vp is not None
                and vp.is_file()
                and trial < videos_cap
            ):
                try:
                    if vp.stat().st_size < 64:
                        print(f"    [wandb] skip video trial {trial}: file too small {vp}", flush=True)
                    else:
                        wandb_video_payload[vid_key] = wandb.Video(
                            str(vp.resolve()),
                            caption=f"epoch {epoch} | trial {trial} | success={success}",
                            format="mp4",
                            fps=wandb_vid_fps,
                        )
                except Exception as e:
                    print(f"    [wandb] video upload failed for trial {trial}: {e}", flush=True)

            if top_view_path.is_file() and trial < videos_cap:
                try:
                    wandb_topview_payload[
                        f"integration/top_view/epoch_{epoch:03d}/trial_{trial:02d}"
                    ] = wandb.Image(
                        str(top_view_path),
                        caption=f"epoch {epoch} | trial {trial} | success={success}",
                    )
                except Exception as e:
                    print(f"    [wandb] top_view upload failed for trial {trial}: {e}", flush=True)

            wandb_table_rows.append(
                [
                    epoch,
                    trial,
                    bool(success),
                    float(elapsed),
                    int(ex),
                    bool(tout),
                    (float(latent_error_l2) if isinstance(latent_error_l2, (int, float)) else None),
                    (int(path_len) if isinstance(path_len, int) else None),
                    (str(stop_r) if stop_r is not None else None),
                    (float(goal_l2) if isinstance(goal_l2, (int, float)) else None),
                    (float(ee_dist) if isinstance(ee_dist, (int, float)) else None),
                    ee_thresh_ok,
                    str(trial_dir),
                ]
            )

    times = np.asarray(per_trial_times, dtype=np.float64) if per_trial_times else np.zeros(0)
    latent_l2_arr = (
        np.asarray(per_trial_latent_l2, dtype=np.float64)
        if per_trial_latent_l2
        else np.zeros(0)
    )
    path_arr = np.asarray(per_trial_path_len, dtype=np.float64) if per_trial_path_len else np.zeros(0)
    goal_l2_arr = np.asarray(per_trial_goal_l2, dtype=np.float64) if per_trial_goal_l2 else np.zeros(0)
    ee_arr = np.asarray(per_trial_ee, dtype=np.float64) if per_trial_ee else np.zeros(0)

    batch_payload = batch.get("batch")
    agg: Dict[str, Any] = {
        "epoch": epoch,
        "num_trials": num_trials,
        "num_success": n_success,
        "success_rate": (n_success / num_trials) if num_trials > 0 else 0.0,
        "subprocess_exit_code": exit_code,
        "subprocess_timed_out": timed_out,
        "integration_session_dir": session_dir_s,
        "mean_time_sec": float(times.mean()) if times.size else 0.0,
        "median_time_sec": float(np.median(times)) if times.size else 0.0,
        "min_time_sec": float(times.min()) if times.size else 0.0,
        "max_time_sec": float(times.max()) if times.size else 0.0,
        "std_time_sec": float(times.std()) if times.size else 0.0,
        "wall_time_sec": elapsed_total,
        "num_trials_with_latent_error_l2": int(latent_l2_arr.size),
        "mean_true_latent_error_l2": float(latent_l2_arr.mean()) if latent_l2_arr.size else None,
        "median_true_latent_error_l2": float(np.median(latent_l2_arr)) if latent_l2_arr.size else None,
        "min_true_latent_error_l2": float(latent_l2_arr.min()) if latent_l2_arr.size else None,
        "max_true_latent_error_l2": float(latent_l2_arr.max()) if latent_l2_arr.size else None,
        "mean_path_length": float(path_arr.mean()) if path_arr.size else None,
        "mean_latent_goal_l2": float(goal_l2_arr.mean()) if goal_l2_arr.size else None,
        "num_trials_with_ee_dist": int(ee_arr.size),
        "mean_ee_to_target_dist_m": float(ee_arr.mean()) if ee_arr.size else None,
        "median_ee_to_target_dist_m": float(np.median(ee_arr)) if ee_arr.size else None,
        "min_ee_to_target_dist_m": float(ee_arr.min()) if ee_arr.size else None,
        "max_ee_to_target_dist_m": float(ee_arr.max()) if ee_arr.size else None,
        "batch_summary": batch_payload,
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
            f.write(
                "epoch,num_trials,num_success,success_rate,wall_time_sec,mean_time_sec,"
                "median_time_sec,min_time_sec,max_time_sec,"
                "num_trials_with_latent_error_l2,mean_true_latent_error_l2,"
                "mean_path_length,mean_latent_goal_l2,mean_ee_to_target_dist_m,"
                "num_trials_with_ee_dist,subprocess_exit_code\n"
            )
        mean_latent_l2_csv = (
            ""
            if agg["mean_true_latent_error_l2"] is None
            else f"{agg['mean_true_latent_error_l2']:.6f}"
        )
        mpl = agg["mean_path_length"]
        mgl = agg["mean_latent_goal_l2"]
        mpl_s = "" if mpl is None else f"{mpl:.4f}"
        mgl_s = "" if mgl is None else f"{mgl:.6f}"
        mee = agg["mean_ee_to_target_dist_m"]
        mee_s = "" if mee is None else f"{mee:.6f}"
        nee = agg["num_trials_with_ee_dist"]
        f.write(
            f"{agg['epoch']},{agg['num_trials']},{agg['num_success']},"
            f"{agg['success_rate']:.6f},{agg['wall_time_sec']:.4f},"
            f"{agg['mean_time_sec']:.4f},"
            f"{agg['median_time_sec']:.4f},{agg['min_time_sec']:.4f},"
            f"{agg['max_time_sec']:.4f},"
            f"{agg['num_trials_with_latent_error_l2']},"
            f"{mean_latent_l2_csv},{mpl_s},{mgl_s},{mee_s},{nee},"
            f"{agg['subprocess_exit_code']}\n"
        )

    print(
        f" -> Integrated eval @ epoch {epoch}: "
        f"success {n_success}/{num_trials} ({100.0*agg['success_rate']:.1f}%) | "
        f"wall {agg['wall_time_sec']:.1f}s | "
        f"mean per-trial time {agg['mean_time_sec']:.2f}s | "
        f"session={session_dir_s}",
        flush=True,
    )

    # ---- wandb: scalars + table, then media (second log, same step) ----
    if _wandb_active():
        payload: dict = {
            "epoch": epoch,
            "integration/success_rate":  agg["success_rate"],
            "integration/num_success":   agg["num_success"],
            "integration/num_trials":    agg["num_trials"],
            "integration/subprocess_exit_code": agg["subprocess_exit_code"],
            "integration/subprocess_timed_out": int(bool(agg["subprocess_timed_out"])),
            "integration/wall_time_sec": agg["wall_time_sec"],
            "integration/mean_time_sec": agg["mean_time_sec"],
            "integration/median_time_sec": agg["median_time_sec"],
            "integration/min_time_sec":    agg["min_time_sec"],
            "integration/max_time_sec":    agg["max_time_sec"],
            "integration/std_time_sec":    agg["std_time_sec"],
            "integration/num_trials_with_latent_error_l2": agg["num_trials_with_latent_error_l2"],
        }
        if agg["mean_path_length"] is not None:
            payload["integration/mean_path_length"] = agg["mean_path_length"]
        if agg["mean_latent_goal_l2"] is not None:
            payload["integration/mean_latent_goal_l2"] = agg["mean_latent_goal_l2"]
        if agg["mean_ee_to_target_dist_m"] is not None:
            payload["integration/mean_ee_to_target_dist_m"] = agg["mean_ee_to_target_dist_m"]
            payload["integration/median_ee_to_target_dist_m"] = agg["median_ee_to_target_dist_m"]
            payload["integration/min_ee_to_target_dist_m"] = agg["min_ee_to_target_dist_m"]
            payload["integration/max_ee_to_target_dist_m"] = agg["max_ee_to_target_dist_m"]
            payload["integration/num_trials_with_ee_dist"] = agg["num_trials_with_ee_dist"]
        if agg["mean_true_latent_error_l2"] is not None:
            payload["integration/mean_true_latent_error_l2"] = agg["mean_true_latent_error_l2"]
            payload["integration/median_true_latent_error_l2"] = agg["median_true_latent_error_l2"]
            payload["integration/min_true_latent_error_l2"] = agg["min_true_latent_error_l2"]
            payload["integration/max_true_latent_error_l2"] = agg["max_true_latent_error_l2"]

        # Per-trial table (nice sortable/filterable view in wandb UI).
        try:
            trials_table = wandb.Table(
                columns=[
                    "epoch",
                    "trial",
                    "success",
                    "elapsed_sec",
                    "exit_code",
                    "timed_out",
                    "planner_final_latent_dist",
                    "path_length",
                    "planner_stop_reason",
                    "latent_goal_l2",
                    "ee_to_target_dist_m",
                    "ee_thresh_success",
                    "trial_dir",
                ],
                data=wandb_table_rows,
            )
            payload[f"integration/trials_table_epoch_{epoch:03d}"] = trials_table
        except Exception as e:
            print(f"[wandb] trial table log failed: {e}", flush=True)

        # Log scalars + table first, then media in a second call (same step). Large
        # mp4s + Table in one payload sometimes fails to render video in the UI.
        media_merged = {**wandb_video_payload, **wandb_topview_payload}
        _wandb_log(payload, step=wandb_step)
        if media_merged:
            _wandb_log(media_merged, step=wandb_step)
            print(
                f"[wandb] integration media @ step {wandb_step}: "
                f"{len(wandb_video_payload)} videos (WANDB fps={wandb_vid_fps}), "
                f"{len(wandb_topview_payload)} top_views",
                flush=True,
            )
        elif videos_cap > 0:
            print(
                f"[wandb] integration: no video/image media logged "
                f"(videos_cap={videos_cap}; missing mp4 under trial dirs?)",
                flush=True,
            )

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
    dropout_p = float(os.getenv("TRAIN_DROPOUT", "0.2"))

    model = get_model_wonorm(
        output_dim=z_dim,
        vocab_size=len(token_to_id),
        dropout_p=dropout_p,
    ).to(device)

    epochs = int(os.getenv("TRAIN_EPOCHS", "30"))
    eval_every = int(os.getenv("TRAIN_EVAL_EVERY", "2"))
    batch_size = int(os.getenv("TRAIN_BATCH_SIZE", "128"))
    lr = float(os.getenv("TRAIN_LR", "1e-4"))
    kl_beta_start = float(os.getenv("TRAIN_KL_BETA_START", "0.0"))
    kl_beta_end = float(os.getenv("TRAIN_KL_BETA_END", "1.0"))
    kl_beta_schedule = os.getenv("TRAIN_KL_BETA_SCHEDULE", "linear").strip().lower()
    kl_anneal_steps = int(os.getenv("TRAIN_KL_ANNEAL_STEPS", "20000"))
    kl_cycle_steps = int(os.getenv("TRAIN_KL_CYCLE_STEPS", "10000"))
    z_align_weight = float(os.getenv("TRAIN_Z_ALIGN_WEIGHT", "0.1"))
    if kl_beta_schedule not in {"linear", "cyclical"}:
        raise ValueError(
            f"TRAIN_KL_BETA_SCHEDULE must be 'linear' or 'cyclical', got: {kl_beta_schedule}"
        )

    # Approximate optimizer steps per epoch (one full pass over train indices).
    batches_per_epoch = max(1, (train_size + batch_size - 1) // batch_size)
    approx_total_steps = batches_per_epoch * epochs
    approx_kl_cycles = approx_total_steps / max(kl_cycle_steps, 1)
    print(
        f"Step budget (train): n_total={n_total} train_size={train_size} | "
        f"~{batches_per_epoch} batches/epoch @ batch_size={batch_size} | "
        f"~{approx_total_steps} steps over {epochs} epochs | "
        f"~{approx_kl_cycles:.1f} full β cycles (TRAIN_KL_CYCLE_STEPS={kl_cycle_steps})",
        flush=True,
    )

    best_val_mae = float("inf")
    best_val_cos = float("inf")
    best_val_n   = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # CVAE uses fixed reconstruction loss (MSE) + KL term in run_batches.
    criterion = nn.MSELoss()
        
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    pca_output_dir = Path(os.getenv("TRAIN_PCA_OUTPUT_DIR", str(PCA_OUTPUT_DIR)))
    wandb_enabled = _env_bool("WANDB_ENABLED", WANDB_ENABLED)
    wandb_run_name = _env_opt_str("WANDB_RUN_NAME", WANDB_RUN_NAME)
    wandb_log_batches = _env_bool("WANDB_LOG_BATCHES", WANDB_LOG_BATCHES)
    wandb_batch_log_every = int(os.getenv("WANDB_BATCH_LOG_EVERY", str(WANDB_BATCH_LOG_EVERY)))
    train_ckpt_name = _env_opt_str("TRAIN_CKPT_NAME", None)

    print(
        f"Config | epochs={epochs} eval_every={eval_every} batch_size={batch_size} "
        f"lr={lr:.6g} recon_loss=mse "
        f"dropout={dropout_p:.3f}",
        flush=True,
    )
    print(
        "KL beta | "
        f"schedule={kl_beta_schedule} start={kl_beta_start:.6f} end={kl_beta_end:.6f} "
        f"anneal_steps={kl_anneal_steps} cycle_steps={kl_cycle_steps} "
        f"| z_align_weight={z_align_weight:.4f}",
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
    run_integration_every = max(1, int(os.getenv("RUN_INTEGRATION_EVERY", str(RUN_INTEGRATION_EVERY))))
    integration_num_trials = max(1, int(os.getenv("INTEGRATION_NUM_TRIALS", str(INTEGRATION_NUM_TRIALS))))
    integration_on_val_improve_only = _env_bool("INTEGRATION_ON_VAL_IMPROVE_ONLY", True)
    integration_force_final = _env_bool("INTEGRATION_FORCE_FINAL_EPOCH", False)
    if integration_on_val_improve_only:
        integ_mode = (
            f"on val MAE improvement (set INTEGRATION_ON_VAL_IMPROVE_ONLY=0 to use every "
            f"{run_integration_every} ep; INTEGRATION_FORCE_FINAL_EPOCH=1 to also run last epoch)"
        )
    else:
        integ_mode = f"every {run_integration_every} epochs (RUN_INTEGRATION_EVERY) and last epoch"
    print(
        f"Integration | live_ckpt={integration_live_ckpt} | {integ_mode} | "
        f"trials={integration_num_trials} (INTEGRATION_NUM_TRIALS) | "
        f"timeout mult={INTEGRATION_BATCH_TIMEOUT_MULT} (INTEGRATION_BATCH_TIMEOUT_MULT)",
        flush=True,
    )

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
            "loss": "cvae_recon_mse_plus_beta_kl",
            "z_align_weight": z_align_weight,
            "kl_beta_schedule": kl_beta_schedule,
            "kl_beta_start": kl_beta_start,
            "kl_beta_end": kl_beta_end,
            "kl_anneal_steps": kl_anneal_steps,
            "kl_cycle_steps": kl_cycle_steps,
            "model": "StudentModelWonorm(ResNet18 + text prompt embedding fusion -> MLP 512 -> z_dim)",
            "dropout": dropout_p,
            "prompt_template": "grasp {object_name}",
            "text_vocab_size": len(token_to_id),
            "max_prompt_len": max_prompt_len,
            "run_integration_every": run_integration_every,
            "integration_on_val_improve_only": integration_on_val_improve_only,
            "integration_force_final_epoch": integration_force_final,
            "integration_num_trials": integration_num_trials,
            "integration_videos_per_eval": INTEGRATION_VIDEOS_PER_EVAL,
            "integration_ntfield_tol": INTEGRATION_NTFIELD_TOL,
            "integration_ntfield_max_steps": INTEGRATION_NTFIELD_MAX_STEPS,
            "integration_ntfield_step_size": INTEGRATION_NTFIELD_STEP_SIZE,
            "integration_batch_subprocess": True,
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
            did_new_best = False
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
                    kl_beta_schedule, kl_beta_start, kl_beta_end, kl_anneal_steps, kl_cycle_steps,
                    z_align_weight,
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

            # Per-epoch training scalars (same step as batch logs — avoids mixed implicit/explicit step).
            _wandb_log(
                {
                    "epoch": epoch + 1,
                    "train/loss": avg_train_loss,
                    "train/lr": current_lr,
                    "train/num_batches": n_batches_total,
                },
                step=global_batch_counter[0],
            )

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
                _wandb_log(val_payload, step=global_batch_counter[0])

                did_new_best = val_mae < best_val_mae
                if did_new_best:
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
                        "latent_dim": int(getattr(model, "latent_dim", 128)),
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
                    _wandb_log(
                        {
                            "epoch": epoch + 1,
                            "val/best_mae": best_val_mae,
                            "val/best_cos_distance": best_val_cos,
                            "val/best_epoch": epoch + 1,
                        },
                        step=global_batch_counter[0],
                    )
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
            # Integrated-pipeline evaluation (Isaac Gym + NTField)
            # Default: only when validation MAE improves. Optional: legacy schedule
            # (INTEGRATION_ON_VAL_IMPROVE_ONLY=0) or final-epoch run (INTEGRATION_FORCE_FINAL_EPOCH=1).
            # ------------------------------------------------------------------
            run_integration = False
            if integration_on_val_improve_only:
                if did_new_best:
                    run_integration = True
                elif integration_force_final and (epoch + 1) == epochs:
                    run_integration = True
            else:
                if (epoch + 1) % run_integration_every == 0 or (epoch + 1) == epochs:
                    run_integration = True

            if run_integration:
                try:
                    run_integration_eval(
                        model,
                        z_dim=z_dim,
                        epoch=epoch + 1,
                        live_ckpt_path=integration_live_ckpt,
                        vocab_size=len(token_to_id),
                        num_trials=integration_num_trials,
                        wandb_step=global_batch_counter[0],
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