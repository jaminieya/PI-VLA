"""
full_train_multi_mdn.py — MDN training script for multimodal z_goal prediction.

Drop-in replacement for full_train_multi_cvae.py. Key differences:
  - Uses MDNStudent instead of CVAEStudent
  - Loss = NLL of z_goal under the predicted Gaussian mixture (no KL term)
  - No beta / KL schedule needed
  - Val eval uses deterministic best-component prediction for MSE/MAE/Cos metrics
  - All other infrastructure (shards, wandb, integration, PCA) unchanged

Environment variables (same as before, new ones marked *NEW*):
  TRAIN_EPOCHS            default 60
  TRAIN_EVAL_EVERY        default 3
  TRAIN_BATCH_SIZE        default 256
  TRAIN_LR                default 3e-4
  TRAIN_DROPOUT           default 0.2
  MDN_N_COMPONENTS        *NEW* number of mixture components  default 8
  TRAIN_CKPT_NAME         checkpoint filename template
  TRAIN_PCA_OUTPUT_DIR    PCA plot directory
  WANDB_ENABLED / WANDB_RUN_NAME / WANDB_LOG_BATCHES / WANDB_BATCH_LOG_EVERY
  RUN_INTEGRATION_EVERY / INTEGRATION_NUM_TRIALS
  INTEGRATION_ON_VAL_IMPROVE_ONLY (default True)
  INTEGRATION_FORCE_FINAL_EPOCH   (default False)
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch import nn
from torchvision import models, transforms
from student_model_mdn import MDNStudent

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    _WANDB_AVAILABLE = False

# ── Paths ────────────────────────────────────────────────────────────────────
DATASET_ROOT     = Path("/home/hojinsohn/VLM-NT/PI-VLA/student_model_training/data/pt_shards_multi")
PCA_OUTPUT_DIR   = Path("/home/hojinsohn/VLM-NT/PI-VLA/output/pca_training_plots_multi_mdn")
LOG_EVERY_BATCHES = 100

PI_VLA_ROOT            = Path("/home/hojinsohn/VLM-NT/PI-VLA")
INTEGRATION_SCRIPT     = PI_VLA_ROOT / "final_integrate" / "run_integrated_pipeline_latent_multi_obj_mdn.py"
NTFIELD_CHECKPOINT     = str(PI_VLA_ROOT / "teacher_model.pt")
INTEGRATION_OUTPUT_ROOT = PI_VLA_ROOT / "output" / "integration_eval_during_training_mdn"
INTEGRATION_LIVE_CKPT  = PI_VLA_ROOT / "final_integrate" / "_training_live_latent_mdn.pth"

RUN_INTEGRATION_EVERY       = 20
INTEGRATION_NUM_TRIALS      = 10
INTEGRATION_PER_TRIAL_TIMEOUT = 300
INTEGRATION_BATCH_TIMEOUT_MULT = float(os.getenv("INTEGRATION_BATCH_TIMEOUT_MULT", "1.25"))
INTEGRATION_NTFIELD_TOL     = 0.01
INTEGRATION_NTFIELD_MAX_STEPS = 200
INTEGRATION_NTFIELD_STEP_SIZE = 0.02
INTEGRATION_VIDEOS_PER_EVAL = 2

# ── wandb defaults ────────────────────────────────────────────────────────────
WANDB_ENABLED        = True
WANDB_PROJECT        = "pi-vla-latent-goal"
WANDB_ENTITY         = None
WANDB_RUN_NAME       = "mdn_multi_obj"
WANDB_LOG_BATCHES    = True
WANDB_BATCH_LOG_EVERY = 10


# ── Helpers (identical to CVAE script) ───────────────────────────────────────

def _env_bool(name, default):
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}

def _env_opt_str(name, default):
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip()
    return default if v == "" else v

def _wandb_active():
    return WANDB_ENABLED and _WANDB_AVAILABLE and wandb.run is not None

def _wandb_log(data, step=None):
    if not _wandb_active():
        return
    try:
        wandb.log(data, step=step) if step is not None else wandb.log(data)
    except Exception as e:
        print(f"[wandb] log failed: {e}", flush=True)


# ── Shard helpers (unchanged) ─────────────────────────────────────────────────

def label_tensor_from_shard(shard):
    if "z_goals" in shard:
        return shard["z_goals"]
    if "configs" in shard:
        c = shard["configs"]
        return c[:, -1, :] if c.dim() == 3 else c
    if "obj_locs" in shard:
        return shard["obj_locs"]
    raise ValueError("Shard must contain 'z_goals', 'configs', or 'obj_locs'.")

def prompt_from_object_name(name):
    return f"grasp {str(name).strip().lower()}"

def _tokenize_prompt(text):
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

def _is_list_shard(s):
    return isinstance(s, list)

def discover_shards(root):
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
    for sp in shard_files:
        shard = torch.load(sp, map_location="cpu")
        if _is_list_shard(shard):
            if not shard:
                continue
            n = len(shard)
            s0 = shard[0]
            if "z_goal" not in s0 or "image" not in s0:
                raise ValueError(f"List shard missing 'image'/'z_goal' in {sp}")
            if z_dim is None:
                z_dim = int(s0["z_goal"].numel())
            for dp in shard:
                if "object_name" in dp:
                    object_names.add(str(dp["object_name"]))
        else:
            if "images" not in shard:
                raise ValueError(f"Missing 'images' in {sp}")
            labels = label_tensor_from_shard(shard)
            n = shard["images"].shape[0]
            if z_dim is None:
                z_dim = int(labels.shape[-1])
        total += n
        cumulative.append(total)
    print(f"Found {total} total samples across {len(shard_files)} shards. z_dim: {z_dim}", flush=True)
    return shard_files, cumulative, z_dim, sorted(object_names)

def build_train_val_split(n_total, val_fraction=0.1, seed=42):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=g)
    val_size  = int(val_fraction * n_total)
    train_size = n_total - val_size
    return set(perm[:train_size].tolist()), set(perm[train_size:].tolist()), train_size, val_size

def train_val_indices_for_shard(shard_idx, cumulative, train_idx, val_idx):
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


# ── Batch runner ──────────────────────────────────────────────────────────────

def run_batches(
    model, optimizer, device,
    shard_path, local_indices, normalize,
    batch_size, train, epoch, epochs,
    batch_counter, wandb_log_batches, wandb_batch_log_every,
    token_to_id, max_prompt_len,
):
    """
    MDN training loop. Loss = NLL of z_goal under the mixture.
    No KL term, no beta schedule.
    """
    if not local_indices:
        return 0.0, 0

    shard = torch.load(shard_path, map_location="cpu")
    list_shard = _is_list_shard(shard)
    if not list_shard:
        images  = shard["images"]
        z_goals = label_tensor_from_shard(shard)

    perm = torch.randperm(len(local_indices)) if train else torch.arange(len(local_indices))
    locals_shuffled = [local_indices[i] for i in perm.tolist()]

    running_loss = 0.0
    n_batches    = 0
    model.train() if train else model.eval()

    for start in range(0, len(locals_shuffled), batch_size):
        chunk = locals_shuffled[start : start + batch_size]
        if list_shard:
            dps  = [shard[i] for i in chunk]
            x    = torch.stack([dp["image"] for dp in dps]).to(device, non_blocking=True)
            y    = torch.stack([dp["z_goal"] for dp in dps]).to(device, non_blocking=True)
            prompts = [prompt_from_object_name(dp.get("object_name", "object")) for dp in dps]
        else:
            x    = images[chunk].to(device, non_blocking=True)
            y    = z_goals[chunk].to(device, non_blocking=True)
            prompts = ["grasp object"] * len(chunk)

        text_tokens = encode_prompts(prompts, token_to_id, max_prompt_len).to(device, non_blocking=True)
        x = normalize(x)

        if train:
            optimizer.zero_grad(set_to_none=True)
            # MDNStudent.forward returns (nll_loss, zeros) when z_goal is given
            nll_loss, _ = model(x, text_tokens, z_goal=y)
            nll_loss.backward()
            # Gradient clipping — MDN losses can spike with very tight sigmas
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            loss = nll_loss
        else:
            with torch.no_grad():
                # Val: use deterministic best-component prediction → MSE
                z_pred = model.predict_best(x, text_tokens)
                loss   = nn.functional.mse_loss(z_pred, y)

        running_loss   += loss.item()
        n_batches      += 1
        batch_counter[0] += 1

        if train and batch_counter[0] % LOG_EVERY_BATCHES == 0:
            print(
                f"Epoch {epoch+1}/{epochs} | batch {batch_counter[0]} | "
                f"nll_loss {loss.item():.6f}",
                flush=True,
            )

        if train and wandb_log_batches and (batch_counter[0] % wandb_batch_log_every == 0):
            _wandb_log({
                "batch/train_loss": loss.item(),
                "batch/nll_loss":   loss.item(),
                "batch/train_lr":   optimizer.param_groups[0]["lr"],
                "batch/epoch":      epoch + 1,
                "batch/step":       batch_counter[0],
            }, step=batch_counter[0])

    return running_loss, n_batches


# ── Val evaluation ────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_shardwise(model, device, shard_files, cumulative, val_idx,
                       normalize, batch_size, token_to_id, max_prompt_len):
    """
    Val metrics use deterministic predict_best (highest-weight component mean).
    Reports MSE, MAE, cosine distance — same keys as CVAE script for easy comparison.
    """
    model.eval()
    mse_sum = mae_sum = cos_sum = 0.0
    n_samples = 0
    mse_crit = nn.MSELoss(reduction="sum")
    mae_crit = nn.L1Loss(reduction="sum")

    for si, sp in enumerate(shard_files):
        _, val_locals = train_val_indices_for_shard(si, cumulative, set(), val_idx)
        if not val_locals:
            continue
        shard = torch.load(sp, map_location="cpu")
        list_shard = _is_list_shard(shard)
        if not list_shard:
            images  = shard["images"]
            z_goals = label_tensor_from_shard(shard)

        for start in range(0, len(val_locals), batch_size):
            chunk = val_locals[start : start + batch_size]
            if list_shard:
                dps  = [shard[i] for i in chunk]
                x    = torch.stack([dp["image"] for dp in dps]).to(device, non_blocking=True)
                y    = torch.stack([dp["z_goal"] for dp in dps]).to(device, non_blocking=True)
                prompts = [prompt_from_object_name(dp.get("object_name", "object")) for dp in dps]
            else:
                x    = images[chunk].to(device, non_blocking=True)
                y    = z_goals[chunk].to(device, non_blocking=True)
                prompts = ["grasp object"] * len(chunk)

            text_tokens = encode_prompts(prompts, token_to_id, max_prompt_len).to(device, non_blocking=True)
            x = normalize(x)
            pred = model.predict_best(x, text_tokens)

            b = x.size(0)
            n_samples += b
            mse_sum   += mse_crit(pred, y).item()
            mae_sum   += mae_crit(pred, y).item()
            cos_sim    = nn.functional.cosine_similarity(pred, y, dim=1)
            cos_sum   += (1 - cos_sim).sum().item()

    if n_samples == 0:
        return 0.0, 0.0, 0.0, 0
    return mse_sum / n_samples, mae_sum / n_samples, cos_sum / n_samples, n_samples


# ── PCA helpers (unchanged) ───────────────────────────────────────────────────

def fit_transform_pca(x, n_components=3):
    n, d = x.shape
    mean = x.mean(axis=0)
    xc   = x - mean
    u, s, vt = np.linalg.svd(xc, full_matrices=False)
    explained = (s ** 2) / max(n - 1, 1)
    ratio = explained[:n_components] / explained.sum()
    return xc @ vt[:n_components].T, vt[:n_components], ratio

@torch.no_grad()
def collect_val_embeddings(model, device, shard_files, cumulative, val_idx,
                           normalize, batch_size, token_to_id, max_prompt_len, max_points=2000):
    model.eval()
    all_val = sorted(val_idx)
    rng = np.random.default_rng(0)
    chosen = set(rng.choice(all_val, size=min(max_points, len(all_val)), replace=False).tolist())

    preds, targets = [], []
    for si, sp in enumerate(shard_files):
        start_g   = 0 if si == 0 else cumulative[si - 1]
        shard_len = cumulative[si] - start_g
        local_idx = [j for j in range(shard_len) if (start_g + j) in chosen]
        if not local_idx:
            continue
        shard = torch.load(sp, map_location="cpu")
        list_shard = _is_list_shard(shard)
        if not list_shard:
            images  = shard["images"]
            z_goals = label_tensor_from_shard(shard)

        for start in range(0, len(local_idx), batch_size):
            chunk = local_idx[start : start + batch_size]
            if list_shard:
                dps  = [shard[i] for i in chunk]
                x    = torch.stack([dp["image"] for dp in dps]).to(device, non_blocking=True)
                y    = torch.stack([dp["z_goal"] for dp in dps])
                prompts = [prompt_from_object_name(dp.get("object_name", "object")) for dp in dps]
            else:
                x    = images[chunk].to(device, non_blocking=True)
                y    = z_goals[chunk]
                prompts = ["grasp object"] * len(chunk)
            text_tokens = encode_prompts(prompts, token_to_id, max_prompt_len).to(device, non_blocking=True)
            x = normalize(x)
            pred = model.predict_best(x, text_tokens).cpu()
            preds.append(pred)
            targets.append(y)

    return (torch.cat(preds).float().numpy(),
            torch.cat(targets).float().numpy())

def generate_pca_plot(z_pred, z_target, epoch, out_dir):
    import matplotlib.pyplot as plt
    z64t = z_target.astype(np.float64)
    z64p = z_pred.astype(np.float64)
    n    = len(z64p)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Target-axes plot
    sc_t, comp, ratio = fit_transform_pca(z64t)
    sc_p = (z64p - z64t.mean(0)) @ comp.T
    fig  = plt.figure(figsize=(9, 7))
    ax   = fig.add_subplot(111, projection="3d")
    ax.scatter(*sc_t.T, c="#1f77b4", s=6, alpha=0.55, label="ground truth")
    ax.scatter(*sc_p.T, c="#ff7f0e", s=6, alpha=0.55, label="predicted (best component)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    ax.set_title(f"Epoch {epoch} — PCA (target axes, MDN best component)")
    ax.legend(fontsize=9)
    fig.text(0.02, 0.02, f"PC1–3 var: {', '.join(f'{v*100:.1f}%' for v in ratio)} | N={n}", fontsize=9)
    fig.tight_layout()
    p1 = out_dir / f"pca_target_axes_epoch_{epoch:03d}.png"
    plt.savefig(p1, dpi=130); plt.close(fig)
    print(f"  -> PCA saved: {p1}", flush=True)

    # Joint-axes plot
    xall = np.vstack([z64p, z64t])
    scj, _, rj = fit_transform_pca(xall)
    fig  = plt.figure(figsize=(9, 7))
    ax   = fig.add_subplot(111, projection="3d")
    ax.scatter(*scj[n:].T, c="#1f77b4", s=6, alpha=0.55, label="ground truth")
    ax.scatter(*scj[:n].T, c="#ff7f0e", s=6, alpha=0.55, label="predicted")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    ax.set_title(f"Epoch {epoch} — PCA (joint axes, MDN)")
    ax.legend(fontsize=9)
    fig.text(0.02, 0.02, f"PC1–3 var: {', '.join(f'{v*100:.1f}%' for v in rj)} | N={n}", fontsize=9)
    fig.tight_layout()
    p2 = out_dir / f"pca_joint_axes_epoch_{epoch:03d}.png"
    plt.savefig(p2, dpi=130); plt.close(fig)
    print(f"  -> PCA saved: {p2}", flush=True)
    return p1, p2


# ── Integration eval (copy of CVAE version, model-agnostic) ──────────────────

def _save_live_checkpoint(model, z_dim, ckpt_path, vocab_size=None, n_components=None):
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "z_dim":        z_dim,
        "vocab_size":   vocab_size,
        "n_components": n_components,
        "model_type":   "mdn",
    }, ckpt_path)

def _find_latest_session(eval_dir):
    if not eval_dir.is_dir():
        return None
    hits = []
    for p in eval_dir.rglob("batch_summary.json"):
        hits.append(p.parent)
    for p in eval_dir.rglob("pipeline_summary.json"):
        hits.append(p.parent)
    if not hits:
        return None
    hits.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return hits[0]

def _run_integration_subprocess(eval_dir, live_ckpt, epoch, num_trials):
    eval_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(INTEGRATION_SCRIPT),
        "--ntfield_checkpoint", str(NTFIELD_CHECKPOINT),
        "--latent_checkpoint",  str(live_ckpt),
        "--output_dir",         str(eval_dir.resolve()),
        "--num_trials",         str(num_trials),
        "--seed",               str(10_000 + epoch),
        "--ntfield_step_size",  str(INTEGRATION_NTFIELD_STEP_SIZE),
        "--ntfield_max_steps",  str(INTEGRATION_NTFIELD_MAX_STEPS),
        "--ntfield_tol",        str(INTEGRATION_NTFIELD_TOL),
    ]
    log_path = eval_dir / f"integration_subprocess_epoch_{epoch:03d}.log"
    t_budget = int(os.getenv(
        "INTEGRATION_EVAL_TIMEOUT",
        str(int(num_trials * INTEGRATION_PER_TRIAL_TIMEOUT * INTEGRATION_BATCH_TIMEOUT_MULT + 120))
    ))
    t0 = time.time()
    timed_out = False
    exit_code = -999
    try:
        with log_path.open("w") as lf:
            proc = subprocess.run(cmd, cwd=str(PI_VLA_ROOT), stdout=lf,
                                  stderr=subprocess.STDOUT, timeout=max(120, t_budget), check=False)
            exit_code = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        exit_code = -1
    elapsed = time.time() - t0
    session = _find_latest_session(eval_dir)
    merged = {"exit_code": exit_code, "timed_out": timed_out,
              "elapsed_sec": elapsed, "session_dir": str(session) if session else None,
              "trials": [], "success_rate": 0.0, "num_success": 0}
    if session is not None:
        try:
            bp = session / "batch_summary.json"
            if bp.is_file():
                with bp.open() as f:
                    batch = json.load(f)
                trials_out = []
                for t in batch.get("trials") or []:
                    st = str(t.get("status","")).lower()
                    ok = st == "success"
                    dist = t.get("ntfield_final_latent_dist")
                    ee   = t.get("ee_to_target_dist_m")
                    trials_out.append({
                        "trial":   int(t.get("trial_index", len(trials_out))),
                        "success": ok,
                        "latent_error_l2": dist,
                        "planner_final_latent_dist": dist,
                        "path_length": t.get("path_length"),
                        "planner_stop_reason": t.get("planner_stop_reason"),
                        "video": t.get("predicted_video"),
                        "ee_to_target_dist_m": ee,
                        "exit_code": 0 if st != "error" else 1,
                        "timed_out": False,
                    })
                merged["trials"]       = trials_out
                merged["success_rate"] = float(batch.get("success_rate", 0.0))
                merged["num_success"]  = int(batch.get("success_count", 0))
        except Exception as e:
            merged["parse_error"] = str(e)
    return merged

def run_integration_eval(model, z_dim, epoch, live_ckpt_path,
                         vocab_size, n_components, num_trials, wandb_step=None):
    was_training = model.training
    model.eval()
    _save_live_checkpoint(model, z_dim, live_ckpt_path,
                          vocab_size=vocab_size, n_components=n_components)
    eval_dir = INTEGRATION_OUTPUT_ROOT / f"epoch_{epoch:03d}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    print(f" -> Running MDN integration eval: {num_trials} trials (epoch {epoch})", flush=True)
    batch = _run_integration_subprocess(eval_dir, live_ckpt_path, epoch, num_trials)

    n_success = 0
    raw_trials = list(batch.get("trials") or [])
    for rec in raw_trials:
        success = bool(rec.get("success", False))
        if success:
            n_success += 1
        print(
            f"    trial {rec['trial']+1:02d}/{num_trials}: "
            f"success={success} | "
            f"planner_final_latent_dist={rec.get('planner_final_latent_dist')} | "
            f"path_len={rec.get('path_length')} | "
            f"ee_m={rec.get('ee_to_target_dist_m')} | "
            f"stop={rec.get('planner_stop_reason')}",
            flush=True,
        )

    sr = n_success / num_trials if num_trials > 0 else 0.0
    print(f" -> MDN integration @ epoch {epoch}: {n_success}/{num_trials} ({100*sr:.1f}%)", flush=True)

    _wandb_log({
        "epoch": epoch,
        "integration/success_rate": sr,
        "integration/num_success":  n_success,
        "integration/num_trials":   num_trials,
    }, step=wandb_step)

    if was_training:
        model.train()
    return {"success_rate": sr, "num_success": n_success, "trials": raw_trials}


# ── Main training loop ────────────────────────────────────────────────────────

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    shard_files, cumulative, z_dim, object_names = discover_shards(DATASET_ROOT)
    token_to_id = build_text_vocab(object_names)
    max_prompt_len = int(os.getenv("TRAIN_MAX_PROMPT_LEN", "8"))
    n_total = cumulative[-1]
    train_idx, val_idx, train_size, val_size = build_train_val_split(n_total)
    print(f"Train: {train_size} | Val: {val_size}", flush=True)

    # ── Hyperparameters ───────────────────────────────────────────────────────
    epochs        = int(os.getenv("TRAIN_EPOCHS",        "60"))
    eval_every    = int(os.getenv("TRAIN_EVAL_EVERY",    "3"))
    batch_size    = int(os.getenv("TRAIN_BATCH_SIZE",    "256"))
    lr            = float(os.getenv("TRAIN_LR",          "3e-4"))
    dropout_p     = float(os.getenv("TRAIN_DROPOUT",     "0.2"))
    n_components  = int(os.getenv("MDN_N_COMPONENTS",    "8"))
    wandb_enabled      = _env_bool("WANDB_ENABLED", WANDB_ENABLED)
    wandb_run_name     = _env_opt_str("WANDB_RUN_NAME", WANDB_RUN_NAME)
    wandb_log_batches  = _env_bool("WANDB_LOG_BATCHES", WANDB_LOG_BATCHES)
    wandb_batch_log_every = int(os.getenv("WANDB_BATCH_LOG_EVERY", str(WANDB_BATCH_LOG_EVERY)))
    train_ckpt_name    = _env_opt_str("TRAIN_CKPT_NAME", None)
    pca_output_dir     = Path(os.getenv("TRAIN_PCA_OUTPUT_DIR", str(PCA_OUTPUT_DIR)))
    run_integration_every    = max(1, int(os.getenv("RUN_INTEGRATION_EVERY", str(RUN_INTEGRATION_EVERY))))
    integration_num_trials   = max(1, int(os.getenv("INTEGRATION_NUM_TRIALS", str(INTEGRATION_NUM_TRIALS))))
    integration_on_val_only  = _env_bool("INTEGRATION_ON_VAL_IMPROVE_ONLY", True)
    integration_force_final  = _env_bool("INTEGRATION_FORCE_FINAL_EPOCH", False)

    steps_per_epoch = max(1, train_size // batch_size)
    total_steps     = steps_per_epoch * epochs
    print(
        f"Step budget: n_total={n_total} train_size={train_size} | "
        f"~{steps_per_epoch} batches/epoch @ batch_size={batch_size} | "
        f"~{total_steps} steps over {epochs} epochs",
        flush=True,
    )
    print(
        f"Config | epochs={epochs} eval_every={eval_every} batch_size={batch_size} "
        f"lr={lr:.6g} dropout={dropout_p:.3f} n_components={n_components}",
        flush=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MDNStudent(
        output_dim=z_dim,
        vocab_size=len(token_to_id),
        n_components=n_components,
        dropout_p=dropout_p,
    ).to(device)
    model.count_parameters()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    run_slug = "".join(ch if ch.isalnum() or ch in {"-","_"} else "_"
                       for ch in (wandb_run_name or "run"))
    integration_live_ckpt = INTEGRATION_LIVE_CKPT.with_name(
        f"_training_live_latent_mdn_{run_slug}.pth"
    )
    INTEGRATION_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # ── wandb init ────────────────────────────────────────────────────────────
    if wandb_enabled and _WANDB_AVAILABLE:
        try:
            wandb.init(
                project=WANDB_PROJECT, entity=WANDB_ENTITY,
                name=wandb_run_name,
                config={
                    "model": "MDNStudent",
                    "n_components": n_components,
                    "output_dim": z_dim, "epochs": epochs,
                    "batch_size": batch_size, "lr": lr,
                    "dropout": dropout_p, "loss": "nll_mixture",
                    "scheduler": "CosineAnnealingLR",
                    "vocab_size": len(token_to_id),
                    "train_size": train_size, "val_size": val_size,
                },
                dir=str(PI_VLA_ROOT / "output"),
            )
            print(f"[wandb] run initialized: {wandb.run.name}", flush=True)
        except Exception as e:
            print(f"[wandb] init failed: {e}", flush=True)

    best_val_mae = float("inf")
    best_val_cos = float("inf")
    global_batch_counter = [0]

    try:
        for epoch in range(epochs):
            did_new_best = False
            shard_order  = torch.randperm(len(shard_files)).tolist()
            running_loss = 0.0
            n_batches_total = 0

            for si in shard_order:
                train_locals, _ = train_val_indices_for_shard(si, cumulative, train_idx, val_idx)
                loss_sum, nb = run_batches(
                    model, optimizer, device,
                    shard_files[si], train_locals, normalize,
                    batch_size, True, epoch, epochs, global_batch_counter,
                    wandb_log_batches, wandb_batch_log_every,
                    token_to_id, max_prompt_len,
                )
                running_loss    += loss_sum
                n_batches_total += nb

            scheduler.step()
            avg_train_loss = running_loss / max(n_batches_total, 1)
            current_lr     = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch+1}/{epochs} | Train NLL: {avg_train_loss:.6f} "
                f"| LR: {current_lr:.8f}",
                flush=True,
            )
            _wandb_log({"epoch": epoch+1, "train/nll_loss": avg_train_loss,
                        "train/lr": current_lr})

            # ── Val eval ──────────────────────────────────────────────────────
            if (epoch + 1) % eval_every == 0 or (epoch + 1) == epochs:
                val_mse, val_mae, val_cos, val_n = evaluate_shardwise(
                    model, device, shard_files, cumulative, val_idx,
                    normalize, batch_size, token_to_id, max_prompt_len,
                )
                print(
                    f" -> Val MSE: {val_mse:.6f} | Val MAE: {val_mae:.6f} "
                    f"| Val Cos: {val_cos:.6f} | N: {val_n}",
                    flush=True,
                )

                z_pred, z_target = collect_val_embeddings(
                    model, device, shard_files, cumulative, val_idx,
                    normalize, batch_size, token_to_id, max_prompt_len,
                )
                pca_t, pca_j = generate_pca_plot(z_pred, z_target, epoch+1, pca_output_dir)

                val_payload = {
                    "epoch": epoch+1,
                    "val/mse": val_mse, "val/mae": val_mae,
                    "val/cos_distance": val_cos, "val/n_samples": val_n,
                }
                if _wandb_active():
                    try:
                        val_payload["val/pca_target_axes"] = wandb.Image(str(pca_t))
                        val_payload["val/pca_joint_axes"]  = wandb.Image(str(pca_j))
                    except Exception as e:
                        print(f"[wandb] PCA image failed: {e}", flush=True)
                _wandb_log(val_payload)

                did_new_best = val_mae < best_val_mae
                if did_new_best:
                    best_val_mae = val_mae
                    best_val_cos = val_cos
                    ckpt_suffix  = wandb_run_name or "run"
                    ckpt_name    = (train_ckpt_name or
                                    f"best_z_goal_model_mdn_{{run_name}}.pth"
                                    ).replace("{run_name}", ckpt_suffix)
                    if not ckpt_name.endswith(".pth"):
                        ckpt_name += ".pth"
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "z_dim": z_dim, "vocab_size": len(token_to_id),
                        "token_to_id": token_to_id, "max_prompt_len": max_prompt_len,
                        "n_components": n_components,
                        "epoch": epoch+1, "val_mae": best_val_mae, "val_cos": best_val_cos,
                        "model_type": "mdn",
                    }, ckpt_name)
                    print(f" -> New best saved. MAE: {best_val_mae:.6f} | Cos: {best_val_cos:.6f}",
                          flush=True)
                    _wandb_log({"epoch": epoch+1, "val/best_mae": best_val_mae,
                                "val/best_cos_distance": best_val_cos})

            # ── Integration eval ──────────────────────────────────────────────
            run_integ = False
            if integration_on_val_only:
                if did_new_best:
                    run_integ = True
                elif integration_force_final and (epoch + 1) == epochs:
                    run_integ = True
            else:
                if (epoch + 1) % run_integration_every == 0 or (epoch + 1) == epochs:
                    run_integ = True

            if run_integ:
                try:
                    run_integration_eval(
                        model, z_dim=z_dim, epoch=epoch+1,
                        live_ckpt_path=integration_live_ckpt,
                        vocab_size=len(token_to_id),
                        n_components=n_components,
                        num_trials=integration_num_trials,
                        wandb_step=global_batch_counter[0],
                    )
                except Exception as e:
                    print(f" -> Integration eval failed: {e}", flush=True)

    finally:
        if _wandb_active():
            try:
                wandb.finish()
            except Exception:
                pass


if __name__ == "__main__":
    train()