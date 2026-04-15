#!/usr/bin/env python3
"""
train_fast.py

Trains FiLM + MLP head on pre-extracted ResNet/CLIP feature shards
produced by extract_features.py.

Since encoders are already baked into the feature shards, each training
batch is just: load (img_feat, txt_feat, xyz) → FiLM → MLP → loss.
No image loading, no encoder forward pass. GPU stays at ~100% util.

Architecture
------------
  img_feat  (2048,) ──► proj ──► (512,) ──┐
                                            ├─► FiLM ──► MLP ──► (3,)
  txt_feat   (512,) ──────────────────────┘

Usage
-----
  python train_fast.py \
    --data-dir /path/to/multi_obj_layout_feat_shards \
    --output-dir /path/to/runs/exp_fast \
    --epochs 100 --batch-size 512 --lr 3e-4
"""
from __future__ import annotations

import argparse
import bisect
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_DATA_DIR   = Path("/home/hojinsohn/VLM-NT/PI-VLA/output/multi_obj_layout_feat_shards")
DEFAULT_OUTPUT_DIR = Path("/home/hojinsohn/VLM-NT/PI-VLA/output/runs/exp_fast")
DEFAULT_EPOCHS     = 100
DEFAULT_BATCH      = 512     # can go much higher now — no image loading
DEFAULT_LR         = 3e-4
DEFAULT_VAL_SPLIT  = 0.1
DEFAULT_SEED       = 42
DEFAULT_NUM_WORKERS = 4


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FeatureShardDataset(Dataset):
    """
    Loads pre-extracted feature shards from extract_features.py.
    Each shard contains img_feat, txt_feat, object_location tensors.
    Much faster than raw image shards — no encoder forward pass at runtime.
    """

    def __init__(self, data_dir: Path, max_shards: int = None, preload: bool = False) -> None:
        super().__init__()
        self.data_dir = data_dir

        manifest_path = data_dir / "manifest.pt"
        if manifest_path.exists():
            manifest = torch.load(manifest_path)
            self.shard_files = [Path(p) for p in manifest["shards"]]
            shard_size = int(manifest["shard_size"])
            total      = int(manifest["num_samples"])
            num_shards = len(self.shard_files)
            self._sizes = [shard_size] * (num_shards - 1)
            self._sizes.append(total - shard_size * (num_shards - 1))
        else:
            self.shard_files = sorted(data_dir.glob("shard_*.pt"))
            if not self.shard_files:
                raise FileNotFoundError(f"No shard_*.pt files in {data_dir}")
            self._sizes = []
            for sf in self.shard_files:
                s = torch.load(sf)
                self._sizes.append(int(s["num_samples"]))

        if not self.shard_files:
            raise FileNotFoundError(f"No shards found in {data_dir}")

        if max_shards is not None and max_shards < len(self.shard_files):
            self.shard_files = self.shard_files[:max_shards]
            self._sizes      = self._sizes[:max_shards]
            print(f"  [max_shards={max_shards}] capped to {max_shards} shards "
                  f"({sum(self._sizes)} samples)")

        self._cum: list[int] = []
        running = 0
        for n in self._sizes:
            running += n
            self._cum.append(running)
        self._total = running

        self._cached_idx:   int         = -1
        self._cached_shard: dict | None = None

        # Preload: concatenate all shards into flat tensors in RAM.
        # Feature shards are tiny (~5MB each) so 337 shards ≈ 1.7GB total.
        # This eliminates all disk I/O during training.
        self._preloaded = False
        if preload:
            print("  Preloading all feature shards into RAM...", flush=True)
            img_feats, txt_feats, locs, prompts, names, idxs, sources = [], [], [], [], [], [], []
            for sf in tqdm(self.shard_files, desc='  Loading', leave=False):
                s = torch.load(sf)
                img_feats.append(s['img_feat'])
                txt_feats.append(s['txt_feat'])
                locs.append(s['object_location'])
                prompts.extend(s['prompt'])
                names.extend(s['object_name'])
                idxs.extend(s['object_idx'])
                sources.extend(s['source_file'])
            self._img_feats  = torch.cat(img_feats, dim=0)
            self._txt_feats  = torch.cat(txt_feats, dim=0)
            self._locs       = torch.cat(locs, dim=0)
            self._prompts    = prompts
            self._names      = names
            self._idxs       = idxs
            self._sources    = sources
            self._preloaded  = True
            print(f"  Preloaded {self._total} samples into RAM.", flush=True)

    def __len__(self) -> int:
        return self._total

    def _load_shard(self, shard_idx: int) -> dict:
        if shard_idx != self._cached_idx:
            self._cached_shard = torch.load(self.shard_files[shard_idx])
            self._cached_idx   = shard_idx
        return self._cached_shard  # type: ignore[return-value]

    def __getitem__(self, idx: int) -> dict:
        if self._preloaded:
            return {
                'img_feat':        self._img_feats[idx],
                'txt_feat':        self._txt_feats[idx],
                'object_location': self._locs[idx],
                'prompt':          self._prompts[idx],
                'object_name':     self._names[idx],
                'object_idx':      self._idxs[idx],
                'source_file':     self._sources[idx],
            }
        shard_idx = bisect.bisect_right(self._cum, idx)
        offset    = idx - (self._cum[shard_idx - 1] if shard_idx > 0 else 0)
        shard     = self._load_shard(shard_idx)
        return {
            "img_feat":        shard["img_feat"][offset],         # (2048,)
            "txt_feat":        shard["txt_feat"][offset],         # (512,)
            "object_location": shard["object_location"][offset],  # (3,)
            "prompt":          shard["prompt"][offset],
            "object_name":     shard["object_name"][offset],
        }


def worker_init_fn(worker_id: int) -> None:
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    if hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    dataset._cached_idx   = -1
    dataset._cached_shard = None
    seed = worker_info.seed % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


# ---------------------------------------------------------------------------
# FiLM layer
# ---------------------------------------------------------------------------

class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation.
    output = gamma(cond) * x + beta(cond)
    """
    def __init__(self, cond_dim: int, feat_dim: int) -> None:
        super().__init__()
        self.gamma_proj = nn.Linear(cond_dim, feat_dim)
        self.beta_proj  = nn.Linear(cond_dim, feat_dim)
        nn.init.ones_(self.gamma_proj.weight)
        nn.init.zeros_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.gamma_proj(cond) * x + self.beta_proj(cond)


# ---------------------------------------------------------------------------
# Model  (no encoders — just projection + FiLM + MLP)
# ---------------------------------------------------------------------------

class GraspHead(nn.Module):
    """
    Lightweight head that operates on pre-extracted features.
    All parameters are trained.

    img_feat (2048,) ──► proj ──► (512,) ──► FiLM(txt_feat) ──► MLP ──► (3,)
    txt_feat  (512,) ────────────────────────────────────────┘
    """

    EMBED_DIM = 512

    def __init__(self, img_feat_dim: int = 2048, txt_feat_dim: int = 512) -> None:
        super().__init__()
        self.img_proj = nn.Linear(img_feat_dim, self.EMBED_DIM)
        self.film     = FiLM(cond_dim=txt_feat_dim, feat_dim=self.EMBED_DIM)
        self.head     = nn.Sequential(
            nn.LayerNorm(self.EMBED_DIM),
            nn.Linear(self.EMBED_DIM, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 3),
        )

    def forward(self, img_feat: torch.Tensor, txt_feat: torch.Tensor) -> torch.Tensor:
        x = self.img_proj(img_feat)          # (B, 512)
        x = x / x.norm(dim=-1, keepdim=True)
        x = self.film(x, txt_feat)           # (B, 512)
        return self.head(x)                  # (B, 3)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def mean_l2_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).norm(dim=-1).mean()


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, scaler,
                    epoch, total_epochs) -> dict:
    model.train()
    total_loss = total_l2 = 0.0
    n = 0

    pbar = tqdm(loader, desc=f"Train [{epoch:04d}/{total_epochs-1}]",
                leave=False, dynamic_ncols=True)
    for batch in pbar:
        img_feat = batch["img_feat"].to(device, non_blocking=True)
        txt_feat = batch["txt_feat"].to(device, non_blocking=True)
        targets  = batch["object_location"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                preds = model(img_feat, txt_feat)
                loss  = criterion(preds, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(img_feat, txt_feat)
            loss  = criterion(preds, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        with torch.no_grad():
            l2 = mean_l2_error(preds.float(), targets.float())

        total_loss += loss.item(); total_l2 += l2.item(); n += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", l2=f"{l2.item():.4f}m")

    pbar.close()
    return {"loss": total_loss / n, "l2": total_l2 / n}


@torch.no_grad()
def evaluate(model, loader, criterion, device, epoch, total_epochs) -> dict:
    model.eval()
    total_loss = total_l2 = 0.0
    n = 0

    pbar = tqdm(loader, desc=f"Val   [{epoch:04d}/{total_epochs-1}]",
                leave=False, dynamic_ncols=True)
    for batch in pbar:
        img_feat = batch["img_feat"].to(device, non_blocking=True)
        txt_feat = batch["txt_feat"].to(device, non_blocking=True)
        targets  = batch["object_location"].to(device, non_blocking=True)

        preds = model(img_feat, txt_feat)
        loss  = criterion(preds, targets)
        l2    = mean_l2_error(preds, targets)

        total_loss += loss.item(); total_l2 += l2.item(); n += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", l2=f"{l2.item():.4f}m")

    pbar.close()
    return {"loss": total_loss / n, "l2": total_l2 / n}


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(output_dir, epoch, model, optimizer, scheduler, metrics, is_best):
    ckpt = {
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "metrics":   metrics,
    }
    torch.save(ckpt, output_dir / f"ckpt_epoch{epoch:04d}.pt")
    if is_best:
        torch.save(ckpt, output_dir / "best.pt")


def load_checkpoint(path, model, optimizer=None, scheduler=None) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    print(f"Resumed from {path} (epoch {ckpt['epoch']})")
    return int(ckpt["epoch"]) + 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def main(args):
    set_seed(args.seed)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    # ── Data ──────────────────────────────────────────────────────────
    print("Loading feature dataset...")
    data_dir     = args.data_dir.expanduser().resolve()
    full_dataset = FeatureShardDataset(data_dir, max_shards=args.max_shards, preload=args.preload)
    print(f"  Total samples : {len(full_dataset)}")
    print(f"  Total shards  : {len(full_dataset.shard_files)}")

    val_size   = max(1, int(len(full_dataset) * args.val_split))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}")

    # When preloaded, data is in RAM — workers just slice tensors, no disk I/O
    nw = 0 if args.preload else args.num_workers
    pf = 2 if nw > 0 else None
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=nw, pin_memory=(device.type == "cuda"),
        drop_last=True,
        worker_init_fn=worker_init_fn if nw > 0 else None,
        **({'prefetch_factor': pf} if pf else {}),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=nw, pin_memory=(device.type == "cuda"),
        worker_init_fn=worker_init_fn if nw > 0 else None,
        **({'prefetch_factor': pf} if pf else {}),
    )

    # ── Model ─────────────────────────────────────────────────────────
    print("Building GraspHead (img_proj + FiLM + MLP)...")
    model = GraspHead(img_feat_dim=2048, txt_feat_dim=512).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}  (all trainable)")

    # ── Training ──────────────────────────────────────────────────────
    criterion = nn.SmoothL1Loss(beta=0.01)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler)

    run_cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (output_dir / "config.json").write_text(json.dumps(run_cfg, indent=2))

    history: list[dict] = []
    best_val_l2 = float("inf")

    print(f"\nTraining for {args.epochs} epochs...\n")
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler,
            epoch, args.epochs,
        )
        val_metrics = evaluate(
            model, val_loader, criterion, device, epoch, args.epochs,
        )
        scheduler.step()

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]
        is_best = val_metrics["l2"] < best_val_l2
        if is_best:
            best_val_l2 = val_metrics["l2"]

        row = {
            "epoch":      epoch,
            "train_loss": round(train_metrics["loss"], 6),
            "train_l2":   round(train_metrics["l2"],   6),
            "val_loss":   round(val_metrics["loss"],   6),
            "val_l2":     round(val_metrics["l2"],     6),
            "lr":         lr_now,
            "elapsed_s":  round(elapsed, 1),
        }
        history.append(row)

        print(
            f"[{epoch:04d}/{args.epochs-1}] "
            f"train loss={train_metrics['loss']:.4f} l2={train_metrics['l2']:.4f}m  "
            f"val loss={val_metrics['loss']:.4f} l2={val_metrics['l2']:.4f}m  "
            f"lr={lr_now:.2e}  {elapsed:.1f}s"
            + ("  *** best ***" if is_best else "")
        )

        if (epoch + 1) % args.save_every == 0 or is_best or epoch == args.epochs - 1:
            save_checkpoint(output_dir, epoch, model, optimizer, scheduler, row, is_best)

        (output_dir / "history.json").write_text(json.dumps(history, indent=2))

    print(f"\nBest val L2 : {best_val_l2:.4f} m")
    print(f"Outputs     : {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir",    type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--output-dir",  type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--epochs",      type=int,  default=DEFAULT_EPOCHS)
    p.add_argument("--batch-size",  type=int,  default=DEFAULT_BATCH)
    p.add_argument("--lr",          type=float,default=DEFAULT_LR)
    p.add_argument("--val-split",   type=float,default=DEFAULT_VAL_SPLIT)
    p.add_argument("--num-workers", type=int,  default=DEFAULT_NUM_WORKERS)
    p.add_argument("--seed",        type=int,  default=DEFAULT_SEED)
    p.add_argument("--save-every",  type=int,  default=10)
    p.add_argument("--resume",      type=Path, default=None)
    p.add_argument("--max-shards",  type=int,  default=None,
                   help="Cap shards for testing (None = all)")
    p.add_argument("--preload",     action="store_true",
                   help="Load all feature shards into RAM at startup (~1.7GB). Eliminates disk I/O during training.")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())