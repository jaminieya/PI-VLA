#!/usr/bin/env python3
"""
train_multi.py

Trains a vision-language model to predict object XYZ location from
a scene image and a natural-language grasp prompt ("grasp banana").

Architecture
------------
  Image encoder  : frozen ResNet-50 (ImageNet pretrained) → (B, 2048) → proj → (B, 512)
  Text encoder   : frozen CLIP ViT-B/32 text tower        → (B, 512)
  Fusion         : FiLM (Feature-wise Linear Modulation)
                   text generates (γ, β) to modulate image features
                   output → MLP → (B, 3)

Why this over plain CLIP?
  - ResNet-50 is ~4× faster than ViT-B/32 for image encoding
  - FiLM is more expressive than concat+MLP: text can selectively
    amplify/suppress image channels ("look at the banana, not the box")
  - CLIP text tower kept because it has strong short-prompt understanding
    and its embeddings are well-structured for object names

Loss: smooth-L1 (Huber) on predicted vs ground-truth XYZ.

Data
----
Loads sharded .pt files written by convert_to_pt.py.
Each shard dict contains:
  "image"            (N, 3, S, S)  float32
  "object_location"  (N, 3)        float32
  "prompt"           list[str]     length N

Usage
-----
  python train_multi.py \
    --data-dir /home/hojinsohn/VLM-NT/PI-VLA/output/multi_obj_layout_shards \
    --output-dir /home/hojinsohn/VLM-NT/PI-VLA/output/runs/exp_resnet_film \
    --epochs 50 --batch-size 64 --lr 3e-4

Requirements
------------
  pip install torch torchvision open_clip_torch tqdm
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
import torchvision.models as tvm
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_DATA_DIR    = Path("/scratch/scholar/sohn31/grasp_dataset_shards_multi")
DEFAULT_OUTPUT_DIR  = Path("./runs/grasp_loc")
DEFAULT_EPOCHS      = 100
DEFAULT_BATCH       = 64
DEFAULT_LR          = 3e-4
DEFAULT_VAL_SPLIT   = 0.1
DEFAULT_SEED        = 42
DEFAULT_NUM_WORKERS = 4
DEFAULT_IMG_SIZE    = 224

TENSOR_KEYS = ("image", "object_location")
META_KEYS   = ("prompt", "object_name", "object_idx", "source_file")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ShardedGraspDataset(Dataset):
    """
    Lazily loads shard_*.pt files written by convert_to_pt.py.
    Worker-safe: worker_init_fn resets cache after fork.
    """

    def __init__(self, data_dir: Path, max_shards: int = None) -> None:
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
                raise FileNotFoundError(f"No shard_*.pt files found in {data_dir}")
            print("Warning: no manifest.pt found, scanning shards for sizes...")
            self._sizes = []
            for sf in self.shard_files:
                shard = torch.load(sf)
                self._sizes.append(int(shard["num_samples"]))

        if not self.shard_files:
            raise FileNotFoundError(f"No shards found in {data_dir}")

        # Optionally cap shards for quick smoke-tests
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

    def __len__(self) -> int:
        return self._total

    def _load_shard(self, shard_idx: int) -> dict:
        if shard_idx != self._cached_idx:
            self._cached_shard = torch.load(self.shard_files[shard_idx])
            self._cached_idx   = shard_idx
        return self._cached_shard  # type: ignore[return-value]

    def __getitem__(self, idx: int) -> dict:
        shard_idx = bisect.bisect_right(self._cum, idx)
        offset    = idx - (self._cum[shard_idx - 1] if shard_idx > 0 else 0)
        shard     = self._load_shard(shard_idx)
        return {
            "image":           shard["image"][offset],
            "object_location": shard["object_location"][offset],
            "prompt":          shard["prompt"][offset],
            "object_name":     shard["object_name"][offset],
            "object_idx":      shard["object_idx"][offset],
            "source_file":     shard["source_file"][offset],
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

    Given conditioning vector c (text) and input x (image features):

        output = gamma(c) * x + beta(c)

    gamma and beta are learned linear projections of c.
    This lets text selectively scale and shift every image feature channel,
    much more expressive than simple concat + MLP.

    Reference: Perez et al., "FiLM: Visual Reasoning with a General
    Conditioning Layer", AAAI 2018.
    """

    def __init__(self, cond_dim: int, feat_dim: int) -> None:
        super().__init__()
        self.gamma_proj = nn.Linear(cond_dim, feat_dim)
        self.beta_proj  = nn.Linear(cond_dim, feat_dim)

        # Init: gamma near 1, beta near 0 → starts as near-identity
        nn.init.ones_(self.gamma_proj.weight)
        nn.init.zeros_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x    : (B, feat_dim)  image features
        cond : (B, cond_dim)  text conditioning
        returns (B, feat_dim)
        """
        gamma = self.gamma_proj(cond)
        beta  = self.beta_proj(cond)
        return gamma * x + beta


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GraspLocationPredictor(nn.Module):
    """
    ResNet-50 image encoder + CLIP text encoder + FiLM fusion + MLP head.

    image  ──► ResNet-50 (frozen) ──► proj ──► (B, 512) ──┐
                                                            ├─► FiLM ──► MLP ──► (B, 3)
    prompt ──► CLIP text (frozen) ──────────────────────── ┘

    Trained parameters: image_proj  +  FiLM (gamma/beta)  +  MLP head
    """

    EMBED_DIM = 512

    def __init__(
        self,
        clip_model_name: str = "ViT-B-32",
        clip_pretrained: str = "openai",
    ) -> None:
        super().__init__()

        # ── Image encoder: ResNet-50 ──────────────────────────────────────
        resnet = tvm.resnet50(pretrained=True)
        # Drop the final FC; keep everything up to avgpool → (B, 2048, 1, 1)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        # Learned projection 2048 → 512
        self.image_proj = nn.Linear(2048, self.EMBED_DIM)

        # ── Text encoder: CLIP text tower ─────────────────────────────────
        try:
            import open_clip
        except ImportError:
            raise ImportError("pip install open_clip_torch")

        clip, _, _ = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=clip_pretrained
        )
        self.tokenizer            = open_clip.get_tokenizer(clip_model_name)
        self.text_transformer     = clip.transformer
        self.token_embedding      = clip.token_embedding
        self.positional_embedding = clip.positional_embedding
        self.ln_final             = clip.ln_final
        self.text_projection      = clip.text_projection
        self.attn_mask            = clip.attn_mask

        # ── Freeze both encoders ──────────────────────────────────────────
        for p in self.image_encoder.parameters():
            p.requires_grad_(False)
        for module in [self.text_transformer, self.token_embedding, self.ln_final]:
            for p in module.parameters():
                p.requires_grad_(False)
        self.text_projection.requires_grad_(False)
        self.positional_embedding.requires_grad_(False)

        # ── Learned: FiLM + head ──────────────────────────────────────────
        self.film = FiLM(cond_dim=self.EMBED_DIM, feat_dim=self.EMBED_DIM)

        self.head = nn.Sequential(
            nn.LayerNorm(self.EMBED_DIM),
            nn.Linear(self.EMBED_DIM, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 3),
        )

    # ------------------------------------------------------------------
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) → (B, 512) L2-normalised."""
        with torch.no_grad():
            feats = self.image_encoder(images)   # (B, 2048, 1, 1)
        feats = feats.flatten(1)                 # (B, 2048)
        feats = self.image_proj(feats)           # (B, 512)  trained
        return feats / feats.norm(dim=-1, keepdim=True)

    def encode_text(self, prompts: list, device: torch.device) -> torch.Tensor:
        """list[str] → (B, 512) L2-normalised."""
        tokens = self.tokenizer(prompts).to(device)
        with torch.no_grad():
            x = self.token_embedding(tokens)
            x = x + self.positional_embedding
            x = x.permute(1, 0, 2)
            attn_mask = self.attn_mask.to(x.device) if self.attn_mask is not None else None
            x = self.text_transformer(x, attn_mask=attn_mask)
            x = x.permute(1, 0, 2)
            x = self.ln_final(x)
            x = x[torch.arange(x.shape[0]), tokens.argmax(dim=-1)]
            x = x @ self.text_projection
        return x / x.norm(dim=-1, keepdim=True)

    def forward(self, images: torch.Tensor, prompts: list) -> torch.Tensor:
        """→ (B, 3) predicted XYZ."""
        device    = images.device
        img_feats = self.encode_image(images)           # (B, 512)
        txt_feats = self.encode_text(prompts, device)   # (B, 512)
        fused     = self.film(img_feats, txt_feats)     # (B, 512)
        return self.head(fused)                         # (B, 3)

    def trainable_parameters(self) -> list:
        return (
            list(self.image_proj.parameters()) +
            list(self.film.parameters()) +
            list(self.head.parameters())
        )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def mean_l2_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).norm(dim=-1).mean()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: GraspLocationPredictor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler,
    epoch: int,
    total_epochs: int,
) -> dict:
    model.train()
    total_loss = 0.0
    total_l2   = 0.0
    n_batches  = 0

    pbar = tqdm(loader, desc=f"Train [{epoch:04d}/{total_epochs-1}]",
                leave=False, dynamic_ncols=True)

    for batch in pbar:
        images  = batch["image"].to(device, non_blocking=True)
        targets = batch["object_location"].to(device, non_blocking=True)
        prompts = batch["prompt"]

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                preds = model(images, prompts)
                loss  = criterion(preds, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(images, prompts)
            loss  = criterion(preds, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), max_norm=1.0)
            optimizer.step()

        with torch.no_grad():
            l2 = mean_l2_error(preds.float(), targets.float())

        total_loss += loss.item()
        total_l2   += l2.item()
        n_batches  += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", l2=f"{l2.item():.4f}m")

    pbar.close()
    return {"loss": total_loss / n_batches, "l2": total_l2 / n_batches}


@torch.no_grad()
def evaluate(
    model: GraspLocationPredictor,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_l2   = 0.0
    n_batches  = 0

    pbar = tqdm(loader, desc=f"Val   [{epoch:04d}/{total_epochs-1}]",
                leave=False, dynamic_ncols=True)

    for batch in pbar:
        images  = batch["image"].to(device, non_blocking=True)
        targets = batch["object_location"].to(device, non_blocking=True)
        prompts = batch["prompt"]

        preds = model(images, prompts)
        loss  = criterion(preds, targets)
        l2    = mean_l2_error(preds, targets)

        total_loss += loss.item()
        total_l2   += l2.item()
        n_batches  += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", l2=f"{l2.item():.4f}m")

    pbar.close()
    return {"loss": total_loss / n_batches, "l2": total_l2 / n_batches}


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    output_dir: Path,
    epoch: int,
    model: GraspLocationPredictor,
    optimizer: torch.optim.Optimizer,
    scheduler,
    metrics: dict,
    is_best: bool,
) -> None:
    ckpt = {
        "epoch":      epoch,
        "image_proj": model.image_proj.state_dict(),
        "film":       model.film.state_dict(),
        "head":       model.head.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "scheduler":  scheduler.state_dict() if scheduler else None,
        "metrics":    metrics,
    }
    torch.save(ckpt, output_dir / f"ckpt_epoch{epoch:04d}.pt")
    if is_best:
        torch.save(ckpt, output_dir / "best.pt")


def load_checkpoint(path: Path, model: GraspLocationPredictor,
                    optimizer=None, scheduler=None) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.image_proj.load_state_dict(ckpt["image_proj"])
    model.film.load_state_dict(ckpt["film"])
    model.head.load_state_dict(ckpt["head"])
    if optimizer and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    print(f"Resumed from {path} (epoch {ckpt['epoch']})")
    return int(ckpt["epoch"]) + 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    output_dir: Path = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device: {device}")
    if device.type == "cuda":
        idx  = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        vis  = os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)")
        print(f"  GPU  : cuda:{idx} → {name!r}  CUDA_VISIBLE_DEVICES={vis}")
        torch.backends.cudnn.benchmark = True

    # ------------------------------------------------------------------ data
    print("Loading dataset index...")
    data_dir     = args.data_dir.expanduser().resolve()
    full_dataset = ShardedGraspDataset(data_dir, max_shards=args.max_shards)
    print(f"  Total samples : {len(full_dataset)}")
    print(f"  Total shards  : {len(full_dataset.shard_files)}")

    val_size   = max(1, int(len(full_dataset) * args.val_split))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}")

    pf = 2 if args.num_workers > 0 else 0
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        drop_last=True,
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
        **({'prefetch_factor': pf} if pf else {}),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
        **({'prefetch_factor': pf} if pf else {}),
    )

    # ----------------------------------------------------------------- model
    print("Building model (ResNet-50 + CLIP text + FiLM)...")
    model = GraspLocationPredictor(
        clip_model_name=args.clip_model,
        clip_pretrained=args.clip_pretrained,
    ).to(device)

    n_trainable = sum(p.numel() for p in model.trainable_parameters())
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable : {n_trainable:,}  /  Total: {n_total:,}")
    print(f"  Trained   : image_proj + FiLM (gamma/beta) + MLP head")

    # --------------------------------------------------------------- training
    criterion = nn.SmoothL1Loss(beta=0.01)
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler)

    run_cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (output_dir / "config.json").write_text(json.dumps(run_cfg, indent=2))

    history:     list[dict] = []
    best_val_l2: float      = float("inf")

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

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data-dir",        type=Path,  default=DEFAULT_DATA_DIR)
    p.add_argument("--output-dir",      type=Path,  default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--epochs",          type=int,   default=DEFAULT_EPOCHS)
    p.add_argument("--batch-size",      type=int,   default=DEFAULT_BATCH)
    p.add_argument("--lr",              type=float, default=DEFAULT_LR)
    p.add_argument("--val-split",       type=float, default=DEFAULT_VAL_SPLIT)
    p.add_argument("--num-workers",     type=int,   default=DEFAULT_NUM_WORKERS)
    p.add_argument("--seed",            type=int,   default=DEFAULT_SEED)
    p.add_argument("--save-every",      type=int,   default=10)
    p.add_argument("--max-shards",      type=int,   default=None,
                   help="Cap number of shards loaded (None=all). Use e.g. 2 to smoke-test.")
    p.add_argument("--resume",          type=Path,  default=None)
    p.add_argument("--clip-model",      type=str,   default="ViT-B-32")
    p.add_argument("--clip-pretrained", type=str,   default="openai")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())