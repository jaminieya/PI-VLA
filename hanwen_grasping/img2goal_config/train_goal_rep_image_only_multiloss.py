#!/usr/bin/env python3
"""
Train an image-only student model to predict NTField goal latent z_goal with
multiple losses enabled in a single run (excluding CKA).

Student: image -> z_goal_hat
Teacher target: z_goal from frozen NTField
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.models as models

SCALE = float(np.pi / 0.5)

_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_PI_VLA_ROOT = os.path.dirname(_FILE_DIR)
_NTRL_DEMO = os.path.join(_PI_VLA_ROOT, "ntrl-demo")
if os.path.isdir(_NTRL_DEMO) and _NTRL_DEMO not in sys.path:
    sys.path.insert(0, _NTRL_DEMO)


def _maybe_torchvision():
    try:
        import torchvision.transforms as T  # noqa: WPS433
        return T
    except ImportError:
        return None


def build_coords_batch(q_start: torch.Tensor, q_goal: torch.Tensor, use_scale: bool) -> torch.Tensor:
    if use_scale:
        q_start = q_start / SCALE
        q_goal = q_goal / SCALE
    return torch.cat([q_start, q_goal], dim=1)


class PretrainedImageEncoder(nn.Module):
    """ResNet50 visual encoder with mostly frozen backbone."""

    def __init__(self, out_dim: int = 2048):
        super().__init__()
        try:
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        except AttributeError:
            resnet = models.resnet50(pretrained=True)

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.out_dim = out_dim

        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone[7].parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).flatten(1)


class GoalLatentPredictorImageOnly(nn.Module):
    def __init__(self, ntfield_h: int = 256):
        super().__init__()
        self.img_enc = PretrainedImageEncoder(out_dim=2048)
        self.head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, ntfield_h),
        )

    def forward(self, images_bchw: torch.Tensor) -> torch.Tensor:
        return self.head(self.img_enc(images_bchw))


class H5GraspDemoDataset(Dataset):
    """One sample = (image_i, q_start_i, q_goal_final)."""

    def __init__(
        self,
        h5_paths: List[str],
        image_key: str = "images",
        use_torchvision_resize: bool = True,
        img_size: int = 128,
    ):
        self.samples: List[Tuple[str, int]] = []
        self.image_key = image_key
        self.img_size = img_size
        self._T = _maybe_torchvision()
        self._use_tv = bool(use_torchvision_resize and self._T is not None)
        if self._use_tv:
            self._tfm = self._T.Compose(
                [
                    self._T.ToPILImage(),
                    self._T.Resize((224, 224)),
                    self._T.ToTensor(),
                    self._T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            self._tfm = None

        import h5py

        for path in h5_paths:
            path = os.path.abspath(path)
            try:
                with h5py.File(path, "r") as f:
                    if image_key not in f or "joint_configs" not in f or "final_joint_config" not in f:
                        continue
                    n = f[image_key].shape[0]
                    for i in range(n):
                        self.samples.append((path, i))
            except OSError:
                print(f"[warn] Skipping non-HDF5 or corrupted file: {path}")
                continue

        if not self.samples:
            raise ValueError("No samples found. Check --h5_glob and HDF5 keys.")
        self._file_handles: Dict[str, object] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _image_to_tensor(self, rgb: np.ndarray) -> torch.Tensor:
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        if self._use_tv:
            return self._tfm(rgb)
        h, w = rgb.shape[:2]
        tgt = self.img_size
        if h != tgt or w != tgt:
            ys = (np.linspace(0, h - 1, tgt)).astype(int)
            xs = (np.linspace(0, w - 1, tgt)).astype(int)
            rgb = rgb[np.ix_(ys, xs)]
        return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

    def __getitem__(self, idx: int):
        import h5py

        n = len(self.samples)
        max_retries = min(256, n)
        for step in range(max_retries):
            j = (idx + step * 9973) % n
            path, i = self.samples[j]
            try:
                if path not in self._file_handles:
                    self._file_handles[path] = h5py.File(path, "r")
                f = self._file_handles[path]
                img = np.array(f[self.image_key][i])
                q_start = np.array(f["joint_configs"][i, :6], dtype=np.float32)
                q_goal = np.array(f["final_joint_config"][:6], dtype=np.float32)
            except (OSError, KeyError) as e:
                print(f"[warn] Skipping unreadable sample: file={path} idx={i} error={e}")
                continue
            return self._image_to_tensor(img), torch.from_numpy(q_start), torch.from_numpy(q_goal)
        raise RuntimeError(f"Failed to read sample idx={idx} after {max_retries} attempts.")


def collate_fn(batch):
    imgs, qs, qg = zip(*batch)
    return torch.stack(imgs, 0), torch.stack(qs, 0), torch.stack(qg, 0)


def load_teacher(
    checkpoint: str, device: torch.device, data_path: Optional[str] = None
) -> Tuple[torch.nn.Module, int]:
    from models.metric_arm import model_test_metric as md

    model_path = os.path.dirname(os.path.abspath(checkpoint))
    if data_path is None:
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


def info_nce_loss(z_hat: torch.Tensor, z_tgt: torch.Tensor, temperature: float) -> torch.Tensor:
    b_size = z_hat.shape[0]
    if b_size <= 1:
        return z_hat.new_tensor(0.0)
    z_hat_n = F.normalize(z_hat, dim=1)
    z_tgt_n = F.normalize(z_tgt, dim=1)
    logits = torch.matmul(z_hat_n, z_tgt_n.T) / temperature
    labels = torch.arange(b_size, device=z_hat.device)
    return F.cross_entropy(logits, labels)


def batch_triplet_loss(z_hat: torch.Tensor, z_tgt: torch.Tensor, margin: float) -> torch.Tensor:
    if z_hat.shape[0] <= 1:
        return z_hat.new_tensor(0.0)
    z_hat_n = F.normalize(z_hat, dim=1)
    z_tgt_n = F.normalize(z_tgt, dim=1)
    with torch.no_grad():
        dists = torch.cdist(z_tgt_n, z_tgt_n, p=2)
        dists[dists < 1e-5] = float("inf")
        valid_rows = ~torch.isinf(dists).all(dim=1)
        if not valid_rows.any():
            return z_hat.new_tensor(0.0)
        neg_idx = dists.argmin(dim=1)
    z_hat_n = z_hat_n[valid_rows]
    z_tgt_n = z_tgt_n[valid_rows]
    neg_z = z_tgt_n[neg_idx[valid_rows]]
    return F.triplet_margin_loss(z_hat_n, z_tgt_n, neg_z, margin=margin)


def barlow_twins_loss(z_hat: torch.Tensor, z_tgt: torch.Tensor, lambd: float = 5e-3) -> torch.Tensor:
    b_size = z_hat.shape[0]
    if b_size <= 1:
        return z_hat.new_tensor(0.0)
    z1 = (z_hat - z_hat.mean(0)) / (z_hat.std(0) + 1e-9)
    z2 = (z_tgt - z_tgt.mean(0)) / (z_tgt.std(0) + 1e-9)
    c = (z1.T @ z2) / b_size
    on_diag = (torch.diagonal(c) - 1.0).pow(2).sum()
    off_diag = (c - torch.diag(torch.diagonal(c))).pow(2).sum()
    return on_diag + lambd * off_diag


def vicreg_loss(
    z_hat: torch.Tensor,
    z_tgt: torch.Tensor,
    sim_w: float = 25.0,
    var_w: float = 25.0,
    cov_w: float = 1.0,
) -> torch.Tensor:
    b_size, d = z_hat.shape
    if b_size <= 1:
        return z_hat.new_tensor(0.0)

    sim = F.mse_loss(z_hat, z_tgt)

    std_hat = torch.sqrt(z_hat.var(dim=0) + 1e-4)
    std_tgt = torch.sqrt(z_tgt.var(dim=0) + 1e-4)
    var = torch.mean(F.relu(1.0 - std_hat)) + torch.mean(F.relu(1.0 - std_tgt))

    z_hat_c = z_hat - z_hat.mean(dim=0)
    z_tgt_c = z_tgt - z_tgt.mean(dim=0)
    cov_hat = (z_hat_c.T @ z_hat_c) / (b_size - 1)
    cov_tgt = (z_tgt_c.T @ z_tgt_c) / (b_size - 1)
    off_hat = cov_hat - torch.diag(torch.diagonal(cov_hat))
    off_tgt = cov_tgt - torch.diag(torch.diagonal(cov_tgt))
    cov = (off_hat.pow(2).sum() / d) + (off_tgt.pow(2).sum() / d)

    return sim_w * sim + var_w * var + cov_w * cov


def compute_losses(args, z_hat: torch.Tensor, z_tgt: torch.Tensor) -> Dict[str, torch.Tensor]:
    losses: Dict[str, torch.Tensor] = {}
    losses["mse"] = F.mse_loss(z_hat, z_tgt)
    losses["smooth_l1"] = F.smooth_l1_loss(z_hat, z_tgt, beta=args.smooth_l1_beta)

    z_hat_n = F.normalize(z_hat, dim=1)
    z_tgt_n = F.normalize(z_tgt, dim=1)
    losses["cosine"] = (1.0 - (z_hat_n * z_tgt_n).sum(dim=1)).mean()

    losses["infonce"] = info_nce_loss(z_hat, z_tgt, args.temperature)
    losses["triplet"] = batch_triplet_loss(z_hat, z_tgt, args.triplet_margin)
    losses["barlow"] = barlow_twins_loss(z_hat, z_tgt, lambd=args.barlow_lambda)
    losses["vicreg"] = vicreg_loss(
        z_hat,
        z_tgt,
        sim_w=args.vicreg_sim_w,
        var_w=args.vicreg_var_w,
        cov_w=args.vicreg_cov_w,
    )
    return losses


def weighted_total(args, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
    return (
        args.w_mse * losses["mse"]
        + args.w_smooth_l1 * losses["smooth_l1"]
        + args.w_cosine * losses["cosine"]
        + args.w_infonce * losses["infonce"]
        + args.w_triplet * losses["triplet"]
        + args.w_barlow * losses["barlow"]
        + args.w_vicreg * losses["vicreg"]
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Train image-only NTField student with multi-loss objective")
    p.add_argument("--checkpoint", type=str, required=True, help="NTField Model_Epoch_*.pt")
    p.add_argument("--h5_glob", type=str, required=True, help="Glob for grasp_6dof_demo_*.h5")
    p.add_argument("--out", type=str, default="goal_rep_student_image_only_multiloss.pt")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup_pct", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--image_key", type=str, default="images")
    p.add_argument("--normalize_coords", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Composite objective weights (CKA intentionally excluded).
    p.add_argument("--w_mse", type=float, default=1.0)
    p.add_argument("--w_smooth_l1", type=float, default=0.5)
    p.add_argument("--w_cosine", type=float, default=0.5)
    p.add_argument("--w_infonce", type=float, default=0.2)
    p.add_argument("--w_triplet", type=float, default=0.1)
    p.add_argument("--w_barlow", type=float, default=0.05)
    p.add_argument("--w_vicreg", type=float, default=0.05)

    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--triplet_margin", type=float, default=0.5)
    p.add_argument("--smooth_l1_beta", type=float, default=1.0)
    p.add_argument("--barlow_lambda", type=float, default=5e-3)
    p.add_argument("--vicreg_sim_w", type=float, default=25.0)
    p.add_argument("--vicreg_var_w", type=float, default=25.0)
    p.add_argument("--vicreg_cov_w", type=float, default=1.0)
    args = p.parse_args()

    device = torch.device(args.device)
    paths = sorted(glob.glob(args.h5_glob))
    paths = [os.path.abspath(x) for x in paths if os.path.isfile(x)]
    if not paths:
        print(f"No HDF5 matched: {args.h5_glob}")
        sys.exit(1)
    print(f"Using {len(paths)} HDF5 files")

    ds_full = H5GraspDemoDataset(paths, image_key=args.image_key)
    print(f"Samples {len(ds_full)}")

    n = len(ds_full)
    n_val = max(1, int(0.1 * n))
    indices = np.random.RandomState(0).permutation(n)
    val_list = sorted(indices[:n_val].tolist())
    val_set = set(val_list)
    train_idx = [i for i in range(n) if i not in val_set]

    ds_tr = Subset(ds_full, train_idx)
    ds_va = Subset(ds_full, val_list)

    loader_tr = DataLoader(
        ds_tr,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )
    loader_va = DataLoader(
        ds_va,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    teacher, nt_h = load_teacher(args.checkpoint, device)
    student = GoalLatentPredictorImageOnly(ntfield_h=nt_h).to(device)
    trainable_params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=args.lr,
        steps_per_epoch=len(loader_tr),
        epochs=args.epochs,
        pct_start=args.warmup_pct,
    )

    def teacher_z_goal(qs: torch.Tensor, qg: torch.Tensor) -> torch.Tensor:
        coords = build_coords_batch(qs, qg, args.normalize_coords).to(device)
        with torch.no_grad():
            _, zg = teacher.encode_pair_latents(coords)
        return zg

    out_path = os.path.abspath(args.out)
    loss_keys = ["mse", "smooth_l1", "cosine", "infonce", "triplet", "barlow", "vicreg"]

    for epoch in range(args.epochs):
        student.train()
        tr_sum = {k: 0.0 for k in (["total"] + loss_keys)}
        tr_batches = 0
        for imgs, qs, qg in loader_tr:
            imgs = imgs.to(device)
            qs = qs.to(device)
            qg = qg.to(device)

            z_tgt = teacher_z_goal(qs, qg)
            z_hat = student(imgs)
            losses = compute_losses(args, z_hat, z_tgt)
            total = weighted_total(args, losses)

            opt.zero_grad()
            total.backward()
            opt.step()
            scheduler.step()

            tr_sum["total"] += float(total.item())
            for k in loss_keys:
                tr_sum[k] += float(losses[k].item())
            tr_batches += 1

        student.eval()
        va_sum = {k: 0.0 for k in (["total"] + loss_keys)}
        va_batches = 0
        with torch.no_grad():
            for imgs, qs, qg in loader_va:
                imgs = imgs.to(device)
                qs = qs.to(device)
                qg = qg.to(device)

                z_tgt = teacher_z_goal(qs, qg)
                z_hat = student(imgs)
                losses = compute_losses(args, z_hat, z_tgt)
                total = weighted_total(args, losses)

                va_sum["total"] += float(total.item())
                for k in loss_keys:
                    va_sum[k] += float(losses[k].item())
                va_batches += 1

        tr_avg = {k: tr_sum[k] / max(1, tr_batches) for k in tr_sum}
        va_avg = {k: va_sum[k] / max(1, va_batches) for k in va_sum}
        print(
            (
                f"epoch {epoch + 1}/{args.epochs} "
                f"train_total={tr_avg['total']:.6f} val_total={va_avg['total']:.6f} "
                f"val_mse={va_avg['mse']:.6f} val_cos={va_avg['cosine']:.6f} "
                f"val_s1={va_avg['smooth_l1']:.6f}"
            ),
            flush=True,
        )

        epoch_suffix = f"_epoch{epoch + 1:03d}"
        base, ext = os.path.splitext(out_path)
        epoch_path = base + epoch_suffix + (ext or ".pt")
        payload_epoch = {
            "student_state_dict": student.state_dict(),
            "ntfield_h": nt_h,
            "normalize_coords": args.normalize_coords,
            "image_key": args.image_key,
            "checkpoint_teacher": os.path.abspath(args.checkpoint),
            "epoch": int(epoch + 1),
            "weights": {
                "mse": args.w_mse,
                "smooth_l1": args.w_smooth_l1,
                "cosine": args.w_cosine,
                "infonce": args.w_infonce,
                "triplet": args.w_triplet,
                "barlow": args.w_barlow,
                "vicreg": args.w_vicreg,
            },
            "temperature": args.temperature,
            "triplet_margin": args.triplet_margin,
            "smooth_l1_beta": args.smooth_l1_beta,
            "barlow_lambda": args.barlow_lambda,
            "vicreg": {
                "sim_w": args.vicreg_sim_w,
                "var_w": args.vicreg_var_w,
                "cov_w": args.vicreg_cov_w,
            },
            "train_avg": tr_avg,
            "val_avg": va_avg,
        }
        torch.save(payload_epoch, epoch_path)
        with open(epoch_path + ".json", "w") as f:
            json.dump({k: v for k, v in payload_epoch.items() if k != "student_state_dict"}, f, indent=2)

    payload_final = {
        "student_state_dict": student.state_dict(),
        "ntfield_h": nt_h,
        "normalize_coords": args.normalize_coords,
        "image_key": args.image_key,
        "checkpoint_teacher": os.path.abspath(args.checkpoint),
        "epoch": int(args.epochs),
        "weights": {
            "mse": args.w_mse,
            "smooth_l1": args.w_smooth_l1,
            "cosine": args.w_cosine,
            "infonce": args.w_infonce,
            "triplet": args.w_triplet,
            "barlow": args.w_barlow,
            "vicreg": args.w_vicreg,
        },
        "temperature": args.temperature,
        "triplet_margin": args.triplet_margin,
        "smooth_l1_beta": args.smooth_l1_beta,
        "barlow_lambda": args.barlow_lambda,
        "vicreg": {
            "sim_w": args.vicreg_sim_w,
            "var_w": args.vicreg_var_w,
            "cov_w": args.vicreg_cov_w,
        },
    }
    torch.save(payload_final, out_path)
    with open(out_path + ".json", "w") as f:
        json.dump({k: v for k, v in payload_final.items() if k != "student_state_dict"}, f, indent=2)
    print(f"Saved final student to {out_path}", flush=True)


if __name__ == "__main__":
    main()
