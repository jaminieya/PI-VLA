#!/usr/bin/env python3
"""
Train a model to match the NTField **goal-side latent** (the row fed into the
metric head) using **text prompt**, **RGB image**, and **start joint configuration**.

The frozen trajectory NTField defines the teacher:
  z_start, z_goal = network.encode_pair_latents( concat(q_start, q_goal) )

The student predicts z_goal_hat ≈ z_goal from (prompt, image, q_start), so at
inference you can replace an explicit goal configuration with the predicted
latent when building planner inputs (requires a small planner change).

Dataset: ``grasp_6dof_demo_*.h5`` from collect_data (``images``, ``joint_configs``,
``final_joint_config``, attrs ``prompt``).

Usage:
  cd hanwen_grasping
  python train_goal_rep_alignment.py \\
    --checkpoint ../ntrl-demo/Experiments/UR5_trajectory/.../Model_Epoch_04300_ValLoss_*.pt \\
    --h5_glob '../collected_data/grasp_6dof_demo_*.h5' \\
    --epochs 20 --batch_size 16 --out goal_rep_student.pt

Requires: torch, h5py, numpy; torchvision recommended for image resize.
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

# NTField input normalization (same as planning/gradient_planner_trajectory.py)
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


def normalize_coords_tensor(q6: torch.Tensor, use_scale: bool) -> torch.Tensor:
    if use_scale:
        return q6 / SCALE
    return q6


def build_coords_batch(
    q_start: torch.Tensor, q_goal: torch.Tensor, use_scale: bool
) -> torch.Tensor:
    """q_start, q_goal: (B, 6) radians -> (B, 12) teacher input."""
    if use_scale:
        q_start = q_start / SCALE
        q_goal = q_goal / SCALE
    return torch.cat([q_start, q_goal], dim=1)


class CharTextEncoder(nn.Module):
    """Lightweight character-level encoder (no transformers dependency)."""

    def __init__(self, vocab: Dict[str, int], max_len: int, emb_dim: int, out_dim: int):
        super().__init__()
        self.vocab = vocab
        self.max_len = max_len
        self.pad_idx = vocab["<pad>"]
        self.emb = nn.Embedding(len(vocab), emb_dim, padding_idx=self.pad_idx)
        self.net = nn.Sequential(
            nn.Linear(emb_dim * max_len, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_dim),
            nn.ReLU(inplace=True),
        )

    def encode(self, prompts: List[str]) -> torch.Tensor:
        ids = []
        for p in prompts:
            row = []
            for ch in p[: self.max_len]:
                row.append(self.vocab.get(ch, self.vocab["<unk>"]))
            while len(row) < self.max_len:
                row.append(self.pad_idx)
            ids.append(row)
        t = torch.tensor(ids, dtype=torch.long, device=self.emb.weight.device)
        e = self.emb(t).flatten(1)
        return self.net(e)


def build_vocab_from_prompts(prompts: List[str]) -> Dict[str, int]:
    chars = set()
    for p in prompts:
        for ch in p:
            chars.add(ch)
    special = ["<pad>", "<unk>"]
    vocab = {s: i for i, s in enumerate(special)}
    for i, ch in enumerate(sorted(chars)):
        if ch not in vocab:
            vocab[ch] = len(vocab)
    return vocab


class SmallImageEncoder(nn.Module):
    def __init__(self, in_ch: int, latent_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, latent_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x))


class GoalLatentPredictor(nn.Module):
    """
    Predicts NTField goal latent dim H (256 for current UR5 NN) from
    (image features, text features, q_start).
    """

    def __init__(
        self,
        text_vocab: Dict[str, int],
        text_max_len: int,
        ntfield_h: int,
        text_emb_dim: int = 48,
        text_out_dim: int = 128,
        image_out_dim: int = 256,
    ):
        super().__init__()
        self.text_enc = CharTextEncoder(text_vocab, text_max_len, text_emb_dim, text_out_dim)
        self.img_enc = SmallImageEncoder(3, image_out_dim)
        self.q_enc = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
        )
        fused = text_out_dim + image_out_dim + 128
        self.head = nn.Sequential(
            nn.Linear(fused, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, ntfield_h),
        )

    def forward(self, images_bchw: torch.Tensor, prompts: List[str], q_start: torch.Tensor) -> torch.Tensor:
        t = self.text_enc.encode(prompts)
        im = self.img_enc(images_bchw)
        q = self.q_enc(q_start)
        return self.head(torch.cat([t, im, q], dim=1))


class H5GraspDemoDataset(Dataset):
    """One sample = (image_i, prompt, q_start_i, q_goal_final)."""

    def __init__(
        self,
        h5_paths: List[str],
        image_key: str = "images",
        use_torchvision_resize: bool = True,
        img_size: int = 128,
    ):
        self.samples: List[Tuple[str, int]] = []
        self.prompts_per_file: Dict[str, str] = {}
        self.image_key = image_key
        self.img_size = img_size
        self._T = _maybe_torchvision()
        self._use_tv = bool(use_torchvision_resize and self._T is not None)
        if self._use_tv:
            self._tfm = self._T.Compose(
                [
                    self._T.ToPILImage(),
                    self._T.Resize((img_size, img_size)),
                    self._T.ToTensor(),
                ]
            )
        else:
            self._tfm = None

        import h5py

        for path in h5_paths:
            path = os.path.abspath(path)
            with h5py.File(path, "r") as f:
                if image_key not in f or "joint_configs" not in f or "final_joint_config" not in f:
                    continue
                n = f[image_key].shape[0]
                pr = f.attrs.get("prompt", "")
                if isinstance(pr, bytes):
                    pr = pr.decode("utf-8", errors="replace")
                self.prompts_per_file[path] = str(pr)
                for i in range(n):
                    self.samples.append((path, i))

        if not self.samples:
            raise ValueError("No samples found. Check --h5_glob and HDF5 keys (images, joint_configs).")

    def __len__(self) -> int:
        return len(self.samples)

    def _image_to_tensor(self, rgb: np.ndarray) -> torch.Tensor:
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        if self._use_tv:
            return self._tfm(rgb)
        # numpy fallback: mean-pool downsample
        h, w = rgb.shape[:2]
        tgt = self.img_size
        if h != tgt or w != tgt:
            ys = (np.linspace(0, h - 1, tgt)).astype(int)
            xs = (np.linspace(0, w - 1, tgt)).astype(int)
            rgb = rgb[np.ix_(ys, xs)]
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        return t

    def __getitem__(self, idx: int):
        import h5py

        path, i = self.samples[idx]
        with h5py.File(path, "r") as f:
            img = np.array(f[self.image_key][i])
            q_start = np.array(f["joint_configs"][i, :6], dtype=np.float32)
            q_goal = np.array(f["final_joint_config"][:6], dtype=np.float32)
        pr = self.prompts_per_file[path]
        x = self._image_to_tensor(img)
        return x, pr, torch.from_numpy(q_start), torch.from_numpy(q_goal)


def collate_fn(batch):
    imgs, prs, qs, qg = zip(*batch)
    return torch.stack(imgs, 0), list(prs), torch.stack(qs, 0), torch.stack(qg, 0)


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
    # Infer H from first linear after encoder stack
    if hasattr(model.network, "encoder") and len(model.network.encoder) > 0:
        lin = model.network.encoder[-1]
        if hasattr(lin, "out_features"):
            h = int(lin.out_features)
    return model.network, h


def main() -> None:
    p = argparse.ArgumentParser(description="Train goal latent alignment (prompt+image+q_start -> z_goal)")
    p.add_argument("--checkpoint", type=str, required=True, help="NTField Model_Epoch_*.pt")
    p.add_argument("--h5_glob", type=str, required=True, help="Glob for grasp_6dof_demo_*.h5")
    p.add_argument("--out", type=str, default="goal_rep_student.pt")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--image_key", type=str, default="images", help="HDF5 dataset name (images or image)")
    p.add_argument(
        "--normalize_coords",
        action="store_true",
        help="Divide joints by π/0.5 before NTField (matches gradient_planner_trajectory). "
        "Try without first if loss is unstable; trajectory points.npy is often raw radians.",
    )
    p.add_argument(
        "--loss",
        type=str,
        choices=["mse", "cosine"],
        default="mse",
        help="cosine = 1 - cos(z_hat, z_goal); useful if scales drift",
    )
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    paths = sorted(glob.glob(args.h5_glob))
    paths = [os.path.abspath(x) for x in paths if os.path.isfile(x)]
    if not paths:
        print(f"No HDF5 matched: {args.h5_glob}")
        sys.exit(1)
    print(f"Using {len(paths)} HDF5 files")

    ds_full = H5GraspDemoDataset(paths, image_key=args.image_key)
    prompts = [ds_full.prompts_per_file[ds_full.samples[i][0]] for i in range(len(ds_full))]
    vocab = build_vocab_from_prompts(prompts)
    print(f"Vocab size {len(vocab)}, samples {len(ds_full)}")

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
    student = GoalLatentPredictor(vocab, text_max_len=96, ntfield_h=nt_h).to(device)
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)

    def teacher_z_goal(qs: torch.Tensor, qg: torch.Tensor) -> torch.Tensor:
        coords = build_coords_batch(qs, qg, args.normalize_coords).to(device)
        with torch.no_grad():
            _, zg = teacher.encode_pair_latents(coords)
        return zg

    for epoch in range(args.epochs):
        student.train()
        run = 0.0
        n_b = 0
        for imgs, prs, qs, qg in loader_tr:
            imgs = imgs.to(device)
            qs = qs.to(device)
            qg = qg.to(device)
            z_tgt = teacher_z_goal(qs, qg)
            z_hat = student(imgs, prs, qs)
            if args.loss == "mse":
                loss = F.mse_loss(z_hat, z_tgt)
            else:
                z_hat_n = F.normalize(z_hat, dim=1)
                z_tgt_n = F.normalize(z_tgt, dim=1)
                loss = (1.0 - (z_hat_n * z_tgt_n).sum(dim=1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += float(loss.item())
            n_b += 1
        train_loss = run / max(1, n_b)

        student.eval()
        vrun = 0.0
        vb = 0
        with torch.no_grad():
            for imgs, prs, qs, qg in loader_va:
                imgs = imgs.to(device)
                qs = qs.to(device)
                qg = qg.to(device)
                z_tgt = teacher_z_goal(qs, qg)
                z_hat = student(imgs, prs, qs)
                if args.loss == "mse":
                    loss = F.mse_loss(z_hat, z_tgt)
                else:
                    z_hat_n = F.normalize(z_hat, dim=1)
                    z_tgt_n = F.normalize(z_tgt, dim=1)
                    loss = (1.0 - (z_hat_n * z_tgt_n).sum(dim=1)).mean()
                vrun += float(loss.item())
                vb += 1
        val_loss = vrun / max(1, vb)
        print(f"epoch {epoch+1}/{args.epochs}  train_{args.loss}={train_loss:.6f}  val_{args.loss}={val_loss:.6f}")

    payload = {
        "student_state_dict": student.state_dict(),
        "vocab": vocab,
        "ntfield_h": nt_h,
        "normalize_coords": args.normalize_coords,
        "loss": args.loss,
        "image_key": args.image_key,
        "checkpoint_teacher": os.path.abspath(args.checkpoint),
    }
    out_path = os.path.abspath(args.out)
    torch.save(payload, out_path)
    with open(out_path + ".json", "w") as f:
        json.dump({k: v for k, v in payload.items() if k != "student_state_dict"}, f, indent=2)
    print(f"Saved student to {out_path}")


if __name__ == "__main__":
    main()
