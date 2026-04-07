#!/usr/bin/env python3
"""
Train a conditional GAN to predict goal joint configuration from image.

Generator:     G(image) -> q_goal_hat (6D)
Discriminator: D(image, q_goal) -> real/fake logit

Training objective:
  - Generator: adversarial BCE + lambda_l1 * L1(q_goal_hat, q_goal)
  - Discriminator: BCE real/fake classification
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.models as models


def _maybe_torchvision():
    try:
        import torchvision.transforms as T  # noqa: WPS433
        return T
    except ImportError:
        return None


class H5ImageGoalDataset(Dataset):
    """One sample = (image_i, q_goal_final[6])."""

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
                    if image_key not in f or "final_joint_config" not in f:
                        continue
                    n = f[image_key].shape[0]
                    for i in range(n):
                        self.samples.append((path, i))
            except OSError:
                print(f"[warn] Skipping non-HDF5/corrupt file: {path}")
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
                q_goal = np.array(f["final_joint_config"][:6], dtype=np.float32)
            except (OSError, KeyError) as e:
                print(f"[warn] Skipping unreadable sample file={path} idx={i}: {e}")
                continue

            x = self._image_to_tensor(img)
            return x, torch.from_numpy(q_goal)

        raise RuntimeError(f"Failed to read sample idx={idx} after {max_retries} attempts.")


class ImageBackbone(nn.Module):
    """ResNet50 backbone with layer4 unfrozen for adaptation."""

    def __init__(self, out_dim: int = 2048):
        super().__init__()
        try:
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        except AttributeError:
            resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.out_dim = out_dim

        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.backbone[7].parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).flatten(1)


class GoalGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_enc = ImageBackbone(out_dim=2048)
        self.head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 6),
        )

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        return self.head(self.img_enc(imgs))


class GoalDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_enc = ImageBackbone(out_dim=2048)
        self.classifier = nn.Sequential(
            nn.Linear(2048 + 6, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, imgs: torch.Tensor, q_goal: torch.Tensor) -> torch.Tensor:
        im = self.img_enc(imgs)
        x = torch.cat([im, q_goal], dim=1)
        return self.classifier(x)


def main() -> None:
    p = argparse.ArgumentParser(description="Train conditional GAN for image -> goal joint config")
    p.add_argument("--h5_glob", type=str, required=True, help="Glob for grasp_6dof_demo_*.h5")
    p.add_argument("--out", type=str, default="img_to_goal_gan.pt")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--image_key", type=str, default="images")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--lr_g", type=float, default=2e-4)
    p.add_argument("--lr_d", type=float, default=2e-4)
    p.add_argument("--lambda_l1", type=float, default=50.0, help="Regression weight for exact goal matching")
    p.add_argument("--label_smoothing", type=float, default=0.1, help="Real label becomes 1-smoothing")
    p.add_argument("--d_steps", type=int, default=1, help="Discriminator updates per generator update")
    args = p.parse_args()

    device = torch.device(args.device)
    paths = sorted(glob.glob(args.h5_glob))
    paths = [os.path.abspath(x) for x in paths if os.path.isfile(x)]
    if not paths:
        raise ValueError(f"No HDF5 matched: {args.h5_glob}")
    print(f"Using {len(paths)} HDF5 files")

    ds_full = H5ImageGoalDataset(paths, image_key=args.image_key)
    print(f"Samples {len(ds_full)}")
    n = len(ds_full)
    n_val = max(1, int(0.1 * n))
    indices = np.random.RandomState(0).permutation(n)
    val_list = sorted(indices[:n_val].tolist())
    val_set = set(val_list)
    train_idx = [i for i in range(n) if i not in val_set]
    ds_tr = Subset(ds_full, train_idx)
    ds_va = Subset(ds_full, val_list)

    loader_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    loader_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    net_g = GoalGenerator().to(device)
    net_d = GoalDiscriminator().to(device)
    opt_g = torch.optim.Adam([p for p in net_g.parameters() if p.requires_grad], lr=args.lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam([p for p in net_d.parameters() if p.requires_grad], lr=args.lr_d, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()

    out_path = os.path.abspath(args.out)

    for epoch in range(args.epochs):
        net_g.train()
        net_d.train()
        tr = {"g_total": 0.0, "g_adv": 0.0, "g_l1": 0.0, "d_loss": 0.0}
        nb = 0

        for imgs, q_real in loader_tr:
            imgs = imgs.to(device)
            q_real = q_real.to(device)
            b = imgs.shape[0]
            real_lbl = torch.full((b, 1), 1.0 - args.label_smoothing, device=device)
            fake_lbl = torch.zeros((b, 1), device=device)

            # Train D
            for _ in range(max(1, args.d_steps)):
                q_fake_detached = net_g(imgs).detach()
                d_real = net_d(imgs, q_real)
                d_fake = net_d(imgs, q_fake_detached)
                d_loss_real = bce(d_real, real_lbl)
                d_loss_fake = bce(d_fake, fake_lbl)
                d_loss = 0.5 * (d_loss_real + d_loss_fake)
                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()

            # Train G
            q_fake = net_g(imgs)
            d_fake_for_g = net_d(imgs, q_fake)
            g_adv = bce(d_fake_for_g, real_lbl)
            g_l1 = F.l1_loss(q_fake, q_real)
            g_total = g_adv + args.lambda_l1 * g_l1
            opt_g.zero_grad()
            g_total.backward()
            opt_g.step()

            tr["g_total"] += float(g_total.item())
            tr["g_adv"] += float(g_adv.item())
            tr["g_l1"] += float(g_l1.item())
            tr["d_loss"] += float(d_loss.item())
            nb += 1

        tr = {k: v / max(1, nb) for k, v in tr.items()}

        # Validation (no adversarial updates, just metrics)
        net_g.eval()
        va_l1 = 0.0
        va_mse = 0.0
        vb = 0
        with torch.no_grad():
            for imgs, q_real in loader_va:
                imgs = imgs.to(device)
                q_real = q_real.to(device)
                q_hat = net_g(imgs)
                va_l1 += float(F.l1_loss(q_hat, q_real).item())
                va_mse += float(F.mse_loss(q_hat, q_real).item())
                vb += 1
        va_l1 /= max(1, vb)
        va_mse /= max(1, vb)

        print(
            f"epoch {epoch + 1}/{args.epochs} "
            f"g_total={tr['g_total']:.6f} g_adv={tr['g_adv']:.6f} g_l1={tr['g_l1']:.6f} "
            f"d_loss={tr['d_loss']:.6f} val_l1={va_l1:.6f} val_mse={va_mse:.6f}",
            flush=True,
        )

        base, ext = os.path.splitext(out_path)
        epoch_path = f"{base}_epoch{epoch + 1:03d}{ext or '.pt'}"
        payload_epoch = {
            "generator_state_dict": net_g.state_dict(),
            "discriminator_state_dict": net_d.state_dict(),
            "epoch": int(epoch + 1),
            "lambda_l1": args.lambda_l1,
            "label_smoothing": args.label_smoothing,
            "d_steps": args.d_steps,
            "image_key": args.image_key,
            "train": tr,
            "val": {"l1": va_l1, "mse": va_mse},
        }
        torch.save(payload_epoch, epoch_path)
        with open(epoch_path + ".json", "w") as f:
            json.dump({k: v for k, v in payload_epoch.items() if not k.endswith("_state_dict")}, f, indent=2)

    payload_final = {
        "generator_state_dict": net_g.state_dict(),
        "discriminator_state_dict": net_d.state_dict(),
        "epoch": int(args.epochs),
        "lambda_l1": args.lambda_l1,
        "label_smoothing": args.label_smoothing,
        "d_steps": args.d_steps,
        "image_key": args.image_key,
    }
    torch.save(payload_final, out_path)
    with open(out_path + ".json", "w") as f:
        json.dump({k: v for k, v in payload_final.items() if not k.endswith("_state_dict")}, f, indent=2)
    print(f"Saved final GAN checkpoint to {out_path}", flush=True)


if __name__ == "__main__":
    main()
