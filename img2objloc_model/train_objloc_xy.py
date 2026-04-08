#!/usr/bin/env python3
import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


@dataclass
class Sample:
    image_path: str
    xy: np.ndarray
    timestamp: str


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_samples(segment_root: str, image_name: str) -> List[Sample]:
    samples: List[Sample] = []
    for entry in sorted(os.listdir(segment_root)):
        sample_dir = os.path.join(segment_root, entry)
        if not os.path.isdir(sample_dir):
            continue
        meta_path = os.path.join(sample_dir, "object_location.json")
        img_path = os.path.join(sample_dir, image_name)
        if not os.path.exists(meta_path) or not os.path.exists(img_path):
            continue
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        loc = meta.get("object_location_z_fixed") or meta.get("object_location_original")
        if loc is None or len(loc) < 2:
            continue
        xy = np.array([loc[0], loc[1]], dtype=np.float32)
        samples.append(Sample(image_path=img_path, xy=xy, timestamp=entry))
    if not samples:
        raise RuntimeError(f"No valid samples found in {segment_root}")
    return samples


class ObjLocDataset(Dataset):
    def __init__(self, samples: List[Sample], image_size: int, train: bool) -> None:
        self.samples = samples
        if train:
            self.tf = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.tf = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s.image_path).convert("RGB")
        x = self.tf(img)
        y = torch.from_numpy(s.xy)
        return x, y


def split_samples(samples: List[Sample], val_ratio: float, seed: int) -> Tuple[List[Sample], List[Sample]]:
    rng = random.Random(seed)
    idxs = list(range(len(samples)))
    rng.shuffle(idxs)
    n_val = max(1, int(len(samples) * val_ratio))
    val_idxs = set(idxs[:n_val])
    train_samples = [samples[i] for i in idxs if i not in val_idxs]
    val_samples = [samples[i] for i in idxs if i in val_idxs]
    return train_samples, val_samples


def build_model(backbone: str, pretrained: bool) -> nn.Module:
    if backbone == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif backbone == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif backbone == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, 2)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    return model


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> Tuple[float, float]:
    model.eval()
    mse_loss = nn.MSELoss(reduction="sum")
    abs_err_sum = 0.0
    sq_err_sum = 0.0
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(x)
            sq_err_sum += mse_loss(pred.float(), y).item()
            abs_err_sum += torch.abs(pred - y).sum().item()
            n += y.numel()
    mae = abs_err_sum / max(1, n)
    rmse = (sq_err_sum / max(1, n)) ** 0.5
    return mae, rmse


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    samples = load_samples(args.segment_root, args.image_name)
    train_samples, val_samples = split_samples(samples, args.val_ratio, args.seed)

    train_ds = ObjLocDataset(train_samples, image_size=args.image_size, train=True)
    val_ds = ObjLocDataset(val_samples, image_size=args.image_size, train=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    eval_bs = args.eval_batch_size if args.eval_batch_size is not None else args.batch_size
    val_loader = DataLoader(
        val_ds, batch_size=eval_bs, shuffle=False, num_workers=args.num_workers
    )

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "auto" else args.device)
    use_amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model = build_model(args.backbone, pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.SmoothL1Loss(beta=0.02)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    accum = max(1, args.grad_accum_steps)

    best_val_mae = float("inf")
    best_path = os.path.join(args.save_dir, "best_xy_model.pt")

    print(f"Total samples: {len(samples)} | Train: {len(train_samples)} | Val: {len(val_samples)}")
    print(f"Device: {device} | AMP: {use_amp} | grad_accum: {accum} | train_bs={args.batch_size} eval_bs={eval_bs}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        optimizer.zero_grad(set_to_none=True)
        for step, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(x)
                loss = criterion(pred, y)
            scaler.scale(loss / accum).backward()
            train_loss_sum += loss.detach().item() * x.size(0)
            train_count += x.size(0)
            if (step + 1) % accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        if (step + 1) % accum != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        train_loss = train_loss_sum / max(1, train_count)
        val_mae, val_rmse = evaluate(model, val_loader, device, use_amp)
        print(
            f"[{epoch:03d}/{args.epochs}] "
            f"train_loss={train_loss:.6f} "
            f"val_mae_xy={val_mae:.6f} "
            f"val_rmse_xy={val_rmse:.6f}"
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "backbone": args.backbone,
                    "image_size": args.image_size,
                    "image_name": args.image_name,
                    "best_val_mae_xy": best_val_mae,
                    "z_fixed": 0.12,
                },
                best_path,
            )
            print(f"  -> saved best model to {best_path}")

    print(f"Training finished. Best val MAE(x,y): {best_val_mae:.6f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train object-location regressor on cropped table images.")
    p.add_argument(
        "--segment-root",
        type=str,
        default="output/segment_anything",
        help="Directory containing timestamp sample directories",
    )
    p.add_argument(
        "--image-name",
        type=str,
        default="first_image_table_crop.png",
        choices=["first_image_table_crop.png", "first_image_table_crop_masked.png"],
        help="Image file inside each timestamp directory",
    )
    p.add_argument("--save-dir", type=str, default="img2objloc_model/checkpoints_xy")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Validation batch size (default: same as --batch-size). Use smaller if val OOM.",
    )
    p.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Accumulate gradients this many steps (effective train batch = batch-size * this).",
    )
    p.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable CUDA automatic mixed precision (uses more VRAM).",
    )
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument(
        "--backbone",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet34", "resnet50"],
    )
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
