#!/usr/bin/env python3
"""Evaluate trained (x,y) regressor on a held-out directory (e.g. output/segment_anything/test)."""

import argparse
import csv
import os
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import train_objloc_xy as train_mod


@torch.no_grad()
def evaluate_detailed(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> Tuple[float, float, float, float, int]:
    """Returns mae_xy (avg over both dims), rmse_xy, mae_x, mae_y, num_samples."""
    model.eval()
    mse_loss = nn.MSELoss(reduction="sum")
    abs_err_sum = 0.0
    sq_err_sum = 0.0
    abs_x = 0.0
    abs_y = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = model(x)
        pred_f = pred.float()
        y_f = y.float()
        sq_err_sum += mse_loss(pred_f, y_f).item()
        abs_err_sum += torch.abs(pred_f - y_f).sum().item()
        abs_x += torch.abs(pred_f[:, 0] - y_f[:, 0]).sum().item()
        abs_y += torch.abs(pred_f[:, 1] - y_f[:, 1]).sum().item()
        n += y.size(0)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0
    mae_xy = abs_err_sum / (2.0 * n)
    rmse_xy = (sq_err_sum / (2.0 * n)) ** 0.5
    mae_x = abs_x / n
    mae_y = abs_y / n
    return mae_xy, rmse_xy, mae_x, mae_y, n


@torch.no_grad()
def run_predictions(
    model: nn.Module,
    loader: DataLoader,
    dataset: train_mod.ObjLocDataset,
    device: torch.device,
    use_amp: bool,
) -> List[Tuple[str, float, float, float, float]]:
    """Each row: timestamp, gt_x, gt_y, pred_x, pred_y."""
    model.eval()
    out: List[Tuple[str, float, float, float, float]] = []
    idx = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = model(x)
        pred_f = pred.float().cpu().numpy()
        y_f = y.float().numpy()
        bs = x.size(0)
        for i in range(bs):
            s = dataset.samples[idx + i]
            out.append(
                (
                    s.timestamp,
                    float(y_f[i, 0]),
                    float(y_f[i, 1]),
                    float(pred_f[i, 0]),
                    float(pred_f[i, 1]),
                )
            )
        idx += bs
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate obj-loc (x,y) model on test segment directory.")
    p.add_argument(
        "--segment-root",
        type=str,
        default="output/segment_anything/test",
        help="Directory with timestamp sample folders (e.g. test split)",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default="img2objloc_model/checkpoints_xy/best_xy_model.pt",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument(
        "--predictions-csv",
        type=str,
        default=None,
        help="If set, write timestamp,gt_x,gt_y,pred_x,pred_y rows here.",
    )
    args = p.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.device == "auto" else args.device
    )
    use_amp = device.type == "cuda" and not args.no_amp

    ckpt_path = os.path.abspath(args.checkpoint)
    if not os.path.isfile(ckpt_path):
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    backbone = ckpt.get("backbone", "resnet18")
    image_size = int(ckpt.get("image_size", 256))
    image_name = ckpt.get("image_name", "first_image_table_crop.png")
    z_fixed = float(ckpt.get("z_fixed", 0.1))

    samples = train_mod.load_samples(os.path.abspath(args.segment_root), image_name)
    dataset = train_mod.ObjLocDataset(samples, image_size=image_size, train=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = train_mod.build_model(backbone, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    mae_xy, rmse_xy, mae_x, mae_y, n = evaluate_detailed(model, loader, device, use_amp)

    print(f"checkpoint:      {ckpt_path}")
    print(f"segment_root:    {os.path.abspath(args.segment_root)}")
    print(f"backbone:        {backbone} | image_size: {image_size} | image: {image_name}")
    print(f"z_fixed (info):  {z_fixed}  (labels are x,y only; z not predicted)")
    print(f"samples:         {n}")
    print(f"MAE (x,y) avg:   {mae_xy:.6f}")
    print(f"RMSE (x,y) avg:  {rmse_xy:.6f}")
    print(f"MAE x:           {mae_x:.6f}")
    print(f"MAE y:           {mae_y:.6f}")

    if args.predictions_csv:
        rows = run_predictions(model, loader, dataset, device, use_amp)
        out_path = os.path.abspath(args.predictions_csv)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "gt_x", "gt_y", "pred_x", "pred_y"])
            w.writerows(rows)
        print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
