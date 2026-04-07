#!/usr/bin/env python3
"""
Evaluate a goal-latent student by comparing predicted z_goal_hat to the
frozen teacher z_goal on a dataset of grasp demos.

Usage:
  python student_model_evaluation/evaluate_goal_rep_alignment.py \
    --student goal_rep_student_film_epoch005.pt \
    --h5_glob "../collected_data/grasp_6dof_demo_*.h5" \
    --batch_size 16 \
    --device cuda
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import json

# Training code lives in hanwen_grasping/
_ROOT = os.path.dirname(os.path.abspath(__file__))
_HANWEN = os.path.abspath(os.path.join(_ROOT, "..", "hanwen_grasping"))
if _HANWEN not in sys.path:
    sys.path.insert(0, _HANWEN)

import numpy as np
import torch

# Reuse the model and dataset definitions from training.
from train_goal_rep_alignment import (  # noqa: E402
    GoalLatentPredictorWithFiLM,
    H5GraspDemoDataset,
    Subset,
    build_coords_batch,
    collate_fn,
    encode_teacher_joint_latent,
    load_teacher,
    normalize_coords_tensor,
)


def _loss_value(z_hat: torch.Tensor, z_tgt: torch.Tensor, loss_name: str) -> torch.Tensor:
    if loss_name == "mse":
        return torch.nn.functional.mse_loss(z_hat, z_tgt)
    z_hat_n = torch.nn.functional.normalize(z_hat, dim=1)
    z_tgt_n = torch.nn.functional.normalize(z_tgt, dim=1)
    return (1.0 - (z_hat_n * z_tgt_n).sum(dim=1)).mean()


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate goal latent alignment student")
    p.add_argument("--student", type=str, required=True, help="Saved student .pt checkpoint")
    p.add_argument("--h5_glob", type=str, required=True, help="Glob for grasp_6dof_demo_*.h5")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_batches", type=int, default=0, help="0 = no limit")
    p.add_argument(
        "--split",
        type=str,
        choices=["val10", "all"],
        default="val10",
        help="val10 = deterministic 10%% split (same as training), all = full matched dataset",
    )
    p.add_argument(
        "--out_json",
        type=str,
        default="",
        help="Optional path to save evaluation summary JSON",
    )
    args = p.parse_args()

    device = torch.device(args.device)
    student_path = os.path.abspath(args.student)
    payload = torch.load(student_path, map_location=device)

    # Metadata stored by training.
    ntfield_h = int(payload["ntfield_h"])
    normalize_coords = bool(payload.get("normalize_coords", False))
    loss_name = str(payload.get("loss", "mse"))
    image_key = str(payload.get("image_key", "images"))
    teacher_ckpt = str(payload["checkpoint_teacher"])
    start_cond = str(payload.get("start_cond", "joints"))

    student = GoalLatentPredictorWithFiLM(ntfield_h=ntfield_h, start_cond=start_cond).to(device)
    student.load_state_dict(payload["student_state_dict"], strict=True)
    student.eval()

    teacher, _ = load_teacher(teacher_ckpt, device)
    teacher.eval()

    paths = sorted(glob.glob(args.h5_glob))
    paths = [os.path.abspath(x) for x in paths if os.path.isfile(x)]
    if not paths:
        print(f"No HDF5 matched: {args.h5_glob}")
        sys.exit(1)

    print(f"Evaluating on {len(paths)} HDF5 files")
    ds_full = H5GraspDemoDataset(paths, image_key=image_key)

    n = len(ds_full)
    if args.split == "all":
        ds_eval = ds_full
        split_name = "all"
    else:
        n_val = max(1, int(0.1 * n))
        indices = np.random.RandomState(0).permutation(n)
        val_list = sorted(indices[:n_val].tolist())
        ds_eval = Subset(ds_full, val_list)
        split_name = "val10"

    loader_eval = torch.utils.data.DataLoader(
        ds_eval,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    total_selected_loss = 0.0
    batches = 0
    samples = 0
    cosine_sum = 0.0
    cosine_sq_sum = 0.0
    mse_sum = 0.0
    mse_sq_sum = 0.0

    with torch.no_grad():
        for batch_idx, (imgs, prs, qs, qg) in enumerate(loader_eval):
            imgs = imgs.to(device)
            qs = qs.to(device)
            qg = qg.to(device)

            coords = build_coords_batch(qs, qg, normalize_coords).to(device)
            _, z_tgt = teacher.encode_pair_latents(coords)
            if start_cond == "teacher_z":
                qs_n = normalize_coords_tensor(qs, normalize_coords)
                z_in = encode_teacher_joint_latent(teacher, qs_n)
            else:
                z_in = qs
            z_hat = student(imgs, prs, z_in)

            selected_loss = _loss_value(z_hat, z_tgt, loss_name)
            total_selected_loss += float(selected_loss.item())
            batches += 1
            bsz = int(z_hat.shape[0])
            samples += bsz

            z_hat_n = torch.nn.functional.normalize(z_hat, dim=1)
            z_tgt_n = torch.nn.functional.normalize(z_tgt, dim=1)
            cosine_vals = (z_hat_n * z_tgt_n).sum(dim=1)
            mse_vals = ((z_hat - z_tgt) ** 2).mean(dim=1)

            cosine_sum += float(cosine_vals.sum().item())
            cosine_sq_sum += float((cosine_vals ** 2).sum().item())
            mse_sum += float(mse_vals.sum().item())
            mse_sq_sum += float((mse_vals ** 2).sum().item())

            if args.max_batches and batch_idx + 1 >= args.max_batches:
                break

    mean_selected = total_selected_loss / max(1, batches)
    mean_cos = cosine_sum / max(1, samples)
    mean_mse = mse_sum / max(1, samples)
    std_cos = (max(0.0, cosine_sq_sum / max(1, samples) - mean_cos**2)) ** 0.5
    std_mse = (max(0.0, mse_sq_sum / max(1, samples) - mean_mse**2)) ** 0.5

    summary = {
        "student": student_path,
        "teacher_checkpoint": teacher_ckpt,
        "split": split_name,
        "matched_h5_files": len(paths),
        "dataset_samples_total": n,
        "evaluated_samples": samples,
        "evaluated_batches": batches,
        "selected_loss_name": loss_name,
        "selected_loss_mean_over_batches": mean_selected,
        "cosine_similarity_mean": mean_cos,
        "cosine_similarity_std": std_cos,
        "cosine_loss_mean": 1.0 - mean_cos,
        "mse_mean": mean_mse,
        "mse_std": std_mse,
        "normalize_coords": normalize_coords,
        "image_key": image_key,
        "start_cond": start_cond,
    }

    print("=== Goal-Latent Evaluation Summary ===")
    print(f"start_cond: {start_cond}")
    print(f"Split: {split_name}")
    print(f"Matched HDF5 files: {len(paths)}")
    print(f"Evaluated samples/batches: {samples}/{batches}")
    print(f"Mean {loss_name} (batch-avg): {mean_selected:.6f}")
    print(f"Cosine similarity mean/std: {mean_cos:.6f} / {std_cos:.6f}")
    print(f"Cosine loss mean (1-cos): {1.0 - mean_cos:.6f}")
    print(f"MSE mean/std: {mean_mse:.6f} / {std_mse:.6f}")

    if args.out_json:
        out_path = os.path.abspath(args.out_json)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved JSON summary to {out_path}")


if __name__ == "__main__":
    main()

